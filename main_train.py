import os
import sys
import copy
import wandb
import time
import timm
import torch
import argparse
import pprint
import warnings
import shutil
import yaml
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
from util.util import AverageMeter
from PIL import ImageFile
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn.functional as F
from timm.scheduler import create_scheduler
from util.util import generate_wandb_name
from util import loss_nbdt
from hierarchies import analysis, metrics
from hierarchies import tree as T
from hierarchies.utils import progress_bar, generate_kwargs, Colors
from hierarchies.tree import Tree
from timm.models.layers import trunc_normal_
from datasets.molemap import MolemapDataset
from hierarchies.model import HardEmbeddedDecisionRules, SoftEmbeddedDecisionRules
from models.prototype.metrics.distortion import Pseudo_Huber
from models.prototype.metrics.distortion import Eucl_Mat, Cosine_Mat

DATASETS = (
    "CIFAR10",
    "molemap"
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DistortionLoss(nn.Module):
    """Scale-free squared distortion regularizer"""

    def __init__(self, D, dist="euclidian", scale_free=True):
        super(DistortionLoss, self).__init__()
        self.D = D
        self.scale_free = scale_free
        if dist == "euclidian":
            self.dist = Eucl_Mat()
        elif dist == "cosine":
            self.dist = Cosine_Mat()

    def forward(self, mapping, idxs=None):
        d = self.dist(mapping)

        if self.scale_free:
            a = d / (self.D + torch.eye(self.D.shape[0], device=self.D.device))
            scaling = a.sum() / torch.pow(a, 2).sum()
        else:
            scaling = 1.0

        d = (scaling * d - self.D) ** 2 / (self.D + torch.eye(self.D.shape[0], device=self.D.device)) ** 2
        d = d.sum() / (d.shape[0] ** 2 - d.shape[0])
        return d, scaling


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T)
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, dim, mrg=0.5, alpha=0.):
        torch.nn.Module.__init__(self)
        self.nb_classes = nb_classes
        self.dim = dim
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, similarity, T, prototypes=None):
        true_sim = similarity.gather(dim=-1, index=T.unsqueeze(1))
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes).to(similarity.device)
        # The set of positive proxies of data in the batch
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        if prototypes is None:
            norm = np.sqrt(self.dim)
            pos_exp = torch.exp(-true_sim - self.mrg * norm).sum()
            loss = torch.log(1 + pos_exp) / num_valid_proxies 
        else:
            norm = prototypes.norm(p=2, dim=-1)[T].unsqueeze(1).detach()
            pos_exp = torch.exp(-true_sim - self.mrg * norm).sum()
            loss = torch.log(1 + pos_exp) / num_valid_proxies   #  similarity.size(0)
        return loss


class Creat_Model(nn.Module):
    def __init__(self, backbone, embedding_dim=256, num_classes=65, train_backbone=True, dist='Euclidean', dropout=0.2,
                 squarred=False, lws=False):
        super().__init__()
        self.dist = dist
        self.lws = lws
        self.squarred = squarred
        self.backbone = backbone
        if 'vit' in backbone:
            self.features = timm.create_model(backbone, pretrained=True, num_classes=0, proj_drop_rate=dropout, drop_path_rate=dropout,
            global_pool=''   # This prevents global pooling
            )
            self.features.norm = nn.Identity()
            self.fc = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(self.features.embed_dim, embedding_dim),
                                    nn.LayerNorm(embedding_dim))
        else:
            # resnet
            self.features = timm.create_model(backbone, pretrained=True, features_only=True)

            self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten(),
                                    nn.Dropout(0.3),
                                    nn.Linear(self.features.feature_info[-1]['num_chs'], embedding_dim),
                                    nn.LayerNorm(embedding_dim))

        self.scales = nn.Parameter(torch.ones(num_classes).unsqueeze(1)).requires_grad_(True)
        self.prototypes = nn.Parameter(torch.zeros((num_classes, embedding_dim))).requires_grad_(True)

        trunc_normal_(self.prototypes, std=1.)

        if not train_backbone:
            print('we do not train backbone!!!')
            for parameter in self.features.parameters():
                parameter.requires_grad_(False)

    def forward(self, image):
        if 'vit' in self.backbone:
            x = self.features.forward_features(image)
            embedding = self.fc(x[:, 0]) 
        else:
            x = self.features(image)
            embedding = self.fc(x[-1])

        if self.dist == 'cosine':
            dists = 1 - nn.CosineSimilarity(dim=-1)(embedding[:, None, :], self.prototypes[None, :, :])
        else:
            dists = torch.norm(embedding[:, None, :] - self.prototypes[None, :, :], dim=-1)

        if self.squarred:
            dists = dists ** 2

        return -dists, embedding


def set_seed(seed=2022):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def main(args):
    set_seed(2022)
    torch.cuda.empty_cache()
    pprint.pprint(args.__dict__)

    args = parser.parse_args()
    loss_nbdt.set_default_values(args)

    best_ori_acc = 0  # best test accuracy
    best_tree_acc = 0
    best_loss = 100.
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    print('train ', Path(__file__).parent.absolute())
    # load data
    print('Load data from data dir: {} and csv file: {}'.format(args.data_dir, args.csv_file))
    p = Path(args.data_dir)

    if args.debug:
        print(list(p.iterdir())[:2])

    df = pd.read_csv(args.csv_file)
    if args.debug:
        print(df.head())

    df_train = df[df.is_train == 1]
    df_valid = df[df.is_train == 0]

    if args.debug:
        print(df_train.head())
        print(df_valid.head())
   
    # for train and validation, we use the same df_train but split it inside dataset function
    
    # training data
    trainset = MolemapDataset(p, df_train, size=args.image_size, is_train=True, test_mode=False, debug=args.debug)
    # validation set
    testset = MolemapDataset(p, df_train, size=args.image_size, is_train=False, test_mode=False, debug=args.debug)
    # testset = MolemapDataset(p, df_test, size=args.image_size, is_train=False, test_mode=True, debug=args.debug)

    train_dl = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dl = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    tree = Tree.create_from_args(args, classes=trainset.classes)

    # Initialize the model
    print("==> Building model..")
    args.num_classes = (len(trainset.level_0), len(trainset.level_1), len(trainset.level_2))
    args.classes = trainset.classes
    print('total classes is {}'.format(len(trainset.level_2)))
    model = Creat_Model(backbone=args.backbone, num_classes=len(trainset.level_2), embedding_dim=args.embedding_dim, dropout=args.dropout)

    model.to(args.device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # checkpoint_fname = generate_checkpoint_fname(**vars(args))
    if args.debug:
        args.log_dir = 'runs/debug'
    else:
        args.log_dir = os.path.join(args.log_dir, 'proto_' + generate_wandb_name(**vars(args)) + '-alpha-{}-256-analysis'.format(args.alpha))

        if os.path.exists(args.log_dir):
            args.log_dir = args.log_dir + f'_{datetime.now().strftime("%Y_%m%d_%H%M")}'

    os.makedirs(args.log_dir, exist_ok=True)

    if args.debug_count == 0:
        checkpoint_path = os.path.join(args.log_dir, 'checkpoints')
        os.makedirs(checkpoint_path, exist_ok=True)
        print(f"==> Checkpoints will be saved to: {checkpoint_path}")

    if args.metric_guided:
        print('==> using metric guided prototype learning!!!')

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "features" not in n and 'prototypes' not in n]},
        {"params": [p for n, p in model.named_parameters() if 'prototypes' in n], 'lr': args.lr},
        {"params": [p for n, p in model.named_parameters() if "features" in n and p.requires_grad], "lr": args.lr_backbone},
    ]

    # Define the loss function
    criterion = None
    for _loss in args.loss:
        if criterion is None and not hasattr(nn, _loss):
            criterion = nn.CrossEntropyLoss()

        class_criterion = getattr(loss_nbdt, _loss)
        loss_kwargs = generate_kwargs(
            args,
            class_criterion,
            name=f"Loss {args.loss}",
            globals=locals(),
        )
        criterion = class_criterion(**loss_kwargs)

    distance_matrix = torch.tensor(torch.load(args.class_distance, weights_only=False)).to(args.device)
    dist_criterion = DistortionLoss(D=distance_matrix)
    # used to control the inter/inner class distance
    proxy_criterion = Proxy_Anchor(nb_classes=len(args.classes), dim=args.embedding_dim, mrg=args.mrg, alpha=args.alpha)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    wandb_name = generate_wandb_name(**vars(args))

    print(wandb_name)
    if not args.debug:
        wandb.init(name=f'proto-{wandb_name}-alpha-{args.alpha}',
                   project="Hierarchical_skin_lesion_classification",
                   notes=" ",
                   tags=['HPDT', "prototype"],
                   config=args
                   )

    file = open(os.path.join(args.log_dir, "config_file.yml"), "w")
    yaml.dump(args.__dict__, file)
    shutil.copy(os.path.realpath(__file__), args.log_dir)

    # load from checkpoint
    if args.resume is True:
        print('==> Resume from checkpoint {}...'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print(args.__dict__)
        Colors.cyan(f"==> Checkpoint found for epoch {checkpoint['epoch']} at {args.resume}")

    class_analysis = getattr(analysis, args.analysis or "Noop")
    analyzer_kwargs = generate_kwargs(
        args,
        class_analysis,
        name=f"Analyzer {args.analysis}",
        globals=locals(),
    )
    analyzer = class_analysis(**analyzer_kwargs)
    metric = getattr(metrics, args.metric)()
    best_model_weights = {}
    best_model_weights['models'] = copy.deepcopy(model.state_dict())

    # Training
    @analyzer.train_function
    def train(epoch, data_dl, optimizer, lr_scheduler):
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch, args.epochs)

        print("\nEpoch: %d / LR: %f" % (epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        # if 'SoftTreeSupLoss' in args.loss:
        tree_acc = AverageMeter(acc=True)
        tree_loss = AverageMeter()
        ce_loss = AverageMeter()
        intra_dist_loss = AverageMeter()
        dist_loss = AverageMeter()
        
        intra_dist = AverageMeter()
        model.train()
        train_loss = 0
        metric.clear()
        for batch_idx, (inputs, targets) in enumerate(data_dl):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs, embeddings = model(inputs)
            if 'SoftTreeSupLoss' in args.loss:
                loss = criterion(outputs, targets, embeddings, model.prototypes)
            else:
                loss = criterion(outputs, targets)

            d_loss, scale = dist_criterion(model.prototypes)
            p_loss = proxy_criterion(outputs, targets, prototypes=model.prototypes)
            loss = loss + args.alpha * p_loss

            if args.metric_guided:
                loss = loss + args.beta*d_loss

            loss.backward()
            optimizer.step()
            
            intra_class_dist = -outputs.gather(dim=-1, index=targets.unsqueeze(1)).detach().mean()
            intra_dist.update(intra_class_dist.item(), outputs.size(0))
            
            train_loss += loss.item()
            metric.forward(outputs, targets)

            if 'SoftTreeSupLoss' in args.loss:
                tree_loss.update(nn.NLLLoss()(torch.log(criterion.rules(embeddings, model.prototypes)), targets).item(), outputs.size(0))
                tree_acc.update(torch.sum(criterion.rules(embeddings, model.prototypes).argmax(dim=-1) == targets.data).double(), outputs.size(0))
            else:
                tree_loss.update(nn.NLLLoss()(torch.log(SoftEmbeddedDecisionRules(tree=tree)(embeddings, model.prototypes)), targets).item(), outputs.size(0))
                tree_acc.update(torch.sum(SoftEmbeddedDecisionRules(tree=tree)(embeddings, model.prototypes).argmax(dim=-1) == targets.data).double(), outputs.size(0))
                
            ce_loss.update(nn.CrossEntropyLoss()(outputs, targets).item(), outputs.size(0))
            dist_loss.update(d_loss.item(), outputs.size(0))
            intra_dist_loss.update(p_loss.item(), outputs.size(0))

            transform = trainset.transform_val_inverse().to(args.device)
            stat = analyzer.update_batch(embeddings, targets, transform(inputs), model.prototypes)

            progress_bar(
                batch_idx,
                len(train_dl),
                "Train loss: %.3f | Acc: %.3f%% (%d/%d) %s"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * metric.report(),
                    metric.correct,
                    metric.total,
                    f"| {analyzer.name}: {stat}" if stat else "",
                ),
            )

        lr_scheduler.step(epoch)

        Colors.red(f"Ori acc is {round(100 * metric.report(), 3)}, Tree acc is {round(100 * tree_acc.avg.item(), 3)}")

        if not args.debug:
            wandb.log({'train ce_tree_loss': train_loss / (batch_idx + 1),
                       'train ce_acc': 100.0 * metric.report()}, step=epoch)

            # if 'SoftTreeSupLoss' in args.loss:
            wandb.log({'train_d_loss': dist_loss.avg,
                       'train tree_acc': tree_acc.avg,
                       'train tree_loss': tree_loss.avg,
                       'train ce_loss': ce_loss.avg,
                       'train anchor_loss': intra_dist_loss.avg,
                       'train_intra_dist': intra_dist.avg}, step=epoch)

    @analyzer.test_function
    def test(epoch, data_dl, checkpoint=True):
        nonlocal best_ori_acc
        nonlocal best_tree_acc
        nonlocal best_loss
        model.eval()
        test_loss = 0
        metric.clear()
        # if 'SoftTreeSupLoss' in args.loss:
        tree_acc = AverageMeter(acc=True)
        tree_loss = AverageMeter()
        ce_loss = AverageMeter()
        dist_loss = AverageMeter()
        intra_dist_loss = AverageMeter()
        intra_dist = AverageMeter()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_dl):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs, embeddings = model(inputs)
                d_loss, scale = dist_criterion(model.prototypes)
                if not args.disable_test_eval:
                    if 'SoftTreeSupLoss' in args.loss:
                        loss = criterion(outputs, targets, embeddings, model.prototypes)
                    else:
                        loss = criterion(outputs, targets)

                    p_loss = proxy_criterion(outputs, targets, prototypes=model.prototypes)

                    loss = loss + args.alpha * p_loss

                    if args.metric_guided:
                        loss = loss + args.beta*d_loss

                    test_loss += loss.item()
                    metric.forward(outputs, targets)

                transform = testset.transform_val_inverse().to(args.device)
                stat = analyzer.update_batch(embeddings, targets, transform(inputs), model.prototypes)
                if 'SoftTreeSupLoss' in args.loss:
                    tree_loss.update(nn.NLLLoss()(torch.log(criterion.rules(embeddings, model.prototypes)), targets).item(), outputs.size(0))
                    tree_acc.update(torch.sum(criterion.rules(embeddings, model.prototypes).argmax(dim=-1) == targets.data).double(), outputs.size(0))
                else:
                    tree_loss.update(nn.NLLLoss()(torch.log(SoftEmbeddedDecisionRules(tree=tree)(embeddings, model.prototypes)), targets).item(), outputs.size(0))
                    tree_acc.update(torch.sum(SoftEmbeddedDecisionRules(tree=tree)(embeddings, model.prototypes).argmax(
                        dim=-1) == targets.data).double(), outputs.size(0))
                

                intra_class_dist = -outputs.gather(dim=-1, index=targets.unsqueeze(1)).detach().mean()
            
                intra_dist.update(intra_class_dist.item(), outputs.size(0))
                ce_loss.update(nn.CrossEntropyLoss()(outputs, targets).item(), outputs.size(0))
                intra_dist_loss.update(p_loss.item(), outputs.size(0))
                dist_loss.update(d_loss.item(), outputs.size(0))
                
                progress_bar(
                    batch_idx,
                    len(test_dl),
                    "Valid loss: %.3f | Acc: %.3f%% (%d/%d) %s"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * metric.report(),
                        metric.correct,
                        metric.total,
                        f"| {analyzer.name}: {stat}" if stat else "",
                    ),
                )

        Colors.red(f"Ori acc is {round(100 * metric.report(), 3)}, Tree acc is {round(100 * tree_acc.avg.item(), 3)}")
        if not args.debug:
            wandb.log({'valid ce_tree_loss': test_loss / (batch_idx + 1),
                       'valid ce_acc': 100.0 * metric.report()}, step=epoch)

            wandb.log({'valid_d_loss': dist_loss.avg,
                       'valid tree_acc': tree_acc.avg,
                       'valid tree_loss': tree_loss.avg,
                       'valid ce_loss': ce_loss.avg,
                       'valid anchor_loss': intra_dist_loss.avg,
                       'valid_intra_dist': intra_dist.avg,
                       'scale': scale.item()}, step=epoch)

        ori_acc = 100.0 * metric.report()
        tree_acc = 100.0 * tree_acc.avg
        test_loss = test_loss / (batch_idx + 1)
        if test_loss < best_loss and checkpoint:
            Colors.green(f"Saving to {checkpoint_path} ({ori_acc})..")
            state = {
                'args': args,
                "ori_acc": ori_acc,
                "tree_acc": tree_acc,
                "epoch": epoch,
                "loss": loss,
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }
            torch.save(state, checkpoint_path + '/best_loss_model.pth')
            best_loss = test_loss

        if ori_acc > best_ori_acc and checkpoint:
            Colors.green(f"Saving to {checkpoint_path} ({ori_acc})..")
            state = {
                'args': args,
                "ori_acc": ori_acc,
                "tree_acc": tree_acc,
                "epoch": epoch,
                "loss": loss,
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }
            torch.save(state, checkpoint_path + '/best_ori_acc_model.pth')
            best_ori_acc = ori_acc

        if tree_acc > best_tree_acc:
            Colors.green(f"Saving to {checkpoint_path} ({tree_acc})..")
            state = {
                'args': args,
                "ori_acc": ori_acc,
                "tree_acc": tree_acc,
                "epoch": epoch,
                "loss": loss,
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }
            torch.save(state, checkpoint_path + '/best_tree_acc_model.pth')
            best_tree_acc = tree_acc
            best_model_weights['models'] = copy.deepcopy(model.state_dict())

        # Save checkpoint.
        print("Accuracy: {}, {}/{} | Best Original Accurracy: {}".format(ori_acc, metric.correct, metric.total,
                                                                         best_ori_acc))

    if args.eval:
        if not args.resume and not args.pretrained:
            Colors.red(
                " * Warning: Model is not loaded from checkpoint. "
                "Use --resume or --pretrained (if supported)"
            )
        with analyzer.epoch_context(0):
            test(0, test_dl, checkpoint=False)
    else:
        for epoch in range(args.start_epoch, args.epochs):
            with analyzer.epoch_context(epoch):
                train(epoch, train_dl, optimizer, lr_scheduler)
                test(epoch, test_dl)

    print(f"Best tree accuracy: {best_tree_acc} // Checkpoint name: {checkpoint_path + '/best_tree_acc_model.pth'}")
    print(f"Best ori accuracy: {best_ori_acc} // Checkpoint name: {checkpoint_path + '/best_ori_acc_model.pth'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train data with CNN-Transformer model')
    # data
    parser.add_argument("--dataset", default="molemap", choices=DATASETS)
    parser.add_argument('--data-dir', default='datasets/resize512_MIC_POL', help='data directory')
    parser.add_argument('--data-type', default='MIC.POL', help='data type {MIC.POL, MAC}')
    parser.add_argument('--csv-file', default='molemap/img2targets_d4_MIC_PNG.POL.csv', help='path to csv file')
    parser.add_argument('--class-distance', default='molemap/molemap_class_dis_2.pt', help='class distance')
    parser.add_argument('--log-dir', default='runs/exp/hpdt_molemap_prototypes/vit_hierarchy_npj', help='where to store results')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='device {cuda:0, cpu}')

    # model setting
    parser.add_argument('--backbone', default='vit_base_patch16_384', type=str, help='name of backbone network')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--train-backbone', default=True, type=bool, help='train backbone')
    parser.add_argument('--embedding-dim', default=256, type=int, help='dim of embedding from backbone network')

    parser.add_argument('--lws', default=False, action='store_true', help='turn on debug mode')
    parser.add_argument('--metric-guided', default=False, action='store_true', help='turn on debug mode')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training')
    parser.add_argument("--eval", help="eval only", action="store_true")
    parser.add_argument("--disable-test-eval",
                        help="Allows you to run model inference on a test dataset,  different from train dataset. Use an anlayzer to define a metric.",
                        action="store_true")
    # options specific to this project and its dataloaders
    parser.add_argument("--loss", choices=loss_nbdt.names, default=["CrossEntropyLoss"], nargs="+")
    parser.add_argument("--metric", choices=metrics.names, default="top1")
    parser.add_argument("--analysis", choices=analysis.names, help="Run analysis after each epoch")

    # training
    parser.add_argument('--batch-size', type=int, default=72, help='batch size')
    parser.add_argument('--image-size', type=int, default=384, help='image size to the model')
    parser.add_argument('--num-workers', type=int, default=16, help='number of loader workers')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='start at epoch')
    parser.add_argument('--log-steps', type=int, default=100, help='start at epoch')

    # warm-up scheduler
    parser.add_argument('--alpha', type=float, default=0.4, help='factor for icd loss')
    parser.add_argument('--beta', type=float, default=0.5, help='factor for guided loss')
    parser.add_argument('--mrg', type=float, default=0.0, help='parameters for ramp_down function')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER', help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument('--lr-backbone', type=float, default=3e-5, help='learning rate for backbone')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=3e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--decay-epochs', type=float, default=10, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.04, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # other
    parser.add_argument('--debug', default=False, action='store_true', help='turn on debug mode')
    parser.add_argument('--debug-count', type=int, default=0, help='# of minibatchs for fast testing, 0 to disable')
    T.add_arguments(parser)
    loss_nbdt.add_arguments(parser)
    analysis.add_arguments(parser)
    main(parser.parse_args())


