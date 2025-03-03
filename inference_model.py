import os
import torch
import timm
import pandas as pd
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
from hierarchies.tree import Tree
from hierarchies.model import SoftEmbeddedDecisionRules

class_names = ['benign:keratinocytic:actinic cheilitis', 'benign:keratinocytic:cutaneous horn', 'benign:keratinocytic:lichenoid', 
'benign:keratinocytic:porokeratosis', 'benign:keratinocytic:seborrhoeic', 'benign:keratinocytic:solar lentigo', 'benign:keratinocytic:wart',
'benign:melanocytic:acral', 'benign:melanocytic:agminate', 'benign:melanocytic:atypical', 'benign:melanocytic:blue', 'benign:melanocytic:compound', 
'benign:melanocytic:congenital', 'benign:melanocytic:dermal', 'benign:melanocytic:en cockarde', 'benign:melanocytic:ephilides', 'benign:melanocytic:halo', 
'benign:melanocytic:ink spot lentigo', 'benign:melanocytic:involutingregressing', 'benign:melanocytic:irritated', 'benign:melanocytic:junctional', 
'benign:melanocytic:lentiginous', 'benign:melanocytic:lentigo', 'benign:melanocytic:melanosis', 'benign:melanocytic:papillomatous', 'benign:melanocytic:reed nevus', 
'benign:melanocytic:spitzoid', 'benign:melanocytic:traumatized', 'benign:melanocytic:ungual', 'benign:melanocytic:nevus', 'benign:other:accessory nipple', 
'benign:other:chrondrodermatitis', 'benign:other:comedone', 'benign:other:dermatitis', 'benign:other:dermatofibroma', 'benign:other:eczema', 'benign:other:epidermal cyst', 
'benign:other:excoriation', 'benign:other:fibrous papule face', 'benign:other:foliculitis', 'benign:other:granuloma annulare', 'benign:other:molluscum contagiosum', 
'benign:other:myxoid cyst', 'benign:other:nail dystrophy', 'benign:other:psoriasis', 'benign:other:scar', 'benign:other:sebaceous hyperplasia', 'benign:other:skin tag',
'benign:vascular:angiokeratoma', 'benign:vascular:angioma', 'benign:vascular:haematoma', 'benign:vascular:other', 'benign:vascular:telangiectasia', 'malignant:bcc:basal cell carcinoma', 
'malignant:bcc:nodular basal cell carcinoma', 'malignant:bcc:pigmented basal cell carcinoma', 'malignant:bcc:recurrent basal cell carcinoma', 'malignant:bcc:superficial basaal cell carcinoma',
'malignant:keratinocytic:actinic', 'malignant:melanoma:lentigo maligna', 'malignant:melanoma:melanoma', 'malignant:melanoma:nodular melanoma', 'malignant:scc:keratoacanthoma', 
'malignant:scc:scc in situ', 'malignant:scc:squamous cell carcinoma']

class Creat_Model(nn.Module):
    def __init__(self, backbone, embedding_dim=256, num_classes=65, dist='Euclidean'):
        super().__init__()
        self.dist = dist
        self.backbone = backbone
        self.features = timm.create_model(backbone, pretrained=False, features_only=True)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Dropout(0.3),
                                nn.Linear(self.features.feature_info[-1]['num_chs'], embedding_dim),
                                nn.LayerNorm(embedding_dim))
        self.prototypes = nn.Parameter(torch.zeros((num_classes, embedding_dim))).requires_grad_(False)

    def forward(self, image):
        x = self.features(image)
        embedding = self.fc(x[-1])

        if self.dist == 'cosine':
            dists = 1 - nn.CosineSimilarity(dim=-1)(embedding[:, None, :], self.prototypes[None, :, :])
        else:
            dists = torch.norm(embedding[:, None, :] - self.prototypes[None, :, :], dim=-1)

        return -dists, embedding


def load_model(checkpoint_path, backbone='resnet34', embedding_dim=256, num_classes=65):
    model = Creat_Model(backbone=backbone, embedding_dim=embedding_dim, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model, checkpoint['args']


def get_transform(image_size=320):
    transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),     # use ten augmentations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def inference_single(model, image_path, transform, device, tree=None, class_names=None):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs, embeddings = model(image_tensor)

            # Original model prediction
            ori_pred = outputs.argmax(dim=1).item()
            ori_prob = torch.softmax(outputs, dim=1).max().item()

            # Tree-based prediction if tree is provided
            tree_pred = None
            tree_prob = None

            if tree is not None:
                tree_probs = SoftEmbeddedDecisionRules(tree=tree)(embeddings, model.prototypes)
                tree_pred = tree_probs.argmax(dim=1).item()
                tree_prob = tree_probs.max().item()

        results = {
            'image_path': image_path,
            'ori_pred': ori_pred if class_names is None else class_names[ori_pred],
            'ori_prob': ori_prob,
        }

        if tree is not None:
            results.update({
                'tree_pred': tree_pred if class_names is None else class_names[tree_pred],
                'tree_prob': tree_prob

            })

        return results

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {
            'image_path': image_path,
            'error': str(e)
        }


def main():
    # Direct parameter setting
    checkpoint_path = 'model_weight/best_tree_acc_model.pth'
    backbone = 'resnet34'
    embedding_dim = 256
    num_classes = 65
    image_size = 320
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load class mapping from CSV


    # Load test data
    test_data_path = 'xx.csv'
    test_data = pd.read_csv(test_data_path)
    image_list = test_data['img_path']
    image_labels = test_data['img_label']
   

    # Load model
    model, model_args = load_model(
        checkpoint_path,
        backbone=backbone,
        embedding_dim=embedding_dim,
        num_classes=num_classes
    )
    model = model.to(device)

    # Get transform
    transform = get_transform(image_size)

    # Set tree to None (can load if needed)
    tree = None
    path_graph = 'hierarchies/graphs/molemap/graph-preset-noprune.json'
    path_wnids = 'hierarchies/wnids/molemap.txt'
    tree = Tree('molemap', path_graph=path_graph, path_wnids=path_wnids, classes=class_names, hierarchy='preset-noprune')


    # Process all images and collect results
    all_results = []
    ori_predictions = []
    tree_predictions = []

    for idx, (img_path, img_label) in enumerate(zip(image_list, image_labels)):
        print(f"Processing image {idx + 1}/{len(image_list)}: {img_path}")
        result = inference_single(model, img_path, transform, device, tree, class_names)

        # Add pathology diagnosis to result
        result['label'] = pathology_diagnosis

        # Store predictions for later binary conversion
        if 'error' not in result:
            ori_predictions.append(result['ori_pred'])
            if tree is not None:
                tree_predictions.append(result['tree_pred'])

        all_results.append(result)

        # Print current result
        print(f"Results for {img_path}:")
        for k, v in result.items():
            if k != 'image_path':
                print(f"  {k}: {v}")



if __name__ == '__main__':
    main()