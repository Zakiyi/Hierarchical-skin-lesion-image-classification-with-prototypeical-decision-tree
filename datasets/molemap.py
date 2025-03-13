from fastai.vision import *
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
from datasets.augmentation import Augmentations


class InverseNormalize:
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)[None, :, None, None]
        self.std = torch.Tensor(std)[None, :, None, None]

    def __call__(self, sample):
        return (sample * self.std) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class MolemapDataset(Dataset):
    def __init__(self, p, df, size=64, is_train=True, test_mode=False, debug=False, return_img_dir=False):
        self.p = p
        self.return_img_dir = return_img_dir
        l0, l1, l2 = [], [], []
        for l in df.label:
            parts = l.split(':')
            l0.append(parts[0])
            l1.append(':'.join(parts[0:2]))
            l2.append(l)

        self.level_0, self.level_1, self.level_2 = sorted(set(l0)), sorted(set(l1)), sorted(set(l2))
        self.classes = self.level_2

        # all classes
        l02i = {label: i for i, label in enumerate(self.level_0)}
        l12i = {label: i for i, label in enumerate(self.level_1)}
        l22i = {label: i for i, label in enumerate(self.level_2)}
        self.num_per_class = [len(df[df.label == c]) for c in self.level_2]
        # print(l12i)
        df.loc[:, "level0"] = df.label.apply(lambda x: l02i[x.split(":")[0]])
        df.loc[:, "level1"] = df.label.apply(lambda x: l12i[':'.join(x.split(":")[0:2])])
        df.loc[:, "level2"] = df.label.apply(lambda x: l22i[x])

        self.paths = list(df.name)
        if debug:
            self.paths = self.paths[:300]
        self.size = size
        self.is_train = is_train
        self.transform = Augmentations(size, is_train)
        print('total samples number is: ', len(self.paths))
        
        if not test_mode:
            # split the csv data (is_train==1) into training and validation
            val_index = np.linspace(0, len(self.paths), len(self.paths) // 5, endpoint=False, dtype=int)
            train_index = np.setdiff1d(np.arange(len(self.paths)), val_index)

            if is_train:
                self.images = np.array(self.paths)[train_index]
                self.labels = np.array(df.level2)[train_index]
                print('training samples number is: ', len(self.images))
            else:
                self.images = np.array(self.paths)[val_index]
                self.labels = np.array(df.level2)[val_index]
                print('validation samples number is: ', len(self.images))
        else:
            self.images = np.array(self.paths)
            self.labels = np.array(df.level2)
            print('test samples number is: ', len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # dealing with the image
        img_dir = self.images[idx]

        img = PIL.Image.open(os.path.join(self.p, img_dir)).convert('RGB')
        img = self.transform(img)
        tgt = torch.tensor(self.labels[idx], dtype=torch.int64)

        if self.return_img_dir:
            return img, tgt, img_dir #, idx
        else:
            return img, tgt

    def get_img_dir(self, idx):
        return self.images[idx]

    def get_img_label(self, idx):
        _,  label = self.__getitem__(idx)
        return self.level_2[label.numpy()]

    def get_img_index(self, img_name):
        return np.where(self.images == img_name)


    def show(self, idx):
        x, y = self.__getitem__(idx)
        stds = np.array([0.229, 0.224, 0.225])
        means = np.array([0.485, 0.456, 0.406])
        img = ((x.numpy().transpose((1, 2, 0)) * stds + means) * 255).astype(np.uint8)
        plt.imshow(img)
        plt.title("ground truth: {}-{}".format(y.item(), self.level_2[y.item()]))

    @staticmethod
    def transform_val_inverse():
        return InverseNormalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )


if __name__ == '__main__':
    df = pd.read_csv('molemap_dataset.csv')

    df_test = df[df.is_train == 0]
    testset = MolemapDataset(p, df_test, size=320, is_train=False, test_mode=True)
