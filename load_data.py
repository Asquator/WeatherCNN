import os

import pandas as pd
import torch
import torchvision.io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

dataset_path = 'dataset_weather'
target_categories = ['fogsmog', 'rain', 'snow', 'rime', 'lightning', 'sandstorm', 'shine', 'sunrise', 'cloudy']

'''
class ToRGB():
    def __call__(self, sample):
        if sample.size(0) == 1:
            sample = sample.repeat(3, 1, 1)

        elif sample.size(0) >= 4:
            sample = sample[:3]

        return sample
'''


class ImageDataset(Dataset):
    default_size = (260, 400)

    default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(default_size, antialias=True),
        transforms.Normalize(mean=[0.5324, 0.5285, 0.5349], std=[0.2546, 0.2426, 0.2710]),
    ])

    def __init__(self, img_dir, transform=default_transform, target_transform=None, image_size=default_size):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.data = self.collect_images()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def img_size(self):
        return self.image_size

    def collect_images(self):
        img_paths = []
        labels = []
        for i_category, category in enumerate(target_categories):
            img_dir_path = os.path.join(self.img_dir, category)
            new_img_names = os.listdir(img_dir_path)
            new_paths = [os.path.join(img_dir_path, name) for name in new_img_names if
                         name.endswith(('.jpg', '.png', '.jpeg'))]
            img_paths += new_paths
            labels += [i_category] * len(new_paths)

        df = pd.DataFrame(data={'path': img_paths, 'label': labels})
        return df


def get_dataset():
    return ImageDataset(dataset_path)


def compute_stats(dataset):
    loader = DataLoader(ds,
                        batch_size=400,
                        shuffle=False)

    img_size = dataset.img_size()
    n_total_pixels = len(dataset) * img_size[0] * img_size[1]

    r_sum = g_sum = b_sum = 0
    r_variance_sum = g_variance_sum = b_variance_sum = 0

    for batch, _ in loader:
        r_sum += batch[:, 0].sum()
        g_sum += batch[:, 1].sum()
        b_sum += batch[:, 2].sum()

    mean_r = r_sum / n_total_pixels
    mean_g = g_sum / n_total_pixels
    mean_b = b_sum / n_total_pixels

    for batch, _ in loader:
        r_variance_sum += ((batch[:, 0] - mean_r).pow(2)).sum()
        g_variance_sum += ((batch[:, 1] - mean_g).pow(2)).sum()
        b_variance_sum += ((batch[:, 2] - mean_b).pow(2)).sum()

    mean = torch.tensor([mean_r, mean_g, mean_b])
    std = torch.tensor([torch.sqrt(r_variance_sum / n_total_pixels),
                        torch.sqrt(g_variance_sum / n_total_pixels),
                        torch.sqrt(b_variance_sum / n_total_pixels)])

    return mean, std

# code section that computes mean and variance for the dataset
'''
if __name__ == '__main__':
    ds = get_dataset()
    
    print(compute_stats(ds))

    
    loader = DataLoader(ds,
                        batch_size=32,
                        shuffle=False)

    for data in loader:
        print(data[0])
'''
