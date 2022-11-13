import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets
import numpy as np
import pickle

torch.manual_seed(0)

class CIFARData:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    def save(self, path, filename):
        if not os.path.exists(os.path.join(path, filename)):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, filename), 'wb') as f:
                pickle.dump(self, f)

def download_data():
    '''
    Downloads cifar dataset in torch format at cifar-100-python, then re-partitions and saves it into:
    |___ cifar-dataset
        |___ seen
            |___ train (21600 images over 60 classes)
            |___ val (7200 images over 60 classes)
            |___ test (7200 images over 60 classes)
        |___ new
            |___ train (7200 images over 20 classes)
            |___ val (2400 images over 20 classes)
            |___ test (2400 images over 20 classes)
        |___ unseen
            |___ test (12000 images over 20 classes)
    '''
    train_data = datasets.CIFAR100('./', download=True)
    test_data = datasets.CIFAR100('./', train=False, download=True)
    train_pct, val_pct = 0.6, 0.2
    test_pct = 1 - train_pct - val_pct
    num_classes = 100 # cifar-100
    class_partitions = {'seen': (train_pct, {'train': train_pct, 'val': val_pct, 'test': test_pct}), 'new': (val_pct, {'train': train_pct, 'val': val_pct, 'test': test_pct}), 'unseen': (test_pct, {'test': 1})}
    classes = list(range(num_classes))

    for class_partition, (pct, partitions) in class_partitions.items():
        num_cls = int(num_classes*pct)
        cls_ids = classes[:num_cls]
        classes = classes[num_cls:]

        part1_data = train_data.data[np.in1d(train_data.targets, cls_ids)]
        part1_targets = np.array(train_data.targets)[np.in1d(train_data.targets, cls_ids)]
        part2_data = test_data.data[np.in1d(test_data.targets, cls_ids)]
        part2_targets = np.array(test_data.targets)[np.in1d(test_data.targets, cls_ids)]
        combined_data = np.vstack([part1_data, part2_data])
        combined_targets = np.hstack([part1_targets, part2_targets])
        num_instances = combined_data.shape[0]
        sampled_inds = np.random.permutation(num_instances)
        for partition, part_pct in partitions.items():
            sample_size = int(num_instances*part_pct)
            idx = sampled_inds[:sample_size]
            sampled_inds[sample_size:]
            CIFARData(combined_data[idx], combined_targets[idx]).save(os.path.join('cifar-dataset', class_partition), partition + '.pkl')



        


class CIFARDataset(Dataset):
    '''
    CIFAR Dataset, partitioned by seen/new/unseen and sub-partitioned into train/val/test:
    |___ cifar-dataset
        |___ seen
            |___ train (21600 images over 60 classes)
            |___ val (7200 images over 60 classes)
            |___ test (7200 images over 60 classes)
        |___ new
            |___ train (7200 images over 20 classes)
            |___ val (2400 images over 20 classes)
            |___ test (2400 images over 20 classes)
        |___ unseen
            |___ test (12000 images over 20 classes)

    '''
    def __init__(self, root:str='cifar-dataset', class_partition: str='seen', partition: str='train', transform=None) -> None:
        super().__init__()
        if class_partition == 'unseen':
            assert partition == 'test'
        with open(os.path.join(root, class_partition, partition + '.pkl'), 'rb') as f:
            dataset = pickle.load(f)
            self.images = dataset.data
            self.labels = dataset.targets
        self.transform = transform
    
    def __getitem__(self, index):
        img = self.images[index]
        label = np.array([self.labels[index]])
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return self.images.shape[0]

    def sample_k_images_per_target(self, k: int=1):
        unique_labels = np.unique(self.labels)
        image_shape = self.images[0].shape
        out_images = np.zeros((unique_labels.shape[0], k, *image_shape))
        out_labels = np.zeros((unique_labels.shape[0], k))
        for i, lab in enumerate(unique_labels):
            idx = np.where(self.labels == lab)[0]
            k_idx = np.random.choice(idx, k)
            out_images[i] = self.images[k_idx]
            out_labels[i] = lab
        return out_images, out_labels





if __name__ == '__main__':
    download_data()
    
    train_pct, val_pct = 0.6, 0.2
    test_pct = 1 - train_pct - val_pct
    class_partitions = {'seen': (train_pct, {'train': train_pct, 'val': val_pct, 'test': test_pct}), 'new': (val_pct, {'train': train_pct, 'val': val_pct, 'test': test_pct}), 'unseen': (test_pct, {'test': 1})}
    totals = {}
    
    for class_partition, (pct, partitions) in class_partitions.items():
        totals[class_partition] = 0
        
        for partition, part_pct in partitions.items():
            d = CIFARDataset(class_partition=class_partition, partition=partition) # load dataset for given specification
            
            # can now index dataset
            for i in range(1):
                image, label = d[i] 
            
            # or pass into a dataloader
            dl = DataLoader(d)
            for i, (image, label) in enumerate(dl):
                break
            
            # sample k images per label class:
            out_images, out_labels = d.sample_k_images_per_target(5)
            print("Sample k=5 images per label class:", "images: ", out_images.shape, "labels: ", out_labels.shape)


            print(class_partition, partition, 'images:', d.images.shape, 'labels:', d.labels.shape, '\n')
            totals[class_partition] += d.images.shape[0]
    print(totals)