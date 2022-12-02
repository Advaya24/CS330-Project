import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets
import numpy as np
import pickle

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

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
    train_data = datasets.CIFAR10('./', download=True)
    test_data = datasets.CIFAR10('./', train=False, download=True)
    train_pct, val_pct = 0.6, 0.2
    test_pct = 1 - train_pct - val_pct
    # num_classes = 100 # cifar-100
    num_classes = 10 # cifar-10
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
            CIFARData(combined_data[idx], combined_targets[idx]).save(os.path.join('cifar10-dataset', class_partition), partition + '.pkl')



def pretrain_create(root = 'cifar-dataset'):
    with open(os.path.join(root, 'seen', 'train.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    labels = np.unique(dataset.targets)
    selected = np.random.choice(labels, 5)
    inds = np.in1d(dataset.targets, selected)
    dataset.targets = dataset.targets[inds]
    dataset.data = dataset.data[inds]
    if not os.path.exists(os.path.join(root, 'seen', 'pretrain.pkl')):
        with open(os.path.join(root, 'seen', 'pretrain.pkl'), 'wb') as f:
            dataset = pickle.dump(dataset, f)


class CIFARDataset(Dataset):
    '''
    CIFAR Dataset, partitioned by seen/new/unseen and sub-partitioned into train/val/test:
    |___ cifar-dataset
        |___ seen
            |___ train (21600 images over 60 classes)
            |___ pretrain (subset containing only 5/60 classes)
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
        self.unique_labels = np.unique(self.labels)
        self.image_shape = self.images[0].shape
        self._is_epsilon = False
        self._is_novel = False
        self.image_type = self.images[0].dtype

    
    def __getitem__(self, index):
        if self._is_epsilon:
            return self.sample_epsilon(self._epsilon_support_size, self._epsilon_query_size, self._epsilon_split)
        elif self._is_novel:
            return self.sample_novel_cls(self._novel_support_size, self._num_novels_in_support, self._novel_query_size, self._novel_split)
        else:
            img = self.images[index]
            label = np.array([self.labels[index]])
            if self.transform is not None:
                img = self.transform(img)
            return img, label
    
    def __len__(self):
        return self.images.shape[0]

    def sample_k_images_per_target(self, k: int=1):
        unique_labels = self.unique_labels
        image_shape = self.image_shape
        out_images = np.zeros((unique_labels.shape[0], k, *image_shape))
        out_labels = np.zeros((unique_labels.shape[0], k))
        for i, lab in enumerate(unique_labels):
            idx = np.where(self.labels == lab)[0]
            k_idx = np.random.choice(idx, k)
            out_images[i] = self.images[k_idx]
            out_labels[i] = lab
        return out_images, out_labels

    def sample_epsilon(self, support_size=5, query_size=15, split=4/6):
        unique_labels = self.unique_labels
        image_shape = self.image_shape
        support_classes = np.random.choice(unique_labels, int(len(unique_labels)*split), replace=False)
        support_images = np.zeros((support_classes.shape[0], support_size, *image_shape), dtype=self.image_type)
        support_labels = np.zeros((support_classes.shape[0], support_size))
        for i, lab in enumerate(support_classes):
            idx = np.where(self.labels == lab)[0]
            k_idx = np.random.choice(idx, support_size)
            support_images[i] = self.images[k_idx]
            support_labels[i] = 0

        ood_classes = np.setdiff1d(unique_labels, support_classes)
        query_images = np.zeros((unique_labels.shape[0], query_size, *image_shape), dtype=self.image_type)
        query_labels = np.zeros((unique_labels.shape[0], query_size))

        for i, lab in enumerate(unique_labels):
            idx = np.where(self.labels == lab)[0]
            k_idx = np.random.choice(idx, query_size)
            query_images[i] = self.images[k_idx]
            query_labels[i] = lab in ood_classes

        support_images = self.transform_batch(support_images.reshape(-1, *image_shape)).reshape(*support_images.shape[:2], *image_shape[::-1])
        query_images = self.transform_batch(query_images.reshape(-1, *image_shape)).reshape(*query_images.shape[:2], *image_shape[::-1])

        return support_images, torch.tensor(support_labels), query_images, torch.tensor(query_labels)

    def transform_batch(self, images):
        out_images = []
        for image in images:
            out_images.append(self.transform(image))
        return torch.stack(out_images)

    def sample_novel_cls(self, support_size=5, num_novels_in_support=1, query_size=15, split=4/6):
        """
        Note: this currently only works for class labels starting from 0.
        labels will look like this: 
        array([2., 3., 0., 5., 1., 4.])

        Instead of masking the labels, we create a mask of shape (num_classes, support_size, 1, 1, 1)
        Finding prototypes needs a couple of steps:
        i) prototypes_shuffled = (support_images*suppport_mask).sum(1) / (suppport_mask.sum(1))
        ii) idx = np.argsort(support_labels)
        iii) prototypes = prototypes_shuffled[idx]
        """
        unique_labels = self.unique_labels
        image_shape = self.images[0].shape
        support_images = np.zeros((unique_labels.shape[0], support_size, *image_shape), dtype=self.image_type)
        support_labels = np.zeros((unique_labels.shape[0], support_size))
        mappings = np.random.permutation(unique_labels)
        for i, lab in enumerate(unique_labels):
            idx = np.where(self.labels == lab)[0]
            k_idx = np.random.choice(idx, support_size)
            support_images[i] = self.images[k_idx]
            support_labels[i] = mappings[lab]

        num_ood = int(unique_labels.shape[0] * (1-split))
        mask_row_idx = np.random.choice(unique_labels, num_ood, replace=False)
        support_mask = np.ones((unique_labels.shape[0], support_size, 1, 1, 1))
        support_mask[mask_row_idx, num_novels_in_support:] = 0

        query_images = np.zeros((unique_labels.shape[0], query_size, *image_shape), dtype=self.image_type)
        query_labels = np.zeros((unique_labels.shape[0], query_size))

        for i, lab in enumerate(unique_labels):
            idx = np.where(self.labels == lab)[0]
            k_idx = np.random.choice(idx, query_size)
            query_images[i] = self.images[k_idx]
            query_labels[i] = mappings[lab]
        

        support_images = self.transform_batch(support_images.reshape(-1, *image_shape)).reshape(*support_images.shape[:2], *image_shape[::-1])
        query_images = self.transform_batch(query_images.reshape(-1, *image_shape)).reshape(*query_images.shape[:2], *image_shape[::-1])

        return support_images, torch.tensor(support_labels), torch.tensor(support_mask), query_images, torch.tensor(query_labels)
    
    def switch_to_epsilon(self, support_size=5, query_size=15, split=4/6):
        self._is_epsilon = True
        self._is_novel = False
        self._epsilon_support_size = support_size
        self._epsilon_query_size = query_size
        self._epsilon_split = split
    
    def switch_to_novel(self, support_size=5, num_novels_in_support=1, query_size=15, split=4/6):
        self._is_epsilon = False
        self._is_novel = True
        self._novel_support_size = support_size
        self._num_novels_in_support = num_novels_in_support
        self._novel_query_size = query_size
        self._novel_split = split

    def switch_to_default(self):
        self._is_epsilon = False
        self._is_novel = False


if __name__ == '__main__':
    download_data()
    pretrain_create()
    
    train_pct, val_pct = 0.6, 0.2
    test_pct = 1 - train_pct - val_pct
    class_partitions = {'seen': (train_pct, {'train': train_pct, 'val': val_pct, 'test': test_pct}), 'new': (val_pct, {'train': train_pct, 'val': val_pct, 'test': test_pct}), 'unseen': (test_pct, {'test': 1})}
    totals = {}
    
    for class_partition, (pct, partitions) in class_partitions.items():
        totals[class_partition] = 0
        
        for partition, part_pct in partitions.items():
            d = CIFARDataset(root='cifar10-dataset', class_partition=class_partition, partition=partition) # load dataset for given specification
            
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
    
    d = CIFARDataset(class_partition='seen', partition='pretrain')
    print('seen pretrain ', 'images:', d.images.shape, 'labels:', d.labels.shape, '\n')
    print(totals)