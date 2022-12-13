"""Implementation of prototypical networks for Omniglot."""

import argparse
from dis import dis
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard
# from torch.utils.data import DataLoader
from torchvision import transforms

import util
from data.cifardata import CIFARDataset, CIFARData
from pretraining.networks.resnet_big import SupConResNet
from tqdm import tqdm
import time

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
ENC_SAVE_INTERVAL = 250
THRESHOLD_SAVE_INTERVAL = 50
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TEST_TASKS = 600


class ProtoNetNetwork(nn.Module):
    """Container for ProtoNet weights and image-to-latent computation and the threshold for ood detection."""

    def __init__(self, pre_trained_encoder, prototypes=None, prototype_labels=None):
        """Inits ProtoNetNetwork.
        """
        super().__init__()
        self.encoder = pre_trained_encoder
        self.threshold = torch.nn.Parameter(torch.randn(1))
        self.to(DEVICE)
        self._prototypes = prototypes
        self._prototype_labels = prototype_labels

    def embed_forward(self, images):
        """Computes the latent representation of a batch of images.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)

        Returns:
            a Tensor containing a batch of latent representations
                shape (num_images, latents)
        """
        return self.encoder(images)

    def ood_forward(self):
        return self.threshold

    def forward(self, images):
        return self.embed_forward(images), self.ood_forward()
    
    def add_prototypes(self, prototypes, labels):
        """
        Take given prototypes and add them to the existing prototype array. Requires resizing.
        Precondition: labels are disjoin from self._prototype_labels
        """
        if self._prototypes is None:
            # create new prototype array
            self._prototypes = prototypes
            self._prototype_labels = labels
        else:
            # old_shape_0 = self._prototypes.shape[0]
            # new_shape_0 = old_shape_0 + labels.shape[0]
            # new_prototypes = torch.zeros((new_shape_0, self._prototypes.shape[1]))
            # new_labels = torch.zeros(new_shape_0)
            # new_prototypes[:old_shape_0] = self._prototypes
            # new_prototypes[old_shape_0:] = prototypes
            # new_labels[:old_shape_0] = self._prototype_labels
            # new_labels[old_shape_0:] = labels
            # self._prototypes = new_prototypes
            # self._prototype_labels = new_labels
            self._prototypes = torch.cat((self._prototypes, prototypes), dim=0)
            self._labels = torch.cat((self._prototype_labels, labels), dim=0)


    


class ProtoNetTrainer:
    """Trains and assesses a prototypical network and ood detection."""

    def __init__(self, encoder, proto_learning_rate, threshold_learning_rate, log_dir):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
        """

        self._network = ProtoNetNetwork(encoder)
        self.encoder_optimizer = torch.optim.Adam(
            self._network.encoder.parameters(),
            lr=proto_learning_rate
        )
        self.threshold_optimizer = torch.optim.Adam(
            [self._network.threshold],
            lr=threshold_learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_enc_step = 1
        self._start_train_thres_step = 1

    def embd_train_step(self, task, print_=False):
        """Computes ProtoNet mean loss (and accuracy) on a batch of tasks.

        Args:
            task_batch (tuple[Tensor, Tensor, Tensor, Tensor]):
                batch of tasks from an Omniglot DataLoader

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        images_support, labels_support, mask_support, images_query, labels_query = task
        images_support = images_support.to(DEVICE)
        labels_support = labels_support.to(DEVICE)
        images_query = images_query.to(DEVICE)
        labels_query = labels_query.type(torch.LongTensor).to(DEVICE)
        mask_support = mask_support.to(DEVICE)
        
        num_classes = images_support.shape[0]
        num_support = images_support.shape[1]

        mask_support = torch.reshape(mask_support, (mask_support.shape[0], mask_support.shape[1], 1))

        latents_support = self._network.embed_forward(torch.reshape(images_support, (-1, images_support.shape[2], images_support.shape[3], images_support.shape[4])))
        latents_query = self._network.embed_forward(torch.reshape(images_query, (-1, images_query.shape[2], images_query.shape[3], images_query.shape[4])))

        latents_support_reshaped = torch.reshape(latents_support, (num_classes, num_support, -1))
        prototypes_shuffled = torch.sum(latents_support_reshaped * mask_support, axis = 1) / torch.sum(mask_support, axis = 1)
        idx = torch.argsort(labels_support.mean(1).int())
        prototypes = prototypes_shuffled[idx]
        
        distances_query = torch.stack([torch.linalg.vector_norm(prototypes - x, dim = -1) for x in latents_query])

        logits_query = - torch.square(distances_query)
        distances_support = torch.stack([torch.linalg.vector_norm(prototypes - x, dim = -1) for x in latents_support])
        logits_support = - torch.square(distances_support)
        labels_query = labels_query.reshape(-1, *labels_query.shape[2:])
        labels_support = labels_support.reshape(-1, *labels_support.shape[2:])



        return (
            F.cross_entropy(logits_query, labels_query),
            util.score(logits_support, labels_support),
            util.score(logits_query, labels_query, print_)
        )

    def threshold_train_step(self, task, print_=False):

        images_support, labels_support, images_query, labels_query = task
        images_support = images_support.to(DEVICE)
        labels_support = labels_support.to(DEVICE)
        images_query = images_query.to(DEVICE)
        labels_query = labels_query.float().to(DEVICE)

        num_seen_classes = images_support.shape[0]
        num_support = images_support.shape[1]

        with torch.no_grad():
            latents_support = self._network.embed_forward(torch.reshape(images_support, (-1, images_support.shape[2], images_support.shape[3], images_support.shape[4])))
            latents_query = self._network.embed_forward(torch.reshape(images_query, (-1, images_query.shape[2], images_query.shape[3], images_query.shape[4])))

            latents_support_reshaped = torch.reshape(latents_support, (num_seen_classes, num_support, -1))
            prototypes = torch.mean(latents_support_reshaped, 1)
            distances_query = torch.stack([torch.linalg.vector_norm(prototypes - x, dim = 1) for x in latents_query])
            min_dist_query = torch.min(distances_query, dim = 1)[0]

            distances_support = torch.stack([torch.linalg.vector_norm(prototypes - x, dim = 1) for x in latents_support])
            min_dist_support = torch.min(distances_support, dim = 1)[0]
            
            labels_query = labels_query.reshape(-1, *labels_query.shape[2:])
            labels_support = labels_support.reshape(-1, *labels_support.shape[2:])
        
        prob_ood_query = torch.sigmoid(min_dist_query - self._network.ood_forward())
        prob_ood_support = torch.sigmoid(min_dist_support - self._network.ood_forward())

        return (F.binary_cross_entropy_with_logits(min_dist_query - self._network.ood_forward(), labels_query, pos_weight=torch.tensor(6.0/4)),
        util.bin_score(prob_ood_support, labels_support),
        util.bin_score(prob_ood_query, labels_query, print_)
        )

    def train(self, dataloader_train, dataloader_val, writer, args):
        # train encoder
        print("===========================")
        print("     Training Encoder:")
        print("===========================")
        dataloader_train.dataset.switch_to_novel(args.num_support_novel, args.num_shots_novel, args.num_query_novel, args.seen_unseen_split)
        dataloader_val.dataset.switch_to_novel(args.num_support_novel, args.num_shots_novel, args.num_query_novel, args.seen_unseen_split)
        self.train_encoder(dataloader_train, dataloader_val, writer, args.num_encoder_train_iterations, args.batch_size)

        # train epsilon
        print("===========================")
        print("     Training Epsilon:")
        print("===========================")
        dataloader_train.dataset.switch_to_epsilon(args.num_support_epsilon, args.num_query_epsilon, args.seen_unseen_split)
        dataloader_val.dataset.switch_to_epsilon(args.num_support_epsilon, args.num_query_epsilon, args.seen_unseen_split)
        self.train_threshold(dataloader_train, dataloader_val, writer, args.num_threshold_train_iterations, args.batch_size)

        # calculate and store prototypes
        print("===========================")
        print("  Calculating Prototypes:")
        print("===========================")
        dataloader_train.collate_fn = torch.utils.data.default_collate
        dataloader_train.dataset.switch_to_default()
        dataloader_val.collate_fn = torch.utils.data.default_collate
        dataloader_val.dataset.switch_to_default()
        self.calculate_prototypes(dataloader_train)
        self._save('', 'prototypes')

    def train_encoder(self, dataloader_train, dataloader_val, writer, num_train_iterations, batch_size):
        """Train the ProtoNet.

        Consumes dataloader_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_enc_step}.')
        for i_step in tqdm(range(self._start_train_enc_step, num_train_iterations+1)):
            self.encoder_optimizer.zero_grad()
            loss_batch = []
            accuracy_support_batch = []
            accuracy_query_batch = []
            # data_start = time.time()
            task_batch = next(iter(dataloader_train))
            # print(f'Data time: {time.time() - data_start}')
            for task in task_batch:
                # step_start = time.time()
                loss_task, accuracy_support_task, accuracy_query_task = self.embd_train_step(task)
                # print(f'Step time: {time.time() - step_start}')
                loss_batch.append(loss_task)
                accuracy_query_batch.append(accuracy_query_task)
                accuracy_support_batch.append(accuracy_support_task)
            loss, accuracy_support, accuracy_query = (torch.mean(torch.stack(loss_batch)),
                                                        np.mean(accuracy_support_batch),
                                                        np.mean(accuracy_query_batch))
            loss.backward()
            self.encoder_optimizer.step()

            writer.add_scalar('enc_loss/train', loss.item(), i_step)
            writer.add_scalar(
                'enc_train_accuracy/support',
                accuracy_support.item(),
                i_step
            )
            writer.add_scalar(
                'enc_train_accuracy/query',
                accuracy_query.item(),
                i_step
            )
            if i_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                    f'support accuracy: {accuracy_support.item():.3f}, '
                    f'query accuracy: {accuracy_query.item():.3f}'
                )

            if i_step % VAL_INTERVAL == 0:
                with torch.no_grad():
                    losses, accuracies_support, accuracies_query = [], [], []
                    for _ in range(4):

                        loss_batch = []
                        accuracy_support_batch = []
                        accuracy_query_batch = []
                        task_batch = next(iter(dataloader_val))
                        for task in task_batch:
                            loss_task, accuracy_support_task, accuracy_query_task = self.embd_train_step(task)
                            loss_batch.append(loss_task)
                            accuracy_query_batch.append(accuracy_query_task)
                            accuracy_support_batch.append(accuracy_support_task)
                        loss, accuracy_support, accuracy_query = (torch.mean(torch.stack(loss_batch)),
                                                        np.mean(accuracy_support_batch),
                                                        np.mean(accuracy_query_batch))
                        losses.append(loss.item())
                        accuracies_support.append(accuracy_support)
                        accuracies_query.append(accuracy_query)
                    loss = np.mean(losses)
                    accuracy_support = np.mean(accuracies_support)
                    accuracy_query = np.mean(accuracies_query)
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'support accuracy: {accuracy_support:.3f}, '
                    f'query accuracy: {accuracy_query:.3f}'
                )
                writer.add_scalar('enc_loss/val', loss, i_step)
                writer.add_scalar(
                    'enc_val_accuracy/support',
                    accuracy_support,
                    i_step
                )
                writer.add_scalar(
                    'enc_val_accuracy/query',
                    accuracy_query,
                    i_step
                )

            if i_step % ENC_SAVE_INTERVAL == 0:
                self._save(i_step, aux_string = 'encd')

    def train_threshold(self, dataloader_train, dataloader_val, writer, num_train_iterations, batch_size):
        """Train the ProtoNet.

        Consumes dataloader_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_thres_step}.')
        for i_step in tqdm(range(self._start_train_thres_step, num_train_iterations+1)):
            self.encoder_optimizer.zero_grad()
            loss_batch = []
            accuracy_support_batch = []
            accuracy_query_batch = []
            # data_start = time.time()
            task_batch = next(iter(dataloader_train))
            # print(f'Data time: {time.time() - data_start}')
            for task in task_batch:
                # step_start = time.time()
                loss_task, accuracy_support_task, accuracy_query_task = self.threshold_train_step(task, print_=False)
                # print(f'Step time: {time.time() - step_start}')
                loss_batch.append(loss_task)
                accuracy_query_batch.append(accuracy_query_task)
                accuracy_support_batch.append(accuracy_support_task)
            loss, accuracy_support, accuracy_query = (torch.mean(torch.stack(loss_batch)),
                                                        np.mean(accuracy_support_batch),
                                                        np.mean(accuracy_query_batch))
            loss.backward()
            self.threshold_optimizer.step()

            writer.add_scalar('thresh_loss/train', loss.item(), i_step)
            writer.add_scalar(
                'thresh_train_accuracy/support',
                accuracy_support.item(),
                i_step
            )
            writer.add_scalar(
                'thresh_train_accuracy/query',
                accuracy_query.item(),
                i_step
            )
            if i_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                    f'support accuracy: {accuracy_support.item():.3f}, '
                    f'query accuracy: {accuracy_query.item():.3f}'
                )

            if i_step % VAL_INTERVAL == 0:
                with torch.no_grad():
                    losses, accuracies_support, accuracies_query = [], [], []
                    for _ in range(4):

                        loss_batch = []
                        accuracy_support_batch = []
                        accuracy_query_batch = []
                        task_batch = next(iter(dataloader_val))
                        for task in task_batch:
                            loss_task, accuracy_support_task, accuracy_query_task = self.threshold_train_step(task, print_=False)
                            loss_batch.append(loss_task)
                            accuracy_query_batch.append(accuracy_query_task)
                            accuracy_support_batch.append(accuracy_support_task)
                        loss, accuracy_support, accuracy_query = (torch.mean(torch.stack(loss_batch)),
                                                        np.mean(accuracy_support_batch),
                                                        np.mean(accuracy_query_batch))
                        losses.append(loss.item())
                        accuracies_support.append(accuracy_support)
                        accuracies_query.append(accuracy_query)
                    loss = np.mean(losses)
                    accuracy_support = np.mean(accuracies_support)
                    accuracy_query = np.mean(accuracies_query)
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'support accuracy: {accuracy_support:.3f}, '
                    f'query accuracy: {accuracy_query:.3f}'
                )
                writer.add_scalar('thresh_loss/val', loss, i_step)
                writer.add_scalar(
                    'thresh_val_accuracy/support',
                    accuracy_support,
                    i_step
                )
                writer.add_scalar(
                    'thresh_val_accuracy/query',
                    accuracy_query,
                    i_step
                )

            if i_step % THRESHOLD_SAVE_INTERVAL == 0:
                self._save(i_step, aux_string = 'thres')
################################################ TODO ##############################################
    # def test(self, dataloader_test):
    #     """Evaluate the ProtoNet on test tasks.

    #     Args:
    #         dataloader_test (DataLoader): loader for test tasks
    #     """
    #     accuracies = []
    #     for task_batch in dataloader_test:
    #         accuracies.append(self._step(task_batch)[2])
    #     mean = np.mean(accuracies)
    #     std = np.std(accuracies)
    #     mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
    #     print(
    #         f'Accuracy over {NUM_TEST_TASKS} test tasks: '
    #         f'mean {mean:.3f}, '
    #         f'95% confidence interval {mean_95_confidence_interval:.3f}'
    #     )

    def load(self, checkpoint_step, aux_string, num_encoder_train_iterations):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{aux_string}{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state['network_state_dict'])
            self.encoder_optimizer.load_state_dict(state['encoder_optimizer_state_dict'])
            self.threshold_optimizer.load_state_dict(state['threshold_optimizer_state_dict'])
            if aux_string == 'encd':
                self._start_train_enc_step = checkpoint_step + 1
            else:
                self._start_train_enc_step = num_encoder_train_iterations + 1
                self._start_train_thres_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {aux_string}{checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step, aux_string=''):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 encoder_optimizer_state_dict=self.encoder_optimizer.state_dict(),
                 threshold_optimizer_state_dict=self.threshold_optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{aux_string}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')
    
    def calculate_prototypes(self, dataloader):
        self._network.eval()
        with torch.no_grad():
            unique_labels = dataloader.dataset.unique_labels
            image_shape = np.flip(dataloader.dataset.image_shape)
            emb_dim = self._network.encoder(torch.randn((1, *image_shape), device=DEVICE)).shape[-1]
            prototypes = torch.zeros((unique_labels.shape[0], emb_dim), device=DEVICE)
            labels = torch.tensor(unique_labels, device=DEVICE)
            counts = torch.zeros_like(labels, dtype=int, device=DEVICE)
            for batch_images, batch_labels in dataloader:
                batch_images = batch_images.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                encs = self._network.encoder(batch_images)
                idx = labels.unsqueeze(0) == batch_labels
                idx = idx.int().argmax(1)
                prototypes.index_put_((idx,), encs, accumulate=True)
                counts.put_(idx, torch.ones_like(idx), accumulate=True)
            self._network.add_prototypes(prototypes/counts.unsqueeze(1), labels)



def main(args):
    log_dir = args.log_dir
    log_dir = '/home/advaya/CS330-Project/logs/protonet/cifar10.support_eps:5.query_eps:15.support_novel:5.query_novel:15.eps_lr:0.01.novel_lr:0.001.batch_size:16'
    if log_dir is None:
        log_dir = f'./logs/protonet/cifar{args.num_way}.' \
            f'support_eps:{args.num_support_epsilon}.query_eps:{args.num_query_epsilon}.'\
            f'support_novel:{args.num_support_novel}.query_novel:{args.num_query_novel}.'\
            f'eps_lr:{args.threshold_learning_rate}.novel_lr:{args.encoder_learning_rate}.'\
            f'eps_iters:{args.num_threshold_train_iterations}.novel_lr:{args.num_encoder_train_iterations}.'\
            f'batch_size:{args.batch_size}.pretrained:{args.pretrained}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    encoder = SupConResNet(args.model, feat_dim=args.feature_dim)
    if args.pretrained:
        print("Pretrained encoder")
        encoder.load_state_dict(torch.load(args.encoder_path)['model'])
    protonet = ProtoNetTrainer(encoder, args.encoder_learning_rate, args.threshold_learning_rate, log_dir)
    tmp = log_dir

    if args.checkpoint_step > -1 and (args.encd_checkpoint ^ args.thresh_checkpoint):
        protonet._log_dir = 'logs/protonet/cifar10.support_eps:5.query_eps:15.support_novel:5.query_novel:15.eps_lr:0.01.novel_lr:0.001.batch_size:16'
        protonet.load(args.checkpoint_step, 'encd' if args.encd_checkpoint else 'thres', args.num_encoder_train_iterations)
        protonet._log_dir = tmp
        for g in protonet.threshold_optimizer.param_groups:
            g['lr'] = args.threshold_learning_rate
        for g in protonet.encoder_optimizer.param_groups:
            g['lr'] = args.threshold_learning_rate
    else:
        print('Checkpoint loading skipped.')

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    dataset_train = CIFARDataset('data/cifar10-dataset', 'seen', 'train', transform=train_transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=0, collate_fn=lambda x: x, pin_memory=torch.cuda.is_available())
    dataset_val_seen = CIFARDataset('data/cifar10-dataset', 'seen', 'val', transform=train_transform)
    dataloader_val_seen = torch.utils.data.DataLoader(dataset_val_seen, batch_size=args.batch_size, num_workers=0, collate_fn=lambda x: x, pin_memory=torch.cuda.is_available())
    protonet.train(
        dataloader_train,
        dataloader_val_seen,
        writer,
        args
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=10,
                        help='number of classes in a task')
    parser.add_argument('--num_support_epsilon', type=int, default=5,
                        help='number of support examples per class in epsilon task')
    parser.add_argument('--num_support_novel', type=int, default=5,
                        help='number of support examples per class in novel class task')
    parser.add_argument('--num_query_epsilon', type=int, default=15,
                        help='number of query examples per class in epsilon task')
    parser.add_argument('--num_query_novel', type=int, default=15,
                        help='number of query examples per class in novel task')
    parser.add_argument('--seen_unseen_split', type=float, default=4/6)
    parser.add_argument('--num_shots_novel', type=int, default=1,
                        help='Number of support examples of "unseen" classes')
    parser.add_argument('--encoder_learning_rate', type=float, default=0.001,
                        help='learning rate for the encoder training')
    parser.add_argument('--threshold_learning_rate', type=float, default=0.001,
                        help='learning rate for the encoder training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_encoder_train_iterations', type=int, default=5000,
                        help='number of outer-loop updates to train the encoder for')
    parser.add_argument('--num_threshold_train_iterations', type=int, default=5000,
                        help='number of outer-loop updates to train the threshold for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--encd_checkpoint', default=False, action='store_true')
    parser.add_argument('--thresh_checkpoint', default=False, action='store_true')
    parser.add_argument('--encoder_path', type=str, default='save/SupCon/cifar100_models/SupCon_cifar10_resnet18_lr_0.5_decay_0.0001_bsz_2048_temp_0.1_embdim_2048_trial_0_partition_train_data/cifar10-dataset_cosine_warm/ckpt_epoch_100.pth')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--feature_dim', type=int, default=2048)
    parser.add_argument('--pretrained', type=bool, default=True)

    main_args = parser.parse_args()
    main(main_args)


"""
d = CIFARDataset(root='data/cifar10-dataset', class_partition='seen', partition='train', transform=train_transform)
trainer.calculate_prototypes(dl)
dl = DataLoader(d, batch_size=128, num_workers=12)
"""
