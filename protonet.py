"""Implementation of prototypical networks for Omniglot."""

import argparse
from dis import dis
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard

import util

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TASKS_PER_ITERATION = 32
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
            old_shape_0 = self._prototypes.shape[0]
            new_shape_0 = old_shape_0 + labels.shape[0]
            new_prototypes = torch.zeros((new_shape_0, self._prototypes.shape[1]))
            new_labels = torch.zeros(new_shape_0)
            new_prototypes[:old_shape_0] = self._prototypes
            new_prototypes[old_shape_0:] = prototypes
            new_labels[:old_shape_0] = self._prototype_labels
            new_labels[old_shape_0:] = labels
            self._prototypes = new_prototypes
            self._prototype_labels = new_labels


    


class ProtoNet:
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
        self.ood_optimizer = torch.optim.Adam(
            [self._network.threshold],
            lr=threshold_learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_enc_step = 0
        self._start_train_thres_step = 0

    def embd_train_step(self, task):
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
        labels_query = labels_query.to(DEVICE)
        
        num_classes = images_support.shape[0]
        num_support = images_support.shape[1]

        mask_support = torch.reshape(mask_support, (mask_support.shape[0], mask_support[1], 1))

        latents_support = self._network.embed_forward(torch.reshape(images_support, (-1, images_support.shape[2], images_support.shape[3], images_support.shape[4])))
        latents_query = self._network.embed_forward(torch.reshape(images_query, (-1, images_query.shape[2], images_query.shape[3], images_query.shape[4])))
        
        latents_support_reshaped = torch.reshape(latents_support, (num_classes, num_support, -1))
        prototypes_shuffled = torch.sum(latents_support_reshaped * mask_support, axis = 1) / torch.sum(mask_support, axis = 1)
        idx = torch.argsort(labels_support)
        prototypes = prototypes_shuffled[idx]

        distances_query = torch.stack([torch.linalg.vector_norm(prototypes - x, dim = 1) for x in latents_query])
        logits_query = - torch.square(distances_query)

        distances_support = torch.stack([torch.linalg.vector_norm(prototypes - x, dim = 1) for x in latents_support])
        logits_support = - torch.square(distances_support)

        return (
            F.cross_entropy(logits_query, labels_query),
            util.score(logits_support, labels_support),
            util.score(logits_query, labels_query)
        )

    def threshold_train_step(self, task):

        images_support, labels_support, images_query, labels_query = task
        images_support = images_support.to(DEVICE)
        labels_support = labels_support.to(DEVICE)
        images_query = images_query.to(DEVICE)
        labels_query = labels_query.to(DEVICE)

        num_seen_classes = images_support.shape[0]
        num_support = images_support.shape[1]

        with torch.no_grad():
            latents_support = self._network.embed_forward(torch.reshape(images_support, (-1, images_support.shape[2], images_support.shape[3], images_support.shape[4])))
            latents_query = self._network.embed_forward(torch.reshape(images_query, (-1, images_query.shape[2], images_query.shape[3], images_query.shape[4])))

            latents_support_reshaped = torch.reshape(latents_support, (num_seen_classes, num_support, -1))
            prototypes = torch.mean(latents_support_reshaped, 1)
            distances_query = torch.stack([torch.linalg.vector_norm(prototypes - x, dim = 1) for x in latents_query])
            min_dist_query = torch.min(distances_query, dim = 1)

            distances_support = torch.stack([torch.linalg.vector_norm(prototypes - x, dim = 1) for x in latents_support])
            min_dist_support = torch.min(distances_support, dim = 1)
        
        prob_ood_query = F.sigmoid(min_dist_query - self._network.ood_forward())
        prob_ood_support = F.sigmoid(min_dist_support - self._network.ood_forward())

        return (F.binary_cross_entropy(prob_ood_query, labels_query),
        util.bin_score(prob_ood_support, labels_support),
        util.bin_score(prob_ood_query, labels_query)
        )

################################################ TODO ##############################################
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
        for i_step in range(self._start_train_enc_step, num_train_iterations):
            self.encoder_optimizer.zero_grad()
            loss_batch = []
            accuracy_support_batch = []
            accuracy_query_batch = []
            for _ in range(batch_size):
                loss_task, accuracy_support_task, accuracy_query_task = self._step(dataloader_train.sample_novel_cls())
                loss_batch.append(loss_task)
                accuracy_query_batch.append(accuracy_query_task)
                accuracy_support_batch.append(accuracy_support_task)
            loss, accuracy_support, accuracy_query = (torch.mean(torch.stack(loss_batch)),
                                                        np.mean(accuracy_support_batch),
                                                        np.mean(accuracy_query_batch))
            loss.backward()
            self.encoder_optimizer.step()

            if i_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                    f'support accuracy: {accuracy_support.item():.3f}, '
                    f'query accuracy: {accuracy_query.item():.3f}'
                )
                writer.add_scalar('loss/train', loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/support',
                    accuracy_support.item(),
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/query',
                    accuracy_query.item(),
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                with torch.no_grad():
                    losses, accuracies_support, accuracies_query = [], [], []
                    for _ in range(4):

                        loss_batch = []
                        accuracy_support_batch = []
                        accuracy_query_batch = []
                        for _ in range(batch_size):
                            loss_task, accuracy_support_task, accuracy_query_task = self._step(dataloader_val.sample_novel_cls())
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
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/support',
                    accuracy_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/query',
                    accuracy_query,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)
################################################ TODO ##############################################
    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state['network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
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
    if log_dir is None:
        log_dir = f'./logs/protonet/omniglot.way:{args.num_way}.support:{args.num_support}.query:{args.num_query}.lr:{args.learning_rate}.batch_size:{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    protonet = ProtoNet(args.learning_rate, log_dir)

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_train = omniglot.get_omniglot_dataloader(
            'train',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            num_training_tasks
        )
        dataloader_val = omniglot.get_omniglot_dataloader(
            'val',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4
        )
        protonet.train(
            dataloader_train,
            dataloader_val,
            writer,
            args.num_train_iterations,
            args.batch_size
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = omniglot.get_omniglot_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS
        )
        protonet.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=5000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))

    main_args = parser.parse_args()
    main(main_args)


"""
d = CIFARDataset(root='data/cifar10-dataset', class_partition='seen', partition='train', transform=train_transform)
trainer.calculate_prototypes(dl)
dl = DataLoader(d, batch_size=128, num_workers=12)
"""
