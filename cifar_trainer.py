import torch
import os
import logging
from torch.utils.tensorboard import SummaryWriter

from utils import data_utils
from nn.lsoftmax import LSoftmaxLinear

logger = logging.getLogger("CIFAR.Trainer")


class CifarTrainer:
    """
    Basic Tainer for cifar 100 task
    """
    def __init__(self,  model, criterion, optimizer, device, snapshot_dir, log_dir, result_dir, scheduler,
                 change_lr_during_epoch):
        """

        :param model:
        :param criterion:
        :param optimizer:
        :param snapshot_dir:
        :param log_dir:
        :param result_dir:
        """

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.snapshot_dir = snapshot_dir
        self.log_dir = log_dir
        self.result_dir = result_dir
        self.device = device
        self.scheduler = scheduler
        self.change_lr_during_epoch = change_lr_during_epoch

        # TensorBoard visualization: needs command tensorboard --logdir path_to_log_dir
        self.writer = SummaryWriter(os.path.join(log_dir, 'tensorboard_dir'))

        os.makedirs(snapshot_dir, exist_ok=True)

    def train_epoch(self, train_loader, test_loader, epoch, train_acc_check=None, test_acc_check=None):
        logger.info('Train epoch')

        torch.cuda.empty_cache()
        self.model.to(self.device)

        num_batches = len(train_loader)

        for batch_idx, batch_info in enumerate(train_loader):
            images_batch = batch_info[0]
            labels_batch = batch_info[1]

            iteration = num_batches * epoch + batch_idx

            if test_acc_check:
                if batch_idx % test_acc_check == 0 and batch_idx > 0:
                    self.test_model(test_loader,
                                    iteration=iteration,
                                    epoch=epoch, batch_idx=batch_idx,
                                    mark='Test')
            if train_acc_check:
                if batch_idx % train_acc_check == 0 and batch_idx > 0:
                    self.test_model(train_loader,
                                    iteration=iteration,
                                    epoch=epoch, batch_idx=batch_idx,
                                    mark='Train')

            self.model.train()

            images_batch = images_batch.to(self.device).float()
            labels_batch = labels_batch.to(self.device)

            if isinstance(self.model.classifier, LSoftmaxLinear):
                output = self.model(images_batch, labels_batch)
            else:
                output = self.model(images_batch)
            loss = self.criterion(output, labels_batch)

            if batch_idx % 10 == 0:
                self.writer.add_scalars('Loss/batch loss', {'loss': loss.item()}, iteration)
                logger.info('{}_{} Train loss: {}'.format(epoch, batch_idx, loss.item()))

            self.optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            if self.change_lr_during_epoch:
                curr_lr = self.scheduler.get_lr()[0]
                self.writer.add_scalars('LR', {'lr': curr_lr}, iteration)
                if self.scheduler:
                    self.scheduler.step()

    def train_mixup_epoch(self, train_loader, test_loader, epoch, train_acc_check=None, test_acc_check=None):
        logger.info('Train epoch')

        torch.cuda.empty_cache()
        self.model.to(self.device)

        num_batches = len(train_loader)

        for batch_idx, batch_info in enumerate(train_loader):
            images_batch = batch_info[0]
            labels_batch = batch_info[1]

            images_batch, labels_a, labels_b, lam = data_utils.mixup_data(images_batch, labels_batch, 1)

            iteration = num_batches * epoch + batch_idx

            if test_acc_check:
                if batch_idx % test_acc_check == 0 and batch_idx > 0:
                    self.test_model(test_loader,
                                    iteration=iteration,
                                    epoch=epoch, batch_idx=batch_idx,
                                    mark='Test')
            if train_acc_check:
                if batch_idx % train_acc_check == 0 and batch_idx > 0:
                    self.test_model(train_loader,
                                    iteration=iteration,
                                    epoch=epoch, batch_idx=batch_idx,
                                    mark='Train')

            self.model.train()

            images_batch = images_batch.to(self.device).float()

            labels_a = labels_a.to(self.device)
            labels_b = labels_b.to(self.device)

            output = self.model(images_batch)

            loss = lam * self.criterion(output, labels_a) + (1 - lam) * self.criterion(output, labels_b)

            if batch_idx % 10 == 0:
                self.writer.add_scalars('Loss/batch loss', {'loss': loss.item()}, iteration)
                logger.info('{}_{} Train loss: {}'.format(epoch, batch_idx, loss.item()))

            self.optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            if self.change_lr_during_epoch:
                curr_lr = self.scheduler.get_lr()[0]
                self.writer.add_scalars('LR', {'lr': curr_lr}, iteration)
                if self.scheduler:
                    self.scheduler.step()

    def test_model(self, test_loader, iteration, epoch, batch_idx, save_model=False, mark=''):
        """
        Evaluate model
        :param test_loader:
        :param iteration:
        :param epoch:
        :param batch_idx:
        :param save_model:
        :return: accuracy
        """
        logger.info('Accuracy check')

        torch.cuda.empty_cache()
        self.model.to(self.device)
        self.model.eval()

        scores = []
        correct = 0
        total = 0
        mean_loss = 0

        for _, batch_info in enumerate(test_loader):
            images_batch = batch_info[0]
            labels_batch = batch_info[1]
            images_batch = images_batch.to(self.device).float()
            labels_batch = labels_batch.to(self.device)
            output = self.model(images_batch)
            loss = self.criterion(output, labels_batch)
            mean_loss += loss.item()
            score, predicted = torch.max(output.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            scores.extend(score.cpu().numpy())

        acc = correct / total
        mean_loss /= len(test_loader)
        self.writer.add_scalars('Accuracy/acc', {mark: acc}, iteration)
        self.writer.add_scalars('Loss/{Mean loss}', {mark: mean_loss}, iteration)
        logger.info('{} {}_{} loss: {} accuracy: {}'.format(mark, epoch, batch_idx, mean_loss, acc))

        if save_model:
            logger.info('Model is saved as {}/{}_{}.pth'.format(self.snapshot_dir, epoch, batch_idx))
            torch.save(self.model.state_dict(), '{}/{}_{}.pth'.format(self.snapshot_dir, epoch, batch_idx))
        return acc, mean_loss
