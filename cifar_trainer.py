import torch
import os
import logging
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("CIFAR.Trainer")


class CifarTrainer:
    """
    Basic Tainer for cifar 100 task
    """
    def __init__(self,  model, criterion, optimizer, device, snapshot_dir, log_dir, result_dir):
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

        # TensorBoard visualization: needs command tensorboard --logdir path_to_log_dir
        self.writer = SummaryWriter(os.path.join(log_dir, 'tensorboard_dir'))

        os.makedirs(snapshot_dir, exist_ok=True)

    def train_epoch(self, train_loader, test_loader, epoch, train_acc_check=100, test_acc_check=100):
        logger.info('Train epoch')

        torch.cuda.empty_cache()
        self.model.to(self.device)

        num_batches = len(train_loader)

        for batch_idx, batch_info in enumerate(train_loader):
            images_batch = batch_info[0]
            labels_batch = batch_info[1]
            iteration = num_batches * epoch + batch_idx

            if batch_idx % test_acc_check == 0 and batch_idx > 0:
                self.test_model(test_loader,
                                iteration=iteration,
                                epoch=epoch, batch_idx=batch_idx)

            self.model.train()

            images_batch = images_batch.to(self.device).float()
            labels_batch = labels_batch.to(self.device)

            output = self.model(images_batch)
            loss = self.criterion(output, labels_batch)

            if batch_idx % train_acc_check == 0:
                self.writer.add_scalar('Loss/train', loss.item(), iteration)
                logger.info('{}_{} Train loss: {}'.format(epoch, batch_idx, loss.item()))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def test_model(self, test_loader, iteration, epoch, batch_idx, save_model=True):
        """
        Evaluate model
        :param test_loader:
        :param iteration:
        :param epoch:
        :param batch_idx:
        :param save_model:
        :return: accuracy
        """
        logger.info('Test model')
        if save_model:
            torch.save(self.model.state_dict(), '{}/{}_{}.pth'.format(self.snapshot_dir, epoch, batch_idx))

        torch.cuda.empty_cache()
        self.model.to(self.device)
        self.model.eval()

        scores = []
        correct = 0
        total = 0
        mean_loss = 0

        for batch_idx, batch_info in enumerate(test_loader):
            images_batch = batch_info[0]
            labels_batch = batch_info[1]
            images_batch = images_batch.to(self.device).float()
            output = self.model(images_batch)
            loss = self.criterion(output, labels_batch)
            mean_loss += loss.item()
            score, predicted = torch.max(output.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            scores.extend(score.cpu().numpy())

        acc = correct / total
        mean_loss /= len(test_loader)
        self.writer.add_scalar('Accuracy/test', acc, iteration)
        logger.info('{}_{} Test loss: {} accuracy: {}'.format(epoch, batch_idx, mean_loss, acc))
        return


    def adjust_learning_rate(self, epoch, step=2, base_lr=1e-3):
        """
        Updating learning rate each epoch multiple to step value
        :param epoch: current epoch
        :param step:
        :param base_lr:
        :return:
        """
        lr = base_lr * (0.1 ** (epoch // step))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        logger.info('Learning rate was updated to {}'.format(lr))