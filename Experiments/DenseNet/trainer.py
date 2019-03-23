import os
import time
import shutil
from tqdm import trange

import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable

from model import DenseNet
from tensorboard_logger import configure, log_value

class Trainer(object):
   
    def __init__(self, config, data_loader):
        
        self.config = config
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
        else:
            self.test_loader = data_loader

        # network params
        self.num_blocks = config.num_blocks
        self.num_layers_total = config.num_layers_total
        self.growth_rate = config.growth_rate
        self.bottleneck = config.bottleneck
        self.theta = config.compression

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.best_valid_acc = 0.
        self.init_lr = config.init_lr
        self.lr = self.init_lr
        self.is_decay = True
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.dropout_rate = config.dropout_rate
        if config.lr_decay == '':
            self.is_decay = False
        else:
            self.lr_decay = [float(x) for x in config.lr_decay.split(',')]

        # other params
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.num_gpu = config.num_gpu
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.dataset = config.dataset
        if self.dataset == 'cifar10':
            self.num_classes = 10
        elif self.dataset == 'pcam':
            self.num_classes = 2
        elif self.dataset == 'cifar100':
            self.num_classes = 100
        else:
            self.num_classes = 1000

        # build densenet model
        self.model = DenseNet(self.num_blocks, self.num_layers_total,
            self.growth_rate, self.num_classes, self.bottleneck, 
                self.dropout_rate, self.theta)

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.init_lr,
                momentum=self.momentum, weight_decay=self.weight_decay)

        if self.num_gpu > 0:
            self.model.cuda()
            self.criterion.cuda()

        # finally configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.get_model_name()
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

    def train(self):
        
        # switch to train mode for dropout
        self.model.train()

        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        for epoch in trange(self.start_epoch, self.epochs):
            
            # decay learning rate
            if self.is_decay:
                self.anneal_learning_rate(epoch)

            # train for 1 epoch
            self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_acc = self.validate(epoch)

            is_best = valid_acc > self.best_valid_acc
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_valid_acc': self.best_valid_acc}, is_best)

    def test(self):
        
        # switch to test mode for dropout
        self.model.eval()

        accs = AverageMeter()
        batch_time = AverageMeter()

        # load the best checkpoint
        self.load_checkpoint(best=True)

        tic = time.time()
        for i, (image, target) in enumerate(self.test_loader):
            if self.num_gpu > 0:
                image = image.cuda()
                target = target.cuda()
            input_var = torch.autograd.Variable(image)
            target_var = torch.autograd.Variable(target)

            # forward pass
            output = self.model(input_var)

            # compute loss & accuracy 
            acc = self.accuracy(output.data, target)
            accs.update(acc, image.size()[0])

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)

            # print to screen
            if i % self.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Test Acc: {acc.val:.3f} ({acc.avg:.3f})'.format(
                        i, len(self.test_loader), batch_time=batch_time,
                        acc=accs))

        print('[*] Test Acc: {acc.avg:.3f}'.format(acc=accs))

    def train_one_epoch(self, epoch):
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        for i, (image, target) in enumerate(self.train_loader):
            if self.num_gpu > 0:
                image = image.cuda()
                target = target.cuda()
            input_var = torch.autograd.Variable(image)
            target_var = torch.autograd.Variable(target)

            # forward pass
            output = self.model(input_var)

            # compute loss & accuracy 
            loss = self.criterion(output, target_var)
            acc = self.accuracy(output.data, target)
            losses.update(loss.data, image.size()[0])
            accs.update(acc, image.size()[0])

            # compute gradients and update SGD
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)

            # print to screen
            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Train Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Train Acc: {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, len(self.train_loader), batch_time=batch_time,
                        loss=losses, acc=accs))

        # log to tensorboard
        if self.use_tensorboard:
            log_value('train_loss', losses.avg, epoch)
            log_value('train_acc', accs.avg, epoch)


    def validate(self, epoch):
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        for i, (image, target) in enumerate(self.valid_loader):
            if self.num_gpu > 0:
                image = image.cuda()
                target = target.cuda()
            input_var = torch.autograd.Variable(image)
            target_var = torch.autograd.Variable(target)

            # forward pass
            output = self.model(input_var)

            # compute loss & accuracy 
            loss = self.criterion(output, target_var)
            acc = self.accuracy(output.data, target)
            losses.update(loss.data[0], image.size()[0])
            accs.update(acc, image.size()[0])

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)

            # print to screen
            if i % self.print_freq == 0:
                print('Valid: [{0}/{1}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Valid Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Valid Acc: {acc.val:.3f} ({acc.avg:.3f})'.format(
                        i, len(self.valid_loader), batch_time=batch_time,
                        loss=losses, acc=accs))

        print('[*] Valid Acc: {acc.avg:.3f}'.format(acc=accs))

        # log to tensorboard
        if self.use_tensorboard:
            log_value('val_loss', losses.avg, epoch)
            log_value('val_acc', accs.avg, epoch)

        return accs.avg

    def save_checkpoint(self, state, is_best):
        
        print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.get_model_name() + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.get_model_name() + '_model_best.pth.tar'
            shutil.copyfile(ckpt_path, 
                os.path.join(self.ckpt_dir, filename))
            print("[*] ==== Best Valid Acc Achieved ====")

    def load_checkpoint(self, best=False):
        
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.get_model_name() + '_ckpt.pth.tar'
        if best:
            filename = self.get_model_name() + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['state_dict'])
        
        print("[*] Loaded {} checkpoint @ epoch {} with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc']))

    def anneal_learning_rate(self, epoch):
        
        sched1 = int(self.lr_decay[0] * self.epochs)
        sched2 = int(self.lr_decay[1] * self.epochs)

        self.lr = self.init_lr * (0.1 ** (epoch // sched1)) \
                               * (0.1 ** (epoch // sched2))

        # log to tensorboard
        if self.use_tensorboard:
            log_value('learning_rate', self.lr, epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def get_model_name(self):
       
        if self.bottleneck:
            return 'DenseNet-BC-{}-{}'.format(self.num_layers_total,
                self.dataset)
        return 'DenseNet-{}-{}'.format(self.num_layers_total,
            self.dataset)

    def accuracy(self, predicted, ground_truth):
        
        predicted = torch.max(predicted, 1)[1]
        total = len(ground_truth)
        correct = (predicted == ground_truth).sum()
        acc = 100 * (correct / total)
        return acc

class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
