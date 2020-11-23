import numpy as np
import time
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from utils.loss import ClassificationLosses
from torch.utils.data import DataLoader


class Cifar(object):
    def __init__(self):
        args = self.get_args()
        self.create_dir(args.model_weight_dir)
        self.gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu else "cpu")
        self.net = self.load_model(args)
        self.load_dataset(args)
        self.build_opt_loss(args)
        self.train(args)

    def get_args(self):
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
        parser.add_argument("--num_classes", type=int, default=10)
        parser.add_argument("--batch-size", type=int, default=256, help="number of batch size")
        parser.add_argument("--learning-rate", type=float, default=1e-3, help="learning_rate")
        parser.add_argument('--model', type=str, default="densenet", help="(example: densenet,GoogLeNet, ShuffleNetG2)")
        parser.add_argument('--loss_type', type=str, default='focal')
        parser.add_argument('--opt_weight_decay', type=float, default=5e-4)
        parser.add_argument('--CosineAnnealingLR_T_max', type=int, default=5)
        parser.add_argument('--CosineAnnealingLR_eta_min', type=float, default=1e-5)
        parser.add_argument('--save_num_epoch', type=int, default=1)
        parser.add_argument('--model_weight_dir', type=str, default='./checkpoints/')
        parser.add_argument('--workers', type=int, default=8)

        args = parser.parse_args()
        return args

    def create_dir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def _init_weight(self, net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_model(self, args):
        try:
            if args.model == "GoogLeNet":
                net = GoogLeNet()
            elif args.model == "densenet":
                net = densenet_cifar()
            elif args.model == "ShuffleNetV2":
                net = ShuffleNetV2()
        except:
            print("not model....")
        self._init_weight(net)
        net = net.to(self.device)
        if self.gpu:
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        return net

    def load_dataset(self, args):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform_train)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        self.num_train = len(trainset)
        self.num_test = len(testset)
        print(self.num_train, self.num_test)

    def build_opt_loss(self, args):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate,
                                          weight_decay=args.opt_weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                       T_max=args.CosineAnnealingLR_T_max,
                                                                       eta_min=args.CosineAnnealingLR_eta_min)
        loss = ClassificationLosses(cuda=self.gpu)
        self.criterion = loss.build_loss(args.loss_type)

    def accuracy(self, output, target):
        with torch.no_grad():
            batch_size = target.size(0)
            _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(1 / batch_size).cpu().detach().numpy()[0]
        return acc

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train_stage(self, train_loader, model, criterion, optimizer, epoch, epoch_size, Epoch):
        model.train()
        acc = []
        total_loss = 0
        start_time = time.time()
        with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for i, (data, target) in enumerate(train_loader):
                if i >= epoch_size:
                    break
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)
                loss = criterion(output, target.long())
                total_loss += loss.detach().item()
                acc.append(self.accuracy(output, target.long()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                waste_time = time.time() - start_time
                pbar.set_postfix(**{'total_loss': total_loss / (i + 1),
                                    'lr': self.get_lr(optimizer), 'step/s': waste_time})
                pbar.update(1)
                start_time = time.time()
        return total_loss

    def validate_stage(self, val_loader, model, criterion, epoch, epoch_size, Epoch):
        model.eval()
        valid_total_loss = 0
        acc = []
        with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            with torch.no_grad():
                for i, (data, target) in enumerate(val_loader):
                    if i >= epoch_size:
                        break
                    data = data.to(self.device)
                    target = target.to(self.device)
                    with torch.no_grad():
                        output = model(data)
                        loss = criterion(output, target.long())
                        valid_total_loss += loss.detach().item()
                        acc.append(self.accuracy(output, target.long()))
                    pbar.set_postfix(**{'total_loss': valid_total_loss / (i + 1)})
                    pbar.update(1)
        return valid_total_loss, np.mean(np.array(acc))

    def train(self, args):
        epoch_size_train = max(1, self.num_train // args.batch_size)
        epoch_size_test = self.num_test // args.batch_size
        for epoch in range(args.epochs):
            train_total_loss = self.train_stage(self.train_loader, self.net, self.criterion, self.optimizer,
                                                epoch, epoch_size_train, args.epochs)
            test_total_loss, test_acc = self.validate_stage(self.test_loader, self.net, self.criterion, epoch,
                                                            epoch_size_test, args.epochs)
            self.lr_scheduler.step()
            if (epoch + 1) % args.save_num_epoch == 0:
                print('Saving state -- iter:', str(epoch + 1))
                self.create_dir(args.model_weight_dir + args.model)
                torch.save(self.net.state_dict(),
                           args.model_weight_dir + args.model + '/Epoch%d-Train_Loss%.4f-Val_Loss%.4f-Val_acc%.4f.pth' % (
                               (epoch + 1), train_total_loss / (epoch_size_train + 1),
                               test_total_loss / (epoch_size_test + 1), test_acc))
            print('Finish Validation')
            print('Epoch:' + str(epoch + 1) + '/' + str(args.epochs))
            print('Train Total Loss: %.4f || Val Total Loss: %.4f || Val acc: %.4f ' % (
                train_total_loss / (epoch_size_train + 1), test_total_loss / (epoch_size_test + 1), test_acc))


if __name__ == "__main__":
    model = Cifar()
