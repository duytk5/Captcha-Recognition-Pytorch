from __future__ import print_function

import argparse
import os
import torch
from torch.utils.data import DataLoader
from config import Config
from torch.autograd import Variable
import torch.optim as optim
from warpctc_pytorch import CTCLoss
from model import CRNNCTC
from dataset import CaptchaDataset
from utils import get_accuracy, AverageMeter, collate_fn, train_result, test_result
from tqdm import tqdm

log_file = open("log_cmd.txt", "w")

normalized = True
if normalized:
    from dataset import mean, std
else:
    mean = [0. for _ in range(3)]
    std = [1. for _ in range(3)]


training_dataset = CaptchaDataset('train', mean=mean, std=std)
testing_dataset = CaptchaDataset('test', mean=mean, std=std)

print('data has been loaded.')

training_loader = DataLoader(training_dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn,
                             pin_memory=True)
testing_loader = DataLoader(testing_dataset, batch_size=Config.batch_size, shuffle=False, collate_fn=collate_fn,
                            pin_memory=True)


def preprocess_target(target):
    """ Preprocess targets.

        :param target: list of `torch.IntTensor`
    """
    lengths = [len(t) for t in target]
    lengths = torch.IntTensor(lengths)

    flatten_target = torch.cat([t for t in target])
    return flatten_target, lengths


def get_seq_length(x):
    """ Get sequence lengths of batch of data
        :param x: batch data
    """
    bsz, length = x.size(0), Config.length_seq
    lengths = torch.IntTensor(bsz).fill_(length)
    return lengths


print("DEVICE : ", Config.device)
model = CRNNCTC().to(Config.device)
criterion = CTCLoss()
solver = optim.SGD(model.parameters(), lr=Config.lr, momentum=0.9)


def train(epoch, max_epoch):
    """ train model
    """
    # if epoch % 10 == 0:
    #     for param_group in solver.param_groups:
    #         param_group['lr'] *= 0.1

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    ind = 0
    for x, target in tqdm(training_loader, desc="TRAINING", leave=False):
        act_lengths = get_seq_length(x)
        flatten_target, target_lengths = preprocess_target(target)
        x = x.to(Config.device)
        x = Variable(x)
        act_lengths = Variable(act_lengths)
        flatten_target = Variable(flatten_target)
        target_lengths = Variable(target_lengths)

        output = model(x)

        loss = criterion(output, flatten_target, act_lengths, target_lengths)
        solver.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), 10)
        solver.step()
        loss_meter.update(loss.data[0])
        acc = get_accuracy(output, target)
        acc_meter.update(acc)
    train_result(epoch, max_epoch, ind, len(training_loader), loss_meter, acc_meter)
    train_result(epoch, max_epoch, ind, len(training_loader), loss_meter, acc_meter, log_file)
    return acc_meter.avg


def test(epoch, max_epoch):
    acc_meter = AverageMeter()
    for x, target in tqdm(testing_loader, desc="TESTING", leave=False):
        x = Variable(x).to(Config.device)
        output = model(x)
        acc = get_accuracy(output, target)
        acc_meter.update(acc)
    test_result(epoch, max_epoch, acc_meter)
    test_result(epoch, max_epoch, acc_meter, log_file)
    return acc_meter.avg


def load_model(model_path):
    if not os.path.exists(model_path):
        raise RuntimeError('cannot find model path: {}'.format(model_path))
    checkpoint = torch.load(model_path, map_location=Config.device)
    # print('load model done.')
    # print('accuracy: {:.2f}'.format(checkpoint['accuracy']))
    return checkpoint['state_dict']


def main(args):
    max_epoch = 500
    if not os.path.exists('pretrained'):
        os.mkdir('pretrained')
    accuracy_save = 0
    if args.load:
        model.load_state_dict(load_model("./pretrained/model-ok.pth.tar"))
    for epoch in range(1, max_epoch + 1):
        train(epoch, max_epoch)
        acc = test(epoch, max_epoch)
        print("[acc max = ", accuracy_save, "]")
        # if accuracy_save < acc:
        accuracy_save = acc
        torch.save({'state_dict': model.state_dict(),
                        'accuracy': acc},
                       os.path.join('pretrained',
                                    'model-ok.pth.tar'.format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=bool, default=False)
    args = parser.parse_args()
    main(args)
