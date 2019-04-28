import torch
from torch.autograd import Variable
import torch.nn.functional as F

BLANK_LABEL = 0


def get_prediction(output, prob=True, lengths=None):
    """ Get prediction for sequence classification.

       :param output(torch.FloatTensor): output of model: seq_len x batch_size x channels
       :param prob(boolean): use `softmax` to conver `output` to probability or not
       :param length(list or None): lengths of batch elements. if have the same length, it can
                                    be inferred from `output`.
    """
    seq_len, batch_size, channels = output.size()
    output = output.view(-1, channels)
    output = output.cpu()

    _, max_ind = torch.max(output, dim=1)
    max_ind = max_ind.view((seq_len, batch_size))
    pred_labels = []
    for b in range(batch_size):
        bind = max_ind[:, b]
        prev_ind = BLANK_LABEL
        pred_label = []
        for t in range(seq_len):
            if bind[t] != prev_ind and bind[t] != BLANK_LABEL:
                pred_label.append(int(bind[t]))
            prev_ind = bind[t]
        pred_labels.append(pred_label)
    return pred_labels, None


class AverageMeter(object):
    """ Average meter.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset items.
        """
        self.n = 0
        self.val = 0.
        self.sum = 0.
        self.avg = 0.

    def update(self, val, n=1):
        """ Update
        """
        self.n += n
        self.val = val
        self.sum += val * n
        self.avg = self.sum / self.n


def get_accuracy(output, targets, prob=True):
    """ Get accuracy given output and targets
    """
    pred, _ = get_prediction(output, prob)
    cnt = 0
    for batch_ind, target in enumerate(targets):
        target = [v.item() for v in target]
        # print(target, pred[batch_ind])
        if target == pred[batch_ind]:
            cnt += 1
    return float(cnt) / len(targets)


def collate_fn(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = torch.stack([x[1] for x in batch], dim=0)
    return imgs, labels


def collate_fn_test(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = [x[1] for x in batch]
    return imgs, labels


def train_result(epoch, max_epoch, ind, len_training_loader, loss_meter, acc_meter, file=None):
    print('\ntrain:\t[{:03d}/{:03d}],\t'
          '[{:02d}/{:02d}]\t'
          'loss: {loss.avg:.4f}({loss.val:.4f})\t'
          'accuracy: {acc.avg:.4f}({acc.val:.4f})'.format(epoch, max_epoch,
                                                          ind + 1, len_training_loader, loss=loss_meter,
                                                          acc=acc_meter), file=file)


def test_result(epoch, max_epoch, acc_meter, file=None):
    print('test:\t[{:03d}/{:03d}],\t'
          'accuracy: {acc.avg:.4f}({acc.val:.4f})'.format(epoch, max_epoch, acc=acc_meter), file=file)
