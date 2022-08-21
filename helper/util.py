from __future__ import print_function

import json
import torch
import numpy as np
import torch.distributed as dist
# from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import pandas as pd

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

# def adjust_learning_rate(optimizer, epoch, step, len_epoch, old_lr):
#     """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
#     if epoch < 5:
#         lr = old_lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)
#     elif 5 <= epoch < 60: return
#     else:
#         factor = epoch // 30
#         factor -= 1
#         lr = old_lr*(0.1**factor)

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


class PredList(object):
  def __init__(self):
    self.reset()
  def reset(self):
    self.y_true = []
    self.y_pred = []
  def update(self, y_true, y_pred):
    with torch.no_grad():
      _, y_pred = y_pred.topk(1, dim = 1, largest = True, sorted = True)
      y_pred = y_pred.squeeze(1)

    y_true = y_true.cpu().detach().numpy().tolist()
    y_pred = y_pred.cpu().detach().numpy().tolist()
    
    self.y_true.extend(y_true)
    self.y_pred.extend(y_pred)

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def update_classification_report(epoch, cl_report, opt, validation=False):
  target = cl_report.y_true
  pred = cl_report.y_pred
  
  report = classification_report(target, pred, labels=[0,1], 
                                           zero_division=0, output_dict=True, target_names=['0', '1'])
  if validation == False:
    with pd.ExcelWriter(opt.save_folder+"/classification_report_training.xlsx", engine='openpyxl', mode=('w' if epoch==1 else 'a')) as writer:  
      pd.DataFrame(report).T.sort_values(by='support', ascending=False).to_excel(writer, sheet_name='epoch{}'.format(epoch))
  else:
    with pd.ExcelWriter(opt.save_folder+"/classification_report_validation.xlsx", engine='openpyxl', mode=('w' if epoch==1 else 'a')) as writer:  
      pd.DataFrame(report).T.sort_values(by='support', ascending=False).to_excel(writer, sheet_name='epoch{}'.format(epoch))
    

# def classification_metrics(output, target, epoch, opt, validation=False):
#   with torch.no_grad():
#     batch_size = target.size(0)

#     _, pred = output.topk(1, dim = 1, largest = True, sorted = True)
#     pred = pred.squeeze(1)
#     prec = precision_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), zero_division=0)
#     rec = recall_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), zero_division=0)
#     macf1 = f1_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='macro', zero_division=0)
#     micf1 = f1_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='micro', zero_division=0)
   
#     return (prec, rec, macf1, micf1)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim = 1, largest = True, sorted = True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def load_json_to_dict(json_path):
    """Loads json file to dict 

    Args:
        json_path: (string) path to json file
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def reduce_tensor(tensor, world_size = 1, op='avg'):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size > 1:
        rt = torch.true_divide(rt, world_size)
    return rt

if __name__ == '__main__':

    pass
