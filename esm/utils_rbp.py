
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score, confusion_matrix
from scipy import stats

import numpy as np
from sklearn import metrics

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


__all__ = [
    "pearsonr",
    "rsquare",
    "accuracy",
    "roc",
    "pr",
    "calculate_metrics"
]

# class MLMetrics(object):
class MLMetrics(object):
    def __init__(self, objective='binary'):
        self.objective = objective
        self.metrics = []

    def update(self, label, pred, other_lst):
        met, _ = calculate_metrics(label, pred, self.objective)
        if len(other_lst)>0:
            met.extend(other_lst)
        self.metrics.append(met)
        self.compute_avg() 

    def compute_avg(self):
        if len(self.metrics)>1:
            self.avg = np.array(self.metrics).mean(axis=0)
            self.sum = np.array(self.metrics).sum(axis=0)
        else:
            self.avg = self.metrics[0]
            self.sum = self.metrics[0]
        self.acc = self.avg[0]
        self.auc = self.avg[1]
        self.prc = self.avg[2]
        self.tp  = int(self.sum[3])
        self.tn  = int(self.sum[4])
        self.fp  = int(self.sum[5])
        self.fn  = int(self.sum[6])
        if len(self.avg)>7:
            self.other = self.avg[7:]


def pearsonr(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        corr = [stats.pearsonr(label, prediction)]
    else:
        num_labels = label.shape[1]
        corr = []
        for i in range(num_labels):
            #corr.append(np.corrcoef(label[:,i], prediction[:,i]))
            corr.append(stats.pearsonr(label[:,i], prediction[:,i])[0])

    return corr


def rsquare(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        y = label
        X = prediction
        m = np.dot(X,y)/np.dot(X, X)
        resid = y - m*X;
        ym = y - np.mean(y);
        rsqr2 = 1 - np.dot(resid.T,resid)/ np.dot(ym.T, ym);
        metric = [rsqr2]
        slope = [m]
    else:
        num_labels = label.shape[1]
        metric = []
        slope = []
        for i in range(num_labels):
            y = label[:,i]
            X = prediction[:,i]
            m = np.dot(X,y)/np.dot(X, X)
            resid = y - m*X;
            ym = y - np.mean(y);
            rsqr2 = 1 - np.dot(resid.T,resid)/ np.dot(ym.T, ym);
            metric.append(rsqr2)
            slope.append(m)
    return metric, slope


def accuracy(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        metric = np.array(accuracy_score(label, np.round(prediction)))
    else:
        num_labels = label.shape[1]
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            metric[i] = accuracy_score(label[:,i], np.round(prediction[:,i]))
    return metric


def roc(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        fpr, tpr, thresholds = roc_curve(label, prediction)
        score = auc(fpr, tpr)
        metric = np.array(score)
        curves = [(fpr, tpr)]
    else:
        num_labels = label.shape[1]
        curves = []
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            fpr, tpr, thresholds = roc_curve(label[:,i], prediction[:,i])
            score = auc(fpr, tpr)
            metric[i]= score
            curves.append((fpr, tpr))
    return metric, curves


def pr(label, prediction):
    ndim = np.ndim(label)
    if ndim == 1:
        precision, recall, thresholds = precision_recall_curve(label, prediction)
        score = auc(recall, precision)
        metric = np.array(score)
        curves = [(precision, recall)]
    else:
        num_labels = label.shape[1]
        curves = []
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            precision, recall, thresholds = precision_recall_curve(label[:,i], prediction[:,i])
            score = auc(recall, precision)
            metric[i] = score
            curves.append((precision, recall))
    return metric, curves

def tfnp(label, prediction):
    try:
        tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()
    except Exception:
        tp, tn, fp, fn =0,0,0,0
    
    return tp, tn, fp, fn


def calculate_metrics(label, prediction, objective):
    """calculate metrics for classification"""
    # import pdb; pdb.set_trace()
    

    if (objective == "binary") | (objective == 'hinge'):
        ndim = np.ndim(label)
        #if ndim == 1:
        #    label = one_hot_labels(label)
        correct = accuracy(label, prediction)
        auc_roc, roc_curves = roc(label, prediction)
        auc_pr, pr_curves = pr(label, prediction)
        # import pdb; pdb.set_trace()
        if ndim == 2:
            prediction=prediction[:,0]
            label = label[:,0]
        # pred_class = prediction[:,0]>0.5
        pred_class = prediction>0.5
        # tp, tn, fp, fn = tfnp(label[:,0], pred_class)
        tp, tn, fp, fn = tfnp(label, pred_class)
        # tn8, fp8, fn8, tp8 = tfnp(label[:,0], prediction[prediction>0.8][:,0])
        # import pdb; pdb.set_trace()
        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr),tp, tn, fp, fn]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]

    elif objective == "categorical":

        correct = np.mean(np.equal(np.argmax(label, axis=1), np.argmax(prediction, axis=1)))
        auc_roc, roc_curves = roc(label, prediction)
        auc_pr, pr_curves = pr(label, prediction)
        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr)]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]
        for i in range(label.shape[1]):
            label_c, prediction_c = label[:,i], prediction[:,i]
            auc_roc, roc_curves = roc(label_c, prediction_c)
            mean.append(np.nanmean(auc_roc))
            std.append(np.nanstd(auc_roc))


    elif (objective == 'squared_error') | (objective == 'kl_divergence') | (objective == 'cdf'):
        ndim = np.ndim(label)
        #if ndim == 1:
        #    label = one_hot_labels(label)
        label[label<0.5] = 0
        label[label>=0.5] = 1
        # import pdb; pdb.set_trace()

        correct = accuracy(label, prediction)
        auc_roc, roc_curves = roc(label, prediction)
        auc_pr, pr_curves = pr(label, prediction)
        # import pdb; pdb.set_trace()
        if ndim == 2:
            prediction=prediction[:,0]
            label = label[:,0]
        # pred_class = prediction[:,0]>0.5
        pred_class = prediction>0.5
        # tp, tn, fp, fn = tfnp(label[:,0], pred_class)
        tp, tn, fp, fn = tfnp(label, pred_class)
        # mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr),tp, tn, fp, fn]
        # std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]
        

        # squared_error
        corr = pearsonr(label,prediction)
        rsqr, slope = rsquare(label, prediction)
        # mean = [np.nanmean(corr), np.nanmean(rsqr), np.nanmean(slope)]
        # std = [np.nanstd(corr), np.nanstd(rsqr), np.nanstd(slope)]

        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr),tp, tn, fp, fn, np.nanmean(corr), np.nanmean(rsqr), np.nanmean(slope)]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr), np.nanstd(corr), np.nanstd(rsqr), np.nanstd(slope)]

    else:
        mean = 0
        std = 0

    return [mean, std]


def param_num(model):
    num_param0 = sum(p.numel() for p in model.parameters())
    num_param1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("===========================")
    print("Total params:", num_param0)
    print("Trainable params:", num_param1)
    print("Non-trainable params:", num_param0 - num_param1)
    print("===========================")


def compute_acc_auc(output, y):
    y1 = y.to(device='cpu', dtype=torch.long).numpy()
    p_class = (output >= 0.5).to(device='cpu').data.numpy()
    prob = output.to(device='cpu').data.numpy()
    acc = metrics.accuracy_score(y1, p_class)
    auc = 0.5
    try:
        auc = metrics.roc_auc_score(y1, prob)
    except Exception as e:
        pass

    return acc, auc


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
