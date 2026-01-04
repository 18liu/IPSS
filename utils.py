import torch.nn.functional as F
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
from scipy.special import expit
import logging


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + expit(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic

def mean_squared_error(actual, predicted, squared=True):
    """计算 MSE or RMSE (squared=False)"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    error = predicted - actual

    res = np.mean(error**2)
    if squared==False:
        res = np.sqrt(res)
    
    return res
    
def train_epoch(epoch, model, criterion, optimizer, scheduler, train_loader, config):
    losses = []
    model.train()

    pred_epoch = []
    labels_epoch = []

    for (img1,img2,labels), position in train_loader:

        img1 = img1.to(config.device) 
        img2 = img2.to(config.device) 
        position = position.to(config.device)
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        pred_d = model(img1,img2,position).to(config.device)

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())
        loss.backward()

        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    logistic_pred_all = fit_function(labels_epoch, pred_epoch)
    # compute correlation coefficient
    srcc, _ = spearmanr(logistic_pred_all, labels_epoch)
    plcc, _ = pearsonr(logistic_pred_all, labels_epoch)
    rmse = mean_squared_error(logistic_pred_all, labels_epoch, squared=False)


    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4} / RMSE:{:.4} '.format(epoch + 1, ret_loss, srcc, plcc, rmse))

    return ret_loss, srcc, plcc, rmse

def eval_epoch(config, epoch, model, criterion, test_loader):
    with torch.no_grad():
        losses = []
        model.eval()

        pred_epoch = []
        labels_epoch = []

        for (img1,img2,labels), position in test_loader:

            img1 = img1.to(config.device) 
            img2 = img2.to(config.device) 
            position = position.to(config.device)
            labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)
            pred = model(img1,img2,position).to(config.device) 

            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        logistic_pred_all = fit_function(labels_epoch, pred_epoch)
        # compute correlation coefficient
        srcc, _ = spearmanr(logistic_pred_all, labels_epoch)
        plcc, _ = pearsonr(logistic_pred_all, labels_epoch)
        rmse = mean_squared_error(logistic_pred_all, labels_epoch, squared=False)

        logging.info(
            'Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} ===== RMSE:{:.4}'.format(epoch + 1, np.mean(losses), srcc, plcc, rmse))
        return np.mean(losses), srcc, plcc, rmse