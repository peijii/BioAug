import numpy as np
import os
import torch
import random
from torch.nn import functional as F
from sklearn.metrics import f1_score


def seed_everything(seed):
    """
    固定各类随机种子，方便消融实验
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch, num_classes):
        model.train()

        conf_mat = np.zeros((num_classes, num_classes))
        loss_sigma = []
        all_labels = []
        all_predictions = []

        for idx, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.to(torch.float32)
            labels = labels.type(torch.LongTensor)

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss值
            loss_sigma.append(loss.item())
            acc_avg = conf_mat.trace() / conf_mat.sum()
            f1_avg = f1_score(all_labels, all_predictions, average="weighted")

            # 每50个iteration 打印一次训练信息, loss为50个iteration的均值
            if idx % 100 == 100 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc: {:.2%} F1: {:.2%}".format(
                    epoch_id + 1, max_epoch, idx + 1, len(data_loader), np.mean(loss_sigma), acc_avg, f1_avg
                ))

        return np.mean(loss_sigma), acc_avg, f1_avg

    @staticmethod
    def valid(data_loader, model, loss_f, device, num_classes):
        model.eval()

        conf_mat = np.zeros((num_classes, num_classes))
        loss_sigma = []
        all_labels = []
        all_predictions = []

        for idx, data in enumerate(data_loader):

            inputs, labels = data
            inputs = inputs.to(torch.float32)
            labels = labels.type(torch.LongTensor)

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_f(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss值t
            loss_sigma.append(loss.item())

        acc_avg = conf_mat.trace() / conf_mat.sum()
        f1_avg = f1_score(all_labels, all_predictions, average="weighted")

        return np.mean(loss_sigma), acc_avg, f1_avg