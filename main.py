# import dataset_nii
# import utils
# import model
# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.models as models
# import torch.utils.data as data
# from torch.utils.data import DataLoader
# from dataload.COVID_Dataload import COVID
# from resnet50 import my_resnet50
from torch import nn, optim
import dataset_nii
# import utils
# # import model
# import torch
# import torchvision
# import utils
#
# import os

import warnings
import numpy as np
from visdom import Visdom

import torchxrayvision as xrv
# from torchvision import transforms
# from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
# from torch.optim.lr_scheduler import StepLR
# from utils import plot_curve
import os
import torch
import torch.nn.functional as F

# device = 'cpu'
import utils


def train(optimizer, epoch, model, train_loader, modelname, criteria):
    model.train()  # 训练模式
    bs = 1
    train_loss = 0
    train_correct = 0

    for batch_index, batch_samples in enumerate(train_loader):

        # move data to device
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
        # data形状，torch.Size([32, 3, 224, 224])
        # data = data[:, 0, :, :]  # 原作者只取了第一个通道的数据来训练，笔者改成了3个通道

        # data = data[:, None, :, :]
        # data形状，torch.Size([32, 1, 224, 224])

        optimizer.zero_grad()
        # print(data.dtype)
        # # data = data.double()
        # print(data.dtype)
        # print(target)
        output = model(data)
        loss = criteria(output, target.long())
        train_loss += criteria(output, target.long())  # 后面求平均误差用的

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()  # 累加预测与标签吻合的次数，用于后面算准确率

        # 显示一个epoch的进度，425张图片，批大小是32，一个epoch需要14次迭代
        if batch_index % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item() / bs))
    # print(len(train_loader.dataset))   # 425
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))

    if os.path.exists('performance') == 0:
        os.makedirs('performance')
    f = open('performance/{}.txt'.format(modelname), 'a+')
    f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f.write('\n')
    f.close()

    return train_loss / len(train_loader.dataset)  # 返回一个epoch的平均误差，用于可视化损失


def val(model, val_loader, criteria):
    model.eval()
    val_loss = 0
    correct = 0

    # Don't update model
    with torch.no_grad():
        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            # data = data[:, 0, :, :]  # 原作者只取了第一个通道的数据，笔者改成了3个通道

            # data = data[:, None, :, :]
            # data形状，torch.Size([32, 1, 224, 224])
            output = model(data)

            val_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            targetcpu = target.long().numpy()  # 由GPU->CPU
            predlist = np.append(predlist, pred.numpy())
            scorelist = np.append(scorelist, score.numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

    return targetlist, scorelist, predlist, val_loss / len(val_loader.dataset)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batchsize = 1  # 原来用的10，这里改成32，根据个人GPU容量来定。
    total_epoch = 2  # 2000个epoch，每个epoch 14次迭代，425/32，训练完就要迭代2000*14次
    votenum = 1

    modelname = 'DenseNet_medical'
    model = xrv.models.DenseNet(num_classes=4, in_channels=3).to('cpu')
    trainset = dataset_nii.TrainDataSet
    valset = dataset_nii.ValDataSet
    # valset = torch.tensor(valset, dtype=torch.float)
    # trainset = torch.tensor(trainset, dtype=torch.float)
    train_loader = dataset_nii.train_loader
    val_loader = dataset_nii.val_loader

    # criteria = nn.CrossEntropyLoss()
    criteria = utils.MultiFocalLoss(num_class=4)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)  # 动态调整学习率策略，初始学习率0.0001

    # ----------------------------  step 5/5 训练 ------------------------------
    # viz = Visdom(server='http://localhost/', port=8097)
    #
    # viz.line([[0., 0., 0., 0., 0.]], [0], win='train_performance', update='replace',
    #          opts=dict(title='train_performance', legend=['precision', 'recall', 'AUC', 'F1', 'acc']))
    # viz.line([[0., 0.]], [0], win='train_Loss', update='replace',
    #          opts=dict(title='train_Loss', legend=['train_loss', 'val_loss']))
    #
    # warnings.filterwarnings('ignore')
    #
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    r_list = []
    p_list = []
    acc_list = []
    AUC_list = []
    #
    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())
    #
    # # 迭代3000*14次
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    print("Start Training...")
    for epoch in range(1, total_epoch + 1):

        train_loss = train(optimizer, epoch, model, train_loader, modelname, criteria)  # 进行一个epoch训练的函数

        targetlist, scorelist, predlist, val_loss = val(model, val_loader, criteria)  # 用验证集验证
        print('target', targetlist)
        print('score', scorelist)
        print('predict', predlist)
        vote_pred = vote_pred + predlist
        vote_score = vote_score + scorelist
        # if epoch % votenum == 0:  # 每10个epoch，计算一次准确率和召回率等
        #
        #     # major vote
        #     vote_pred[vote_pred <= (votenum / 2)] = 0  # 投票，对某样本的预测，超过一半是正例，则判为正例，反之判为负例
        #     vote_pred[vote_pred > (votenum / 2)] = 1
        #     vote_score = vote_score / votenum
        #
        #     print('vote_pred', vote_pred)
        #     print('targetlist', targetlist)
        #
        #     TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        #     TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        #     FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        #     FP = ((vote_pred == 1) & (targetlist == 0)).sum()
        #
        #     print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
        #     print('TP+FP', TP + FP)
        #     p = TP / (TP + FP)
        #     print('precision', p)
        #     p = TP / (TP + FP)
        #     r = TP / (TP + FN)
        #     print('recall', r)
        #     F1 = 2 * r * p / (r + p)
        #     acc = (TP + TN) / (TP + TN + FP + FN)
        #     print('F1', F1)
        #     print('acc', acc)
        #     AUC = roc_auc_score(targetlist, vote_score)
        #     print('AUC', AUC)
        #
        #     # 训练过程可视化
        #     # train_loss = train_loss.numpy()
        #     # val_loss = val_loss.cpu().detach().numpy()
        #     # viz.line([[p, r, AUC, F1, acc]], [epoch], win='train_performance', update='append',
        #     #          opts=dict(title='train_performance', legend=['precision', 'recall', 'AUC', 'F1', 'acc']))
        #     # viz.line([[train_loss], [val_loss]], [epoch], win='train_Loss', update='append',
        #     #          opts=dict(title='train_Loss', legend=['train_loss', 'val_loss']))
        #
        #     print(
        #         '\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, '
        #         'average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        #             epoch, r, p, F1, acc, AUC))
        #
        #     # 更新模型
        #
        #     if os.path.exists('backup') == 0:
        #         os.makedirs('backup')
        #     torch.save(model.state_dict(), "backup/{}.pt".format(modelname))
        #
        #     vote_pred = np.zeros(valset.__len__())
        #     vote_score = np.zeros(valset.__len__())
        #     f = open('performance/{}.txt'.format(modelname), 'a+')
        #     f.write(
        #         '\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, '
        #         'average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        #             epoch, r, p, F1, acc, AUC))
        #     f.close()
        if epoch % (votenum * 1) == 0:  # 每10个epoch，保存一次模型
            torch.save(model.state_dict(), "backup/{}_epoch{}.pt".format(modelname, epoch))
    pass
