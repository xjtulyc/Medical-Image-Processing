
#
# import torch
# import torchvision
# # from dataload.COVID_Dataload import COVID
# # 定义使用GPU
# import dataset_nii
# import utils
# import model
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # from resnet50 import my_resnet50
#
# transform = transforms.Compose([
#     transforms.Resize([224, 224]),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
# ])
#
# test_dataset = dataset_nii.ValDataSet
# test_loader = dataset_nii.val_loader
#
#
# def predict():
#     net = my_resnet50().to(device)
#     net.load_state_dict(torch.load('/home/lwf/code/pytorch学习/ResNet/resnet新冠病毒确诊的预测/model/no_pretrain/epoch200.pth'))
#     print(net)
#     total_correct = 0
#     for batch_idx, (x, y) in enumerate(test_loader):
#         # x = x.view(x.size(0),28*28)
#         # x = x.view(256,28,28)
#         x = x.to(device)
#         print(x.shape)
#         y = y.to(device)
#         print('y', y)
#         out = net(x)
#         # print(out)
#         pred = out.argmax(dim=1)
#         print('pred', pred)
#         correct = pred.eq(y).sum().float().item()
#         total_correct += correct
#     total_num = len(test_loader.dataset)
#
#     acc = total_correct / total_num
#     print("test acc:", acc)
#
#
# predict()
