from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import torch
import os
import nibabel as nib
from skimage import transform
import cv2
import numpy as np
from PIL import Image

data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
                                 transforms.CenterCrop(224),  # 从中心开始裁剪
                                 transforms.Resize([224, 224]),
                                 # transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
                                 transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                                 transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                 # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                                 transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
                                 # transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
                                 ]),
    'valid': transforms.Compose([transforms.Resize([224, 224]),
                                 # transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ]),
}


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()
    t_min = t_min.float()
    t_max = t_max.float()

    result = torch.tensor(t >= t_min) * t + torch.tensor(t < t_min) * t_min
    result = torch.tensor(result <= t_max) * result + torch.tensor(result > t_max) * t_max
    return result


# train_data_loader = DataLoader(
#     datasets.ImageFolder(root=r'D:\desktop\医学图像处理\med_img\dataset\train',
#                          transform=data_transforms['train']),
#     batch_size=4, shuffle=True)
#
# test_data_loader = DataLoader(
#     datasets.ImageFolder(root=r'D:\desktop\医学图像处理\med_img\dataset\train',
#                          transform=data_transforms['train']),
#     batch_size=4, shuffle=True)


class CancerCTDataset(Dataset):

    def __init__(self, train=True, transform=None):
        self.cancer_type = {
            r'health': 0,  # 健康
            r'adrenocortical adeoma': 1,  # 肾上腺皮质腺瘤
            r'adrenocortical hyperplasia': 2,  # 肾上腺皮质增生
            r'adrenal pheochromocytoma': 3  # 肾上腺嗜铬细胞瘤
        }
        self.label_check = {
            0: r'health',  # 健康
            1: r'adrenocortical adeoma',  # 肾上腺皮质腺瘤
            2: r'adrenocortical hyperplasia',  # 肾上腺皮质增生
            3: r'adrenal pheochromocytoma'  # 肾上腺嗜铬细胞瘤
        }
        # self.root_dir = root_dir
        # self.txt_path = [txt_COVID, txt_NonCOVID]
        if train:
            self.dataset_path = r'D:\desktop\医学图像处理\med_img\dataset\train'
        else:
            self.dataset_path = r'D:\desktop\医学图像处理\med_img\dataset\val'
        self.classes = os.listdir(self.dataset_path)
        self.num_cls = len(self.classes)
        self.img_list = []
        for CTClass in self.classes:
            CT_path = os.path.join(self.dataset_path, CTClass)
            CT_nii = os.listdir(CT_path)
            for CT_image in CT_nii:
                self.img_list.append([CT_image, self.cancer_type[CTClass]])
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print('here')
        img_path = self.img_list[idx][0]
        # img_path = os.path.join(self.dataset_path, self.label_check[int(self.img_list[idx][1])], img_path)
        # image = Image.open(img_path).convert('RGB')
        image = nib.load(
            os.path.join(self.dataset_path, self.label_check[int(self.img_list[idx][1])], img_path)).get_fdata()
        # print(image.shape)
        if self.transform:
            # image = self.transform(image)
            image = transform.resize(image, (224, 224, 3))
            # # print(image.astype)
            # # 使用均值方差归一化
            mu = np.average(image)
            sigma = np.std(image)
            image = (image - mu) / sigma
            image = (image.astype(np.float64) - 0.5) * 2048
            np.clip(image, -1024, 1024)
            image = Image.fromarray(image, 'RGB')
            # print(image)
            image = self.transform(image)

            # T = transforms.ToTensor()
            # image = T(image)
            # image = clip_by_tensor(image, torch.tensor(-1024), torch.tensor(1024))
            # image = torch.clip(image, -1024, 1024)
            image = image.float()
            # image = (T(image) - 0.5) * 2048
            # image = image.long().float()
            # image = image.convert("RGB")
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample


# def read_txt(txt_path):
#     with open(txt_path) as f:
#         lines = f.readlines()
#     txt_data = [line.strip() for line in lines]  # 主要是跳过'\n'
#     return txt_data

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 依通道标准化

train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
    transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
    # transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
TrainDataSet = CancerCTDataset(train=True, transform=train_transformer)
ValDataSet = CancerCTDataset(train=False, transform=val_transformer)
# ValDataSet = torch.tensor(ValDataSet, dtype=torch.float)
# TrainDataSet = torch.tensor(TrainDataSet,dtype=torch.float)
batchsize = 1  # 原来用的10，这里改成32，根据个人GPU容量来定。
# total_epoch = 2000  # 2000个epoch，每个epoch 14次迭代，425/32，训练完就要迭代2000*14次
# votenum = 10
train_loader = DataLoader(TrainDataSet, batch_size=batchsize, drop_last=False, shuffle=False)
val_loader = DataLoader(ValDataSet, batch_size=batchsize, drop_last=False, shuffle=False)

if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 依通道标准化

    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transformer = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    DataSet = CancerCTDataset(train=True, transform=train_transformer)
    batchsize = 1  # 原来用的10，这里改成32，根据个人GPU容量来定。
    total_epoch = 2000  # 2000个epoch，每个epoch 14次迭代，425/32，训练完就要迭代2000*14次
    votenum = 10
    train_loader = DataLoader(DataSet, batch_size=batchsize, drop_last=False, shuffle=False)
    pass
