
#
# ##读入文件，显示正确分类和预测分类
# import matplotlib.pyplot as plt
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from PIL import Image
# # from resnet50 import my_resnet50
#
# transform = transforms.Compose([
#     transforms.Resize([224,224]),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
#     ])
#
#
# file_name = input("输入要预测的文件名：")
# img = Image.open(file_name).convert("RGB")
# show_img = img
# img = transform(img)
# #
# # print(img)
# # print(img.shape)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# img = img.to(device)
# img = img.unsqueeze(0)
# net = my_resnet50().to(device)
# net.load_state_dict(torch.load(r'model/no_pretrain/epoch200.pth'))
#
# pred = net(img)
# print(pred)
# print(pred.argmax(dim = 1).cpu().numpy()[0])
# res = ''
# if pred.argmax(dim = 1) == 0:
#     res += 'pred:no_covid'
# else:
#     res += 'pred:covid'
#
# plt.figure("Predict")
# plt.imshow(show_img)
# plt.axis("off")
# plt.title(res)
# plt.show()
#
