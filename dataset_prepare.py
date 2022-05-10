import os
import gzip
from shutil import copyfile
import numpy as np

cancer_type = {
    0: r'health',  # 健康
    1: r'adrenocortical adeoma',  # 肾上腺皮质腺瘤
    2: r'adrenocortical hyperplasia',  # 肾上腺皮质增生
    3: r'adrenal pheochromocytoma'  # 肾上腺嗜铬细胞瘤
}

target_file = r'D:\desktop\医学图像处理\med_img\dataset'


def process_dataset(path, label=None, file_name=None):
    # path = 'G:/DeepLearning/data/'
    if os.path.exists(path):
        # os.system('cd G:/DeepLearning/data/')
        # path = os.getcwd()
        dirs = os.listdir(path)
        # print(dirs)
        for dir in dirs:
            if '.gz' in dir:
                # print(dir)
                filename = file_name + dir.replace(".gz", "")
                gzip_file = gzip.GzipFile(os.path.join(path, dir))
                # print(gzip_file)
                # print(filename)
                with open(os.path.join(path, filename), 'wb+') as f:
                    f.write(gzip_file.read())
                source_file = os.path.join(path, filename)
                if np.random.rand() < 0.3:
                    # Validation Set
                    dataset = 'val'
                    pass
                else:
                    # Train Set
                    dataset = 'train'
                    pass
                destination_file = os.path.join(target_file, dataset, label, file_name + filename)
                copyfile(source_file, destination_file)

    pass


if __name__ == '__main__':
    root_path = r'D:\desktop\医学图像处理\med_img\rawdata'
    # 遍历rawdata文件夹下四类原始数据
    cancer = os.listdir(root_path)
    for label in cancer:
        # patient_series = os.listdir(os.path.join(root_path, label))
        patient_series = os.listdir(os.path.join(root_path, r'adrenal pheochromocytoma'))
        for series in patient_series:
            # path = os.path.join(root_path, 'adrenal pheochromocytoma', series)
            path = os.path.join(root_path, label, series)
            process_dataset(path=path, label=label, file_name=series)
    # process_dataset(path='G:/DeepLearning/data/')
