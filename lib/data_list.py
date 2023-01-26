import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# def make_dataset(image_list, labels):
#     if labels:
#       len_ = len(image_list)
#       images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
#     else:
#       if len(image_list[0].split()) > 2:
#         images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
#       else:
#         images = [(val.split()[0], int(val.split()[1])) for val in image_list]
#     return images

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip().split(' ')[0], int(image_list[i].strip().split(' ')[1])) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        idx_each_class_noisy = [[] for i in range(31)]
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, loader=rgb_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images \n"))

        self.imgs = imgs
        # self.values = [1.0] * len(imgs)
        self.values = [self.imgs[i][1] for i in range(len(self.imgs))]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, int(target)

    def __len__(self):
        return len(self.imgs)

import random


def save_lines(lst, save_path):
    file = open(save_path, "w")
    lst_str = "\t\n".join(lst)
    file.write(lst_str)
    file.close()

def split_data_list(image_list, test_pct=0.1):
    image_dic = {}
    for i in range(len(image_list)):
        label = int(image_list[i].strip().split(' ')[1])
        image_path = image_list[i].strip().split(' ')[0]
        if label not in image_dic.keys():
            image_dic[label] = []
        image_dic[label].append(image_path)
    train_dic = {}
    test_dic = {}
    for key, value in image_dic.items():
        random.shuffle(value)
        s = int(len(value)*(1-test_pct))
        if s < 5:
            s = 5
            print("{} {} {}".format(key, len(value), int(len(value)*(1-test_pct))))
        train_dic[key] = value[:s]
        test_dic[key] = value[s:]
    train_list, test_list = [], []
    for k, v in train_dic.items():
        for v1 in v:
            train_list.append("{} {}".format(v1, k))
    for k, v in test_dic.items():
        for v1 in v:
            test_list.append("{} {}".format(v1, k))
    return train_list, test_list


######## Example ####################
"""
train_list, test_list=split_data_list(open('./data/office/amazon_list.txt').readlines(), test_pct=0.5)
save_lines(train_list, "./data/office/amazon_train_list.txt")
save_lines(test_list, "./data/office/amazon_test_list.txt")


train_list, test_list = split_data_list(open('./data/office/dslr_list.txt').readlines(), test_pct=0.5)
save_lines(train_list, "./data/office/dslr_train_list.txt")
save_lines(test_list, "./data/office/dslr_test_list.txt")
print('--------')

train_list, test_list = split_data_list(open('./data/office/webcam_list.txt').readlines(), test_pct=0.5)
save_lines(train_list, "./data/office/webcam_train_list.txt")
save_lines(test_list, "./data/office/webcam_test_list.txt")


s_train_set = ImageList(open('./data/office/amazon_list.txt').readlines(), noisy=True, noise_type=args.noise_type, noise_rate=args.noise_rate,\
                    transform=transforms.Compose([
                                ResizeImage(256),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

s_test_set = ImageList(open('./data/office/amazon_list.txt').readlines(), noisy=False, noise_type='clean', noise_rate=0, \
                    transform=transforms.Compose([
                                ResizeImage(256),
                                #PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))
t_train_set = ImageList(open('./data/office/dslr_list.txt').readlines(), noisy=False, noise_type='clean', noise_rate=0, \
                    transform=transforms.Compose([
                                ResizeImage(256),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))
t_test_set = ImageList(open('./data/office/dslr_list.txt').readlines(), noisy=False, noise_type='clean', noise_rate=0, \
                    transform=transforms.Compose([
                                ResizeImage(256),
                                #PlaceCrop(224, (256 - 224 - 1) / 2, (256 - 224 - 1) / 2),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))
                            
                            
s_train_set = ImageValueList(open('./data/office/amazon_list.txt').readlines(),labels=True, \
                    transform=transforms.Compose([
                                ResizeImage(256),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))

s_test_set = ImageValueList(open('./data/office/amazon_list.txt').readlines(), labels=True, \
                    transform=transforms.Compose([
                                ResizeImage(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))
"""
#########################################
