import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torchvision 
from torchvision import transforms
from copy import deepcopy
import torch.nn.functional as F
import time
import scipy.ndimage as nd
import imageio
import os
import pickle

# Đầu vào: batch_size (số lượng hình ảnh trong mỗi batch), norm_factor (hệ số chuẩn hóa, mặc định là 1)
# Tác dụng: Tải dữ liệu MNIST và chuyển thành các batch tensor để huấn luyện và kiểm tra.
#           Dữ liệu được quản lý bằng DataLoader để dễ dàng lặp qua.
# Đầu ra: trainset (các batch dữ liệu huấn luyện), testset (các batch dữ liệu kiểm tra)
def load_mnist(batch_size, norm_factor=1, use_cache=True):
    """Load MNIST dataset with caching"""
    cache_file = './mnist_data/cached_mnist.pkl'
    
    if use_cache and os.path.exists(cache_file):
        print("Loading MNIST from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Loading MNIST from torchvision...")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./mnist_data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)
    trainset = list(iter(trainloader))

    testset = torchvision.datasets.MNIST(root='./mnist_data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True)
    testset = list(iter(testloader))
    
    if use_cache:
        print("Saving MNIST to cache...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((trainset, testset), f)
    
    return trainset, testset


# Đầu vào: batch_size (số lượng hình ảnh trong mỗi batch)
# Tác dụng: Tải dữ liệu CIFAR-10 và chuyển thành các batch tensor để huấn luyện và kiểm tra.
#           Dữ liệu được quản lý bằng DataLoader để dễ dàng lặp qua.
# Đầu ra: train_data (các batch dữ liệu huấn luyện), test_data (các batch dữ liệu kiểm tra)
def get_cifar10(batch_size, use_cache=True):
    """Load CIFAR10 dataset with caching"""
    cache_file = './cifar_data/cached_cifar10.pkl'
    
    if use_cache and os.path.exists(cache_file):
        print("Loading CIFAR10 from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Loading CIFAR10 from torchvision...")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)
    train_data = list(iter(trainloader))

    testset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True)
    test_data = list(iter(testloader))
    
    if use_cache:
        print("Saving CIFAR10 to cache...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((train_data, test_data), f)
    
    return train_data, test_data

# Đầu vào: path (đường dẫn tới thư mục chứa tệp 'wnids.txt')
# Tác dụng: Đọc tệp 'wnids.txt' để tạo từ điển ánh xạ từng ID lớp thành một số nguyên.
# Đầu ra: id_dict (từ điển ánh xạ ID lớp thành số nguyên)
def get_id_dictionary(path):
    id_dict = {}
    for i, line in enumerate(open( path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict

# Đầu vào: path (đường dẫn tới thư mục chứa các tệp 'wnids.txt' và 'words.txt')
# Tác dụng: Tạo từ điển ánh xạ số thứ tự lớp thành cặp (ID, tên lớp) bằng cách kết hợp thông tin từ hai tệp.
# Đầu ra: result (từ điển ánh xạ số thứ tự lớp thành cặp (ID, tên lớp)) 
def get_class_to_id_dict(path):
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open( path + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])      
    return result

# Đầu vào: path (đường dẫn tới dữ liệu Tiny ImageNet), id_dict (từ điển ánh xạ ID lớp thành số nguyên)
# Tác dụng: Tải hình ảnh và nhãn của Tiny ImageNet từ đường dẫn chỉ định.
#           Tạo dữ liệu huấn luyện và kiểm tra với nhãn dạng one-hot.
# Đầu ra: train_data (danh sách hình ảnh huấn luyện), train_labels (danh sách nhãn huấn luyện),
#         test_data (danh sách hình ảnh kiểm tra), test_labels (danh sách nhãn kiểm tra)
def get_data(path,id_dict):
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    for key, value in id_dict.items():
        train_data += [plt.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), format='RGB') for i in range(500)]
        train_labels_ = np.array([[0]*200]*500)
        train_labels_[:, value] = 1
        train_labels += train_labels_.tolist()

    for line in open( path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        test_data.append(plt.imread( path + 'val/images/{}'.format(img_name) ,format='RGB'))
        test_labels_ = np.array([[0]*200])
        test_labels_[0, id_dict[class_id]] = 1
        test_labels += test_labels_.tolist()

    print('finished loading data, in {} seconds'.format(time.time() - t))
    return train_data, train_labels, test_data, test_labels

# Đầu vào: train_images (danh sách hình ảnh huấn luyện), train_labels (danh sách nhãn tương ứng), N_imgs (số lượng hình ảnh cần xử lý)
# Tác dụng: Chuyển đổi danh sách hình ảnh thành tensor PyTorch và chuẩn hóa giá trị pixel về khoảng [0, 1].
# Đầu ra: images (tensor chứa các hình ảnh đã chuẩn hóa, kích thước (N_imgs, 64, 64, 3))
def parse_train_data(train_images, train_labels,N_imgs):
    images = torch.zeros((N_imgs, 64,64,3))
    for i,(img,label) in enumerate(zip(train_images, train_labels)):
        if i >= N_imgs:
            break
        print(i)
        if len(img.shape) == 3:
            images[i,:,:,:] = torch.tensor(img,dtype=torch.float) / 255.0 # normalize
            #labels.append(torch.tensor(label, dtype=torch.float))
    return torch.tensor(images)

