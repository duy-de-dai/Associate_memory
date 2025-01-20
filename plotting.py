# functions for plotting graphs given the associative memory networks
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from functions import *
from data import *
from copy import deepcopy
import pickle

SAVE_FORMAT = "png"

# Hàm: parse_sname_for_title
# Đầu vào: sname (chuỗi tên tập dữ liệu)
# Tác dụng: Xác định tiêu đề dựa trên tên tập dữ liệu.
# Đầu ra: Tiêu đề chuỗi (MNIST, CIFAR10, hoặc ImageNet)
def parse_sname_for_title(sname):
    if "tiny_" in sname or "imagenet_" in sname:
        return "ImageNet"
    if "mnist_" in sname:
        return "MNIST"
    else:
        return "CIFAR10"

# Hàm: plot_capacity_graphs
# Đầu vào: Ns (danh sách số lượng hình ảnh), imgs (dữ liệu ảnh), beta (tham số cập nhật), fs (các hàm đo độ tương đồng), labels (nhãn)
#          image_perturb_fn (hàm biến đổi ảnh), sep_fn (hàm tách), sep_param (tham số tách), sigma (độ lệch chuẩn), plot_results (có vẽ hay không), error_threshold (ngưỡng lỗi)
# Tác dụng: Tính tỷ lệ khôi phục chính xác của mạng trí nhớ liên kết khi lưu trữ các hình ảnh. Vẽ đồ thị dung lượng trí nhớ.
# Đầu ra: Mảng chứa tỷ lệ khôi phục chính xác cho mỗi hàm độ tương đồng
def plot_capacity_graphs(Ns, imgs, beta,fs, labels, image_perturb_fn = halve_continuous_img,sep_fn = separation_max, sep_param=1000,sigma=0.1,plot_results = False, error_threshold = 60,normalize_error_threshold = True):
    if normalize_error_threshold:
        error_threshold = (error_threshold * 784) / np.prod(np.array(imgs[0].shape))
        print("ERROR THRESHHOLD")
        print(imgs[0].shape)
    corrects_list = [[] for i in range(len(fs))]
    for i,(f, label) in enumerate(zip(fs, labels)):
        print(label.upper())
        for N in Ns:
            print(N)
            N_correct = PC_retrieve_store_continuous(imgs,N,beta=beta,num_plot=0,f = f, image_perturb_fn = gaussian_perturb_image,sigma = sigma,sep_fn = sep_fn, sep_param=sep_param, ERROR_THRESHOLD=error_threshold)
            corrects_list[i].append(N_correct)
    if plot_results:
        plt.title("Memory Capacity associative memory networks")
        for i in range(len(fs)):
            plt.plot(Ns, corrects_list[i], label = labels[i])##

        plt.xlabel("Images Stored")
        plt.ylabel("Fraction Correctly Retrieved")
        plt.legend()
        plt.show()
    return np.array(corrects_list).reshape(len(fs),len(Ns))

# Hàm: N_runs_capacity_graphs
# Đầu vào: N_runs (số lần chạy), Ns (danh sách số lượng hình ảnh), imgs (dữ liệu ảnh), beta (tham số cập nhật), fs (các hàm đo độ tương đồng), fn_labels (nhãn hàm), image_perturb_fn (hàm biến đổi ảnh)
#          sep_fn (hàm tách), sep_param (tham số tách), sigma (độ lệch chuẩn), sname (tên tệp để lưu dữ liệu), figname (tên tệp để lưu đồ thị), load_data (có tải dữ liệu hay không), plot_results (có vẽ hay không), save_continuously (có lưu liên tục không)
# Tác dụng: Chạy nhiều lần thử nghiệm lưu trữ ảnh và tính toán tỷ lệ khôi phục chính xác cho các mạng trí nhớ liên kết. Vẽ đồ thị kết quả.
# Đầu ra: Mảng chứa kết quả tỷ lệ khôi phục chính xác
def N_runs_capacity_graphs(N_runs, Ns, imgs, beta,fs,fn_labels, image_perturb_fn = halve_continuous_img, sep_fn = separation_max, sep_param = 1000, sigma=0.1,sname = "tiny_N_capacity_results.npy", figname = "tiny_N_runs_capacity_graph.jpg", load_data = True, plot_results = True, save_continuously=True,normalize_error_threshold = False):
    if not load_data:
        N_corrects = []
        max_N = Ns[-1]
        for n in range(N_runs):
            X = imgs[(max_N*n):(max_N * (n+1))]
            corrects_list = plot_capacity_graphs(Ns, X, beta, fs, fn_labels, image_perturb_fn=image_perturb_fn, sep_fn = sep_fn, sep_param = sep_param, sigma=sigma, normalize_error_threshold=normalize_error_threshold)
            N_corrects.append(corrects_list)
            if save_continuously:
                prelim_N_corrects = np.array(deepcopy(N_corrects))
                np.save(sname, prelim_N_corrects)
        N_corrects = np.array(N_corrects)
        np.save(sname, N_corrects)
    else:
        N_corrects = np.load(sname)
    # begin plot
    if plot_results:
        mean_corrects = np.mean(N_corrects,axis=0)
        std_corrects = np.std(N_corrects, axis=0)
        fig = plt.figure(figsize=(12,10))
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
        sns.despine(left=False,top=True, right=True, bottom=False)
        dataset = parse_sname_for_title(sname)
        plt.title(dataset + " Similarity Functions",fontsize=30)
        for i in range(len(fs)):
            plt.plot(Ns, mean_corrects[i,:],label=fn_labels[i])
            plt.fill_between(Ns, mean_corrects[i,:] - std_corrects[i,:], mean_corrects[i,:]+std_corrects[i,:],alpha=0.5)
        plt.xlabel("Images Stored",fontsize=25)
        plt.ylabel("Fraction Correctly Retrieved",fontsize=25)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
     
        plt.legend(fontsize=25)
        plt.ylim(bottom=0,top=1)
        fig.tight_layout()
        plt.savefig(figname, format=SAVE_FORMAT)
        plt.show()
    return N_corrects


# Hàm: plot_noise_level_graphs
# Đầu vào: N (số lượng hình ảnh), imgs (dữ liệu ảnh), beta (tham số cập nhật), fs (các hàm đo độ tương đồng), labels (nhãn hàm), sigmas (danh sách độ lệch chuẩn)
# Tác dụng: Tính tỷ lệ khôi phục chính xác của mạng trí nhớ liên kết với các mức nhiễu khác nhau.
# Đầu ra: Mảng tỷ lệ khôi phục chính xác cho mỗi mức nhiễu
def plot_noise_level_graphs(N, imgs, beta, fs, labels, sigmas,use_norm = True,sep_fn = separation_max, sep_param=1000):
    corrects_list = [[] for i in range(len(sigmas))]
    for i,sigma in enumerate(sigmas):
        print("SIGMA: ", sigma)
        corrects = [[] for i in range(len(fs))]
        for j, (f,label) in enumerate(zip(fs,labels)):
            print(label)
            N_correct = PC_retrieve_store_continuous(imgs, N, beta=beta, num_plot=0,f=f,sigma=sigma,image_perturb_fn=gaussian_perturb_image,use_norm = use_norm,sep_fn = sep_fn, sep_param = sep_param)
            corrects[j].append(deepcopy(N_correct))
        corrects_list[i].append(np.array(corrects))
    corrects_list = np.array(corrects_list)
    return corrects_list.reshape(len(sigmas), len(fs))

# Hàm: N_runs_noise_level_graphs
# Đầu vào: N_runs (số lần chạy), N (số hình ảnh mỗi lần), imgs (dữ liệu ảnh), beta (tham số), fs (các hàm đo), fn_labels (nhãn hàm),
#          sigmas (độ lệch chuẩn), sep_fn (hàm tách), sep_param (tham số tách), load_data (tải dữ liệu), sname (tên tệp lưu), figname (tên tệp đồ thị)
# Tác dụng: Chạy thử nghiệm với các mức nhiễu khác nhau, tính tỷ lệ khôi phục chính xác và vẽ đồ thị nếu cần.
# Đầu ra: Trả về một mảng chứa tỷ lệ khôi phục chính xác cho mỗi mức nhiễu, qua các lần chạy thử nghiệm.

def N_runs_noise_level_graphs(N_runs, N, imgs, beta,fs,fn_labels, sigmas, sep_fn = separation_max, sep_param = 1000, load_data = False,sname = "tiny_N_noise_level_results.npy", figname = "tiny_N_runs_noise_levels.jpg", plot_results = True):
    if not load_data:
        N_corrects = []
        for n in range(N_runs):
            X = imgs[(N*n):(N * (n+1))]
            corrects_list = plot_noise_level_graphs(N, X, beta, fs, fn_labels, sigmas, sep_fn = sep_fn, sep_param = sep_param)
            N_corrects.append(corrects_list)
        N_corrects = np.array(N_corrects)
        np.save(sname, N_corrects)
    else:
        N_corrects = np.load(sname)

    if plot_results:
        mean_corrects = np.mean(N_corrects,axis=0)
        std_corrects = np.std(N_corrects, axis=0)
        # begin plot
        fig = plt.figure(figsize=(12,10))
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
        sns.despine(left=False,top=True, right=True, bottom=False)
        dataset = parse_sname_for_title(sname)
        plt.title(dataset + " Noise Levels",fontsize=30)
        for i in range(len(fs)):
            plt.plot(sigmas, mean_corrects[:,i],label=fn_labels[i])
            plt.fill_between(sigmas, mean_corrects[:,i] - std_corrects[:,i], mean_corrects[:,i]+std_corrects[:,i],alpha=0.5)
        plt.xlabel("Noise variance (sigma)",fontsize=25)
        plt.ylabel("Fraction Correctly Retrieved",fontsize=25)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        if dataset == "MNIST":
            plt.legend(fontsize=25)
        plt.ylim(bottom=0)
        fig.tight_layout()
        plt.savefig(figname, format=SAVE_FORMAT)
        plt.show()
    return N_corrects



# Hàm: generate_demonstration_reconstructions
# Đầu vào: imgs (dữ liệu ảnh), N (số lượng ảnh), perturb_vals (giá trị biến đổi), sname (tên tệp)
# Tác dụng: Tạo và hiển thị ảnh gốc, ảnh biến đổi, và ảnh tái tạo để minh họa.
# Đầu ra: Hiển thị và lưu ảnh
def generate_demonstration_reconstructions(imgs, N, selected_idx=None, f=normalized_dot_product, image_perturb_fn=mask_continuous_img, perturb_vals=[], sep_fn=separation_softmax, sep_param=1000, use_norm=True, sname=""):
    X = imgs[0:N,:]
    
    img_shape = X[0].shape
    img_len = np.prod(np.array(img_shape))
    if len(img_shape) != 1:
        X = reshape_img_list(X, img_len)
    
    # Sử dụng chỉ số được chọn nếu có, ngược lại chọn ngẫu nhiên
    if selected_idx is not None and selected_idx < N:
        img_idx = selected_idx
    else:
        img_idx = N-1
    
    init_img = deepcopy(X[img_idx,:])
    show_init_img = deepcopy(init_img).reshape(img_shape).permute(1,2,0)
    
    perturbed_imgs = []
    reconstructed_imgs = []
    beta = 1
    for val in perturb_vals:
        query_img = image_perturb_fn(init_img, val).reshape(1, img_len)
        perturbed_imgs.append(deepcopy(query_img.reshape(img_shape).permute(1,2,0)))
        out = general_update_rule(X,query_img,beta, f,sep=sep_fn, sep_param=sep_param,norm=use_norm).reshape(img_len)
        reconstructed_imgs.append(deepcopy(out).reshape(img_shape).permute(1,2,0))
    N_vals = len(perturb_vals)
    ncol = N_vals
    nrow = 3
    fig, ax_array = plt.subplots(nrow, ncol, figsize=(ncol+1,nrow+1), gridspec_kw = {'wspace':0, 'hspace':0, 'top':1.-0.5/(nrow+1), 'bottom': 0.5/(nrow+1), 'left': 0.5/(ncol+1), 'right' :1-0.5/(ncol+1)})
    for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
            if i == 0:
                axes.imshow(show_init_img)
                if j == 0:
                    axes.set_ylabel('Memory', fontsize=12, rotation=0, labelpad=40)  # Đặt nhãn ngang
                if j < N_vals:  # Thêm nhãn cho cột
                    axes.set_title(f'{perturb_vals[j]:.1f}', fontsize=10, pad=10)  # Thêm nhãn số lên
            if i == 1:
                axes.imshow(perturbed_imgs[j])
                if j == 0:  # Chỉ đặt nhãn cho cột đầu tiên
                    axes.set_ylabel('Query', fontsize=12, rotation=0, labelpad=40)  # Đặt nhãn ngangThêm chú thích cho cột
            if i == 2:
                axes.imshow(reconstructed_imgs[j])
                if j == 0:  # Chỉ đặt nhãn cho cột đầu tiên
                    axes.set_ylabel('Output', fontsize=12, rotation=0, labelpad=40)  # Đặt nhãn ngang
            axes.set_aspect("auto")
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.set_xticks([])
            axes.set_yticks([])
    #fig.suptitle("Cifar10 Fraction Masked")
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.tight_layout()
    # Check if the directory exists, if not create it
    import os
    directory = os.path.dirname(sname)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)
    # Lưu hình ảnh với xử lý lỗi
    try:
        plt.savefig(sname, format=SAVE_FORMAT, bbox_inches="tight", pad_inches=0)
        print(f"Figure saved to: {sname}")
    except Exception as e:
        print(f"Error saving figure: {e}")
    plt.show()


# Hàm: run_noise_levels_experiments
# Đầu vào: imgs (dữ liệu ảnh), dataset_str (chuỗi tên tập dữ liệu)
# Tác dụng: Chạy thử nghiệm với các mức độ nhiễu khác nhau và vẽ đồ thị kết quả.
# Đầu ra: Đồ thị tỷ lệ khôi phục chính xác ở các mức nhiễu khác nhau    
def run_noise_levels_experiments(imgs, dataset_str):
    sigmas = [0.05,0.1,0.2,0.3,0.5,0.8,1,1.5]#,2]
    N = 100
    N_runs = 5
    fs = [euclidean_distance, manhatten_distance,normalized_dot_product,KL_divergence,reverse_KL_divergence,Jensen_Shannon_divergence]
    labels = ["Euclidean Distance","Manhattan Distance", "Dot Product", "KL divergence","Reverse KL","Jensen-Shannon"]
    beta = 1
    corrects_list = N_runs_noise_level_graphs(N_runs, N,imgs,beta,fs,labels,sigmas, load_data=LOAD_DATA,plot_results = PLOT_RESULTS,sname=dataset_str + "2_N_noise_level_results.npy", figname = dataset_str + "N_runs_noise_levels_3." + SAVE_FORMAT)
    print(corrects_list.shape)


# Hàm: run_similarity_function_experiments
# Đầu vào: imgs (dữ liệu ảnh), dataset_str (chuỗi tên tập dữ liệu), normalize_error_threshold (có chuẩn hóa ngưỡng lỗi hay không)
# Tác dụng: Thử nghiệm với các hàm đo độ tương đồng khác nhau và vẽ đồ thị.
# Đầu ra: Đồ thị so sánh hiệu suất của các hàm đo độ tương đồng    
def run_similarity_function_experiments(imgs, dataset_str,normalize_error_threshold=False):
    # similarity functions
    Ns = [2,5]
    #Ns = [10,20,50,100,200,300,500,700,1000]
    #longer mnist run
    #Ns = [1500,2000,2500,3000]
    # even longer mnist run
    #Ns = [4000,5000,6000,7000,8000,9000,100000]
    N_runs = 3
    beta = 1000
    #fs = [euclidean_distance, manhatten_distance,normalized_dot_product]#,KL_divergence,reverse_KL_divergence,Jensen_Shannon_divergence]#,cosine_similarity]
    fs = [euclidean_distance]#,cosine_similarity]
    labels = ["Euclidean Distance"]
    corrects_list2 = N_runs_capacity_graphs(N_runs, Ns, imgs, beta,fs,labels,image_perturb_fn = gaussian_perturb_image,sigma=0.5,load_data = LOAD_DATA,plot_results=PLOT_RESULTS,sname = dataset_str + "N_capacity_results.npy", figname = dataset_str + "N_runs_capacity_graph_normalized_2." + SAVE_FORMAT, normalize_error_threshold=normalize_error_threshold)
 
# Hàm: run_example_reconstructions
# Đầu vào: imgs (dữ liệu ảnh), dataset_str (chuỗi tên tập dữ liệu)
# Tác dụng: Tạo và hiển thị các ví dụ về khôi phục ảnh với các mức biến đổi khác nhau.
# Đầu ra: Hiển thị và lưu các ví dụ ảnh 
def run_example_reconstructions(imgs, dataset_str):
    sigmas = [0.05,0.1,0.2,0.3,0.5,0.8,1,1.5]
    mask_fracs  = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    # generate_demonstration_reconstructions(imgs, 100, perturb_vals= sigmas)
    # generate_demonstration_reconstructions(imgs, 50, perturb_vals= mask_fracs)   
     # Tạo và lưu hình ảnh với các mức nhiễu
    generate_demonstration_reconstructions(imgs, 100, image_perturb_fn= gaussian_perturb_image, perturb_vals=sigmas, sname="C:\\Users\\Mai Vu Duy\\Desktop\\20241\\Đồ án\\Theory_Associative_Memory\\demonstration_sigmas.png")
    