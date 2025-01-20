import numpy as np 
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


### image transformation functions ###
# function to half the mnist image to use as a probe query
#_____________________________Các hàm biến đổi hình ảnh_________________________________________________________

"""
def halve_continuous_img(img,sigma=None):
	# mnist
	if len(img) == (28 * 28):
		H,W = img.reshape(28,28).shape
		i = deepcopy(img.reshape(28,28))
		i[H//2:H,:] = 0
		return i
	# cifar -- this is a bit of a hack doing it like this
	elif len(img) == (32 * 32 * 3):
		C,H,W = img.reshape(3,32,32).shape
		i = deepcopy(img.reshape(3,32,32))
		i[:,H//2:H,:] = 0
		return i
	else:
		raise ValueError("Input data dimensions not recognized")
""" 
#Tác dụng: Hàm này dùng mask_continuous_img để che đi một nửa hình ảnh (phần trên hoặc phần dưới). 
#Đầu ra: Hình ảnh bị che một nửa (dạng ma trận 2D hoặc 3D).
def halve_continuous_img(img, sigma=None,reversed = False):
    return mask_continuous_img(img, 0.5, reversed = reversed)

# Tác dụng: Che một phần của hình ảnh theo tỷ lệ frac_masked. Nếu reversed là True, phần trên của hình ảnh sẽ bị che, ngược lại phần dưới sẽ bị che.
# Đầu ra: Hình ảnh đã bị che một phần.
def mask_continuous_img(img, frac_masked,reversed = False):
	# mnist
	frac_masked = 1-frac_masked
	if len(img) == (28*28):
		i = deepcopy(img.reshape(28,28))
		H,W = i.shape
		if reversed:
			i[0:int(H * frac_masked),:] = 0
		else:
			i[int(H * frac_masked):H,:] = 0
		return i
	elif len(img) == (32 * 32 * 3):
		i = deepcopy(img.reshape(3,32,32))
		C,H,W = i.shape
		if reversed:
			i[:,0:int(H*frac_masked),:] = 0
		else:
			i[:,int(H*frac_masked):H,:] = 0
		return i
	# imagenet
	elif len(img) == (64 * 64 * 3):
		i = deepcopy(img.reshape(3,64,64))
		C,H,W = i.shape
		if reversed:
			i[:,0:int(H*frac_masked),:] = 0
		else:
			i[:,int(H*frac_masked):H,:] = 0
		return i
	else:
		raise ValueError("Input data dimensions not recognized")

# Tác dụng: Thêm nhiễu Gaussian vào hình ảnh để làm méo. Mức độ nhiễu được điều khiển bởi sigma.
# Đầu ra: Hình ảnh bị nhiễu, với các giá trị pixel được kẹp lại trong khoảng [0, 1].
def gaussian_perturb_image(img, sigma=0.1):
	#print(img.shape)
	if len(img.shape) != 1:
		total_img_len = np.prod(np.array(img.shape))
		img = img.reshape(total_img_len)
	N = len(img)
	variance = torch.tensor(np.identity(N) * sigma).float()
	perturb = torch.normal(0,sigma,size=[N,])
	return torch.clamp(torch.abs(img + perturb),0,1)

# Tác dụng: Che ngẫu nhiên các điểm ảnh trong hình ảnh với xác suất mask_prob. Điểm ảnh bị che sẽ có giá trị là 0.
# Đầu ra: Hình ảnh đã được che ngẫu nhiên.
def random_mask_frac(img, mask_prob):
    img_shape = img.shape
    flat_img = deepcopy(img).flatten()
    for i in range(len(flat_img)):
        r = np.random.uniform(0,1)
        if r <= mask_prob:
            flat_img[i] = 0.0
    return flat_img.reshape(img_shape)

# Tác dụng: Tương tự như random_mask_frac, nhưng hỗ trợ cả hình ảnh màu. Hàm này che điểm ảnh trên cả ba kênh màu RGB.
# Đầu ra: Hình ảnh màu hoặc trắng đen bị che ngẫu nhiên.
def random_mask_frac_handle_color(img, mask_prob):
	img_shape = img.shape
	if len(img) == 28*28:
		return random_mask_frac(img, mask_prob)
	elif len(img) == 32*32*3:
		reshp = deepcopy(img).reshape(3,32,32)
		for i in range(32):
			for j in range(32):
				r = np.random.uniform(0,1)
				if r <= mask_prob:
					reshp[:,i,j] = 0
		return reshp.reshape(img_shape)
	elif len(img) == 64*64*3:
		reshp = deepcopy(img).reshape(3,64,64)
		for i in range(64):
			for j in range(64):
				r = np.random.uniform(0,1)
				if r <= mask_prob:
					reshp[:,i,j] = 0
		return reshp.reshape(img_shape)
	else:
		raise ValueError("image shape not recognized")
					

# Tác dụng: Che một phần hình ảnh quanh biên với tỷ lệ mask_frac. Điều này tạo ra hiệu ứng như làm mất dữ liệu xung quanh biên hình ảnh.
# Đầu ra: Hình ảnh bị che biên.
def image_inpaint(img, mask_frac):
    #pixels_to_mask = 
	if len(img) == (28*28):
		i = deepcopy(img.reshape(28,28))
		H,W = i.shape
		pixels_to_mask = int(H * mask_frac // 2)
		i[0:pixels_to_mask,:] = 0
		i[28-pixels_to_mask:28,:] = 0
		i[:, 0:pixels_to_mask] = 0
		i[:, 28 - pixels_to_mask:28] = 0
		return i
	elif len(img) == (32 * 32 * 3):
		i = deepcopy(img.reshape(3,32,32))
		C,H,W = i.shape
		pixels_to_mask = int(H * mask_frac // 2)
		i[:,0:pixels_to_mask,:] = 0
		i[:,32-pixels_to_mask:32,:] = 0
		i[:,:, 0:pixels_to_mask] = 0
		i[:,:, 32 - pixels_to_mask:32] = 0
		return i
	# imagenet
	elif len(img) == (64 * 64 * 3):
		i = deepcopy(img.reshape(3,64,64))
		C,H,W = i.shape
		pixels_to_mask = int(H * mask_frac // 2)
		i[:,0:pixels_to_mask,:] = 0
		i[:,64-pixels_to_mask:64,:] = 0
		i[:,:, 0:pixels_to_mask] = 0
		i[:,:, 64 - pixels_to_mask:64] = 0
		return i
	else:
		raise ValueError("Input data dimensions not recognized")


#_____________________________Các hàm chuyển đổi dữ liệu_________________________________________________________

# Tác dụng: Chuyển đổi dữ liệu nhị phân (0, 1) thành dữ liệu lưỡng cực (-1, 1). Dữ liệu này thường dùng trong mạng Hopfield.
# Đầu ra: Dữ liệu lưỡng cực.
def binary_to_bipolar(x):
    return torch.sign(x - 0.5)

# Tác dụng: Chuyển đổi dữ liệu lưỡng cực (-1, 1) thành dữ liệu nhị phân (0, 1).
# Đầu ra: Dữ liệu nhị phân
def bipolar_to_binary(x):
  	return (x + 1) /2

### update functions ###
#_____________________________Các hàm cập nhật bộ nhớ liên kết_________________________________________________________

EPSILON = 1e-4
# update rule of the Modern Continuous Hopfield Network which is closely related to attention etc. Computes similarity scores according in the dot-product or cosine similarity space
#Tác dụng: Cập nhật vector z dựa trên mạng Hopfield liên tục hiện đại, sử dụng điểm tương đồng dot-product. Nếu norm là True, vector được chuẩn hóa.
#Đầu ra: Vector z được cập nhật.
def MCHN_update_rule(X,z,beta, norm=True):
	out = X.T @ F.softmax(beta * X @ z,dim=0)
	if norm:
		return out / torch.sum(out)
	else:
		return out

# PC associative memory update rule -- Computes similarity scores in the euclidean distance/ prediction error space
# Tác dụng: Cập nhật vector z dựa trên lỗi dự đoán giữa X và z, với hàm lỗi mặc định là bình phương.
# Đầu ra: Vector z được cập nhật
def PC_update_rule(X,z,beta,f = torch.square):
	e = z - X # compute prediction error
	return X.T @ F.softmax(beta * -torch.sum(f(e), axis=1))


#_____________________________Hàm đo độ tương đồng hoặc khoảng cách_________________________________________________________

# Tác dụng: Tính điểm tương đồng giữa X và z bằng tích vô hướng (dot product).
# Đầu ra: Giá trị điểm tương đồng.
def dot_product_distance(X,z):
	return X @ z.reshape(len(z[0,:]))

# Tác dụng: Tính điểm tương đồng bằng tích vô hướng chuẩn hóa, nhằm đảm bảo các giá trị nằm trong khoảng [0, 1].
# Đầu ra: Giá trị điểm tương đồng chuẩn hóa.
def normalized_dot_product(X,z):
	norm_X = X / torch.sum(X, axis=1).reshape(X.shape[0],1)
	norm_z = z / torch.sum(z)
	dots = norm_X @ norm_z.reshape(len(z[0,:]))
	recip = dots
	norm_dot = recip / torch.sum(recip)
	return norm_dot

# Tác dụng: Tính độ tương đồng cosin giữa hai vector, dựa trên góc giữa chúng.
# Đầu ra: Giá trị độ tương đồng cosin.
def cosine_similarity(X,z):
	return (X @ z.reshape(len(z[0,:]))) / (torch.norm(X) * torch.norm(z))

# Tác dụng: Tính khoảng cách Manhattan (tổng độ chênh lệch tuyệt đối) giữa X và z.
# Đầu ra: Giá trị khoảng cách.
def manhatten_distance(X,z):
	return 1/torch.sum(torch.abs(z - X),axis=1)
	#return -torch.sum(torch.abs(z - X), axis=1)

# Tác dụng: Tính khoảng cách Euclid (tổng bình phương độ chênh lệch) giữa X và z.
# Đầu ra: Giá trị khoảng cách.
def euclidean_distance(X,z):
	return 1/torch.sum(torch.square(z - X),axis=1)
	#return -torch.sum(torch.square(z - X),axis=1)

#_____________________________Hàm chung cho quy tắc cập nhật_________________________________________________________
# Tác dụng: Cập nhật vector z dựa trên hàm độ tương đồng sim và hàm tách sep. Tùy chọn chuẩn hóa kết quả nếu norm là True.
# Đầu ra: Vector z được cập nhật.
def general_update_rule(X,z,beta,sim, sep=F.softmax,sep_param=1000,norm=True):
	sim_score = sim(X,z)
	print(X.shape)
	#print("SIMS: ", sim_score)
	if norm:
		sim_score = sim_score / torch.sum(sim_score)
	# print(sim_score.shape)
	sep_score = F.softmax(sep_param*sim_score)
	if norm:
		sep_score = sep_score / torch.sum(sep_score)
	
	out = X.T @ sep_score
	# print(out.shape)
	return out



### potential separation functions --  linear, sublinear (sqrt, log), polynomial, exponential, max ###
#_____________________________Hàm tách giá trị_________________________________________________________

def separation_log(x, param):
	return torch.log(x)

def separation_identity(x,param):
	return x

def separation_softmax(x,param):
	return F.softmax(param * x) # param = beta = softmax temperature

def separation_polynomial(x,param):	
    return torch.pow(x, param)

def separation_square(x,param):
	return separation_polynomial(x,2)

def separation_cube(x,param):
	return separation_polynomial(x,3)

def separation_sqrt(x,param):
	return separation_polynomial(x,0.5)

def separation_quartic(x,param):
	return separation_polynomial(x,4)

def separation_ten(x,param):
	return separation_polynomial(x,10)

def separation_max(x, param):
	z = torch.zeros(len(x))
	idx = torch.argmax(x).item()
	z[idx] = 1
	return z # create one hot vector based around the max

# function to iterate through the images, retrieve the output and compute the amount of correctly retrieved image queries
# Mục đích: Hàm reshape_img_list được sử dụng để chuyển đổi một danh sách các hình ảnh thành một ma trận, trong đó mỗi hàng là một hình ảnh 
# được làm phẳng (flattened) thành một vector. Nó cũng có thể áp dụng một hàm tùy chọn để biến đổi các hình ảnh trước khi thêm chúng vào ma trận.
# Đầu ra: Ma trận chứa tất cả các hình ảnh được làm phẳng thành vector. Kích thước của ma trận này sẽ là (số lượng hình ảnh, imglen). 
# Mỗi hàng trong ma trận là một hình ảnh đã được làm phẳng.
def reshape_img_list(imglist, imglen, opt_fn = None):
	new_imglist = torch.zeros(len(imglist), imglen)
	for i,img in enumerate(imglist):
		img = img.reshape(imglen)
		if opt_fn is not None:
			img = opt_fn(img)
		new_imglist[i,:] = img
	return new_imglist


#_____________________________Hàm chính để kiểm tra khả năng nhớ________________________________________________________

# key functions which actually tests the storage capacity of the associative memory
# Tác dụng: Kiểm tra khả năng của mạng trí nhớ liên kết trong việc lưu trữ và khôi phục hình ảnh sau khi chúng bị biến dạng. 
# hàm này so sánh hình ảnh gốc với hình ảnh đã được khôi phục và tính tỷ lệ chính xác.
# Đầu ra: Tỷ lệ chính xác khi khôi phục hình ảnh.
def PC_retrieve_store_continuous(imgs,N, P = None, beta=1,num_plot = 5,similarity_metric="error",f=manhatten_distance, image_perturb_fn = halve_continuous_img,sigma=0.5,sep_fn=separation_max, sep_param=1, use_norm = True,ERROR_THRESHOLD = 60, network_type="", return_sqdiff_outputs = False, plot_example_reconstructions = True):
	X = imgs[0:N,:]
	
	img_len = np.prod(np.array(X[0].shape))
	if len(X.shape) != 2:
		X = reshape_img_list(X, img_len)
	N_correct = 0

	for j in range(N):
		z = image_perturb_fn(X[j,:],sigma).reshape(1,img_len)
		if P is None: # autoassociative
			out = general_update_rule(X,z,beta, f,sep=sep_fn, sep_param=sep_param,norm=use_norm).reshape(img_len)
			if network_type == "classical_hopfield":
				out = binary_to_bipolar(torch.sign(out))
			sqdiff = torch.sum(torch.square(X[j,:] - out))
			if plot_example_reconstructions:
				plt.imshow(X[j,:].reshape(3,32,32).permute(1,2,0))
				plt.show()
				plt.imshow(z.reshape(3,32,32).permute(1,2,0))
				plt.show()
				plt.imshow(out.reshape(3,32,32).permute(1,2,0))
				plt.show()
				print("SQDIFF: ", sqdiff)

		if torch.abs(sqdiff) <= ERROR_THRESHOLD:
			N_correct +=1
		if j < num_plot:
			fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
			titles = ["Original","Masked","Reconstruction"]
			plot_list = [X[j,:], z, out]
			for i, ax in enumerate(axs.flatten()):
				plt.sca(ax)
				#print(imgs[0].shape)
				if len(imgs[0].shape) == 3:
					plt.imshow(plot_list[i].reshape(imgs[0].shape).permute(1,2,0))
				else:
					plt.imshow(plot_list[i].reshape(28,28))
				plt.title(titles[i])
			plt.show()
	return N_correct / N



