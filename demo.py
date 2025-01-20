import gradio as gr
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from functions import *
from data import *
from plotting import *

def run_example_reconstructions(dataset_type="CIFAR10", 
                              num_images=5,
                              image_index=None,
                              similarity_fn="Manhattan Distance",
                              separation_fn="Softmax",
                              beta=1.0,
                              perturbation_type="Gaussian",
                              perturbation_values="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"):
    """
    Demo thuật toán khôi phục ảnh.
    
    Args:
        dataset_type: Loại dataset ("MNIST", "CIFAR10")
        num_images: Số lượng ảnh để chọn ngẫu nhiên
        image_index: Chỉ số của ảnh muốn khôi phục (None để chọn ngẫu nhiên)
        similarity_fn: Hàm tính độ tương đồng
        separation_fn: Hàm tách
        beta: Tham số beta cho quá trình cập nhật
        perturbation_type: Loại nhiễu
        perturbation_values: Các mức độ nhiễu (dạng chuỗi, phân tách bằng dấu phẩy)
    """
    # Tạo tên file output
    output_file = f"demo_{dataset_type}_{perturbation_type}.png"
    
    # Chuyển đổi chuỗi perturbation_values thành list các số thực
    perturb_vals = [float(x.strip()) for x in perturbation_values.split(",")]
    
    # 1. Load dữ liệu
    if dataset_type == "CIFAR10":
        trainset, _ = get_cifar10(num_images)
        imgs = trainset[0][0]
    else:  # MNIST
        trainset, _ = load_mnist(num_images)
        imgs = trainset[0][0]

    # 2. Mapping các hàm
    similarity_functions = {
        "Euclidean Distance": euclidean_distance,
        "Manhattan Distance": manhatten_distance,
        "Dot Product": normalized_dot_product,
        "Cosine Similarity": cosine_similarity
    }
    
    separation_functions = {
        "Softmax": separation_softmax,
        "Max": separation_max,
        "Polynomial": separation_polynomial
    }
    
    perturbation_functions = {
        "Mask": mask_continuous_img,
        "Gaussian": gaussian_perturb_image,
        "Random Mask": random_mask_frac_handle_color,
        "Inpaint": image_inpaint
    }

    # 3. Tạo demo reconstruction với các tham số đã chọn
    generate_demonstration_reconstructions(
        imgs=imgs,
        N=num_images,
        selected_idx=image_index,
        f=similarity_functions[similarity_fn],
        image_perturb_fn=perturbation_functions[perturbation_type],
        perturb_vals=perturb_vals,
        sep_fn=separation_functions[separation_fn],
        use_norm=True,
        sep_param=beta,
        sname=output_file
    )
    
    return output_file

# Tạo giao diện Gradio
iface = gr.Interface(
    fn=run_example_reconstructions,
    inputs=[
        gr.Dropdown(
            choices=["MNIST", "CIFAR10"],
            label="Dataset Type",
            value="CIFAR10"
        ),
        gr.Slider(
            minimum=5,
            maximum=2000,
            value=5,
            step=5,
            label="Number of Images"
        ),
        gr.Number(
            label="Image Index (optional)",
            value=None,
            minimum=0,
            precision=0
        ),
        gr.Dropdown(
            choices=["Manhattan Distance", "Euclidean Distance", 
                    "Dot Product", "Cosine Similarity"],
            label="Similarity Function",
            value="Manhattan Distance"
        ),
        gr.Dropdown(
            choices=["Softmax"],
            label="Separation Function",
            value="Softmax"
        ),
        gr.Slider(
            minimum=1,
            maximum=1000000,
            value=1.0,
            step=1,
            label="Beta Parameter"
        ),
        gr.Dropdown(
            choices=["Gaussian", "Mask" , "Random Mask"],
            label="Perturbation Type",
            value="Gaussian"
        ),
        gr.Textbox(
            label="Perturbation Values (comma-separated)",
            value="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"
        )
    ],
    outputs=gr.Image(type="filepath"),
    title="Memory Reconstruction Demo",
)

if __name__ == "__main__":
    iface.launch()