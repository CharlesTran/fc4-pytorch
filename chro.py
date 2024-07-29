import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

EPS = 1e-6

def rgb_to_log_chroma(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32)
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]
    log_R = np.log(R + EPS)
    log_G = np.log(G + EPS)
    log_B = np.log(B + EPS)
    log_chroma_RG = log_R - log_G
    log_chroma_RB = log_R - log_B
    return log_chroma_RG, log_chroma_RB

def compute_histogram(chroma_input, hist_boundary, nbins, rgb_input=None):
    eps = np.sum(np.abs(hist_boundary)) / (nbins - 1)
    hist_boundary = np.sort(hist_boundary)
    A_u = np.arange(hist_boundary[0], hist_boundary[1] + eps / 2, eps)
    A_v = np.flip(A_u)
    if rgb_input is None:
        Iy = np.ones(chroma_input.shape[0])
    else:
        Iy = np.sqrt(np.sum(rgb_input ** 2, axis=1))
    diff_u = np.abs(np.tile(chroma_input[:, 0], (len(A_u), 1)).transpose() -
                    np.tile(A_u, (len(chroma_input[:, 0]), 1)))
    diff_v = np.abs(np.tile(chroma_input[:, 1], (len(A_v), 1)).transpose() -
                    np.tile(A_v, (len(chroma_input[:, 1]), 1)))
    diff_u[diff_u > eps] = 0
    diff_u[diff_u != 0] = 1
    diff_v[diff_v > eps] = 0
    diff_v[diff_v != 0] = 1
    Iy_diff_v = np.tile(Iy, (len(A_v), 1)) * diff_v.transpose()
    N = np.matmul(Iy_diff_v, diff_u)
    norm_factor = np.sum(N) + EPS
    N = np.sqrt(N / norm_factor)
    return N

def plot_histogram(histogram, xedges, yedges):
    plt.imshow(histogram.T, origin='lower', interpolation='nearest', aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar()
    plt.xlabel('log(R/G)')
    plt.ylabel('log(R/B)')
    plt.title('Log-Chroma Histogram')
    plt.savefig("1.png")

def convert_to_tensor(histogram):
    tensor = torch.tensor(histogram, dtype=torch.float32)
    return tensor

def main(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    
    log_chroma_RG, log_chroma_RB = rgb_to_log_chroma(image)
    
    # 设置边界值为 [-12, 12]
    hist_boundary = [-3, 3]
    
    # 计算直方图
    nbins = 256
    chroma_input = np.stack((log_chroma_RG, log_chroma_RB), axis=-1).reshape(-1, 2)
    histogram = compute_histogram(chroma_input, hist_boundary, nbins)
    
    # 绘制直方图
    plot_histogram(histogram, hist_boundary, hist_boundary)
    
    # 转换直方图为tensor
    histogram_tensor = convert_to_tensor(histogram)
    
    return histogram_tensor

if __name__ == "__main__":
    image_path = "/data/czx/dataset/gehler/images/8D5U5531.png"  # Replace with your 16-bit image path
    histogram_tensor = main(image_path)
    print(histogram_tensor)
