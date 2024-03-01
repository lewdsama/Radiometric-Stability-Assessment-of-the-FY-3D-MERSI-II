import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import tifffile as tf


def process_data(band3_folder, band4_folder):
    b3_tif_files = read_tif_files(band3_folder)
    b4_tif_files = read_tif_files(band4_folder)
    b3_mean = []
    b3_std = []
    b4_mean = []
    b4_std = []
    results = []
    for tif_file in b3_tif_files:
        # 读取tif文件
        data = tf.imread(tif_file)
        data[data < -999] = np.nan
        data = data[~np.isnan(data)]
        mean_value = np.mean(data)
        std_value = np.std(data)
        b3_mean.append(mean_value)
        b3_std.append(std_value)

    for tif_file in b4_tif_files:
        # 读取tif文件
        data = tf.imread(tif_file)
        data[data < -999] = np.nan
        data = data[~np.isnan(data)]
        mean_value = np.mean(data)
        std_value = np.std(data)
        b4_mean.append(mean_value)
        b4_std.append(std_value)
    b3_mean = np.array(b3_mean)
    b3_std = np.array(b3_std)
    b4_mean = np.array(b4_mean)
    b4_std = np.array(b4_std)
    N = (b3_std / b3_mean + b4_std / b4_mean) / 2 * 100

    results.append((N, b3_mean, b4_mean))
    return results


# 读取tif文件夹中的所有文件
def read_tif_files(folder_path):
    tif_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            file_path = os.path.join(folder_path, filename)
            tif_files.append(file_path)
    return tif_files


def plot_vertical_grid(ax, x_values, linestyle='--', color='gray'):
    for x in x_values:
        ax.axvline(x, linestyle=linestyle, color=color)

def plot_vertical_grid2(ax, x_values, linestyle='--', color='red'):
    for x in x_values:
        ax.axvline(x, linestyle=linestyle, color=color)

def plot_data(results):
    N_values, B3_means, B4_means = zip(*results)

    plt.figure(figsize=(10, 6))

    ax = plt.axes()
    # 绘制左侧数据
    plt.scatter(N_values, B3_means, marker='o', facecolor='darkblue', linestyle='-', linewidths=1,
                label='Band 3', s=18,alpha=0.7)

    # 绘制右侧数据
    plt.scatter(N_values, B4_means, marker='o', facecolor='red', linestyle='-', linewidths=1, label='Band 4', s=18, alpha=0.5)

    vertical_lines = [0.25, 0.5, 1, 1.25]
    plot_vertical_grid(ax, vertical_lines)
    plot_vertical_grid2(ax, [0.75])
    plt.xlabel("N(%)", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.ylabel("Reflectance", fontname="times new roman", fontsize=20, fontweight='bold')
    # plt.title("Band4", fontname="times new roman", fontsize=14, fontweight='bold')
    plt.grid(False)
    plt.xlim(0.1, 1.5)
    plt.ylim(0.65, 1.0)  # 设置y轴坐标范围

    # 设置x轴刻度线
    plt.xticks(fontname="times new roman", fontsize=20)
    ax.set_xticks([i * 0.05 for i in range(0, 30)], minor=True)  # 设置刻度线位置
    ax.set_xticks([i * 0.25 for i in range(1,7)])  # 设置每500的刻度线为次要刻度线
    ax.xaxis.tick_bottom()  # 刻度线朝向内部

    # 设置y轴刻度线
    plt.yticks(fontname="times new roman", fontsize=20)
    ax.set_yticks([i * 0.01 for i in range(66, 101)], minor=True)  # 设置每0.01的刻度线位置
    ax.set_yticks([i * 0.05 for i in range(13, 21)])  # 设置每0.05的刻度线位置为次要刻度线
    ax.yaxis.tick_left()  # 刻度线朝向内部

    # 设置刻度线样式
    ax.tick_params(which='both', direction='in')  # 刻度线宽度
    ax.tick_params(which='major', width=1, length=10, direction='in')  # 主刻度线长度
    ax.tick_params(which='minor', width=1, length=5, direction='in')  # 次要刻度线长度

    # 添加图例
    font_props = {'family': 'times new roman', 'size': 20, 'weight': 'bold'}
    plt.legend(prop=font_props)
    plt.savefig("Left匀质性分析出图.png", dpi=600)
    # plt.savefig("Right匀质性分析出图.png",dpi=600)
    plt.show()


if __name__ == "__main__":
    b3_folder_path = "J:\\DomeC_data\\20190101_20230228_B3_Left"
    b4_folder_path = "J:\\DomeC_data\\20190101_20230228_B4_Left"
    # b3_folder_path = "J:\\DomeC_data\\20190101_20230228_B3_Right"
    # b4_folder_path = "J:\\DomeC_data\\20190101_20230228_B4_Right"
    results = process_data(b3_folder_path, b4_folder_path)
    plot_data(results)
