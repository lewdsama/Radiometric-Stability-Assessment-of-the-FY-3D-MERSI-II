import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tifffile as tf
from scipy.optimize import curve_fit
import seaborn as sns
from scipy.stats import linregress


# 读取SolZ文件夹中的所有文件
def read_solz_files(folder_path):
    solz_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            file_path = os.path.join(folder_path, filename)
            solz_files.append(file_path)
    return solz_files


# 计算TOA_Modeled值
def calculate_toa_modeled(solz_data, b0, b1, b2):
    return b0 + b1 * np.cos(solz_data) + b2 * np.cos(solz_data) * np.cos(solz_data)


# 处理TOAtif文件
def process_tif_files(folder_path):
    tif_files = read_tif_files(folder_path)
    results = []
    dates = []
    for tif_file in tif_files:
        # 解析时间
        file_date = tif_file.split('_')[5]
        date_object = datetime.strptime(file_date, "%Y%m%d")
        dates.append((date_object - datetime(2019, 1, 1)).days)  # 计算与20190101的差值
        # 读取tif文件
        data = tf.imread(tif_file)
        data[data < -999] = np.nan
        data = data[~np.isnan(data)]
        # 剔除平均值和标准差大于5%的数据点
        mean_value, std_deviation = calculate_mean_and_std(data)
        # print(std_deviation)
        valid_data = data[np.abs(data - mean_value) <= 0.05 * std_deviation]

        # 计算剔除后的平均值和标准差
        mean_value, _ = calculate_mean_and_std(valid_data)
        # print(str(file_date) + ' TOA: ' + str(mean_value))
        results.append(mean_value)
    return dates, results


# 读取tif文件夹中的所有文件
def read_tif_files(folder_path):
    tif_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            file_path = os.path.join(folder_path, filename)
            tif_files.append(file_path)
    return tif_files


# 计算数组的平均值和标准差
def calculate_mean_and_std(data):
    mean_value = np.mean(data)
    std_deviation = np.std(data)
    return mean_value, std_deviation


# 处理SolZ文件
def process_solz_files(folder_path):
    solz_files = read_solz_files(folder_path)
    solz_data = []
    for solz_file in solz_files:
        # 解析时间
        file_date = solz_file.split('_')[5]
        date_object = datetime.strptime(file_date, "%Y%m%d")
        days_from_20190101 = (date_object - datetime(2019, 1, 1)).days
        solz_values = tf.imread(solz_file)
        solz_values[(solz_values > 2 * np.pi) | (solz_values < -2 * np.pi)] = np.nan
        solz_values = solz_values[~np.isnan(solz_values)]
        solz_average = np.mean(solz_values)
        solz_data.append((days_from_20190101, solz_average))  # 存储日期和SolZ数据
    return solz_data


# 定义拟合函数
def fit_func(data, b00, b10, b20, b01, b11, b21, b02, b12, b22, b03, b13, b23):
    # solz = solz_data
    # senz = senz_data
    # rza = rza_data
    #
    # a0 = b00 + b01 * np.cos(solz) + b02 * np.cos(solz) * np.cos(solz)
    # a1 = b10 + b11 * np.cos(solz) + b12 * np.cos(solz) * np.cos(solz)
    # a2 = b20 + b21 * np.cos(solz) + b22 * np.cos(solz) * np.cos(solz)
    # a3 = b30 + b31 * np.cos(solz) + b32 * np.cos(solz) * np.cos(solz)
    # c1 = a0 + a1 * (1 - np.cos(senz))
    # c2 = a2 * (1 - np.cos(senz))
    # c3 = a3 * (1 - np.cos(senz))
    # p_predicted = c1 + c2 * np.cos(np.pi - rza) + c3 * np.cos(2 * (np.pi - rza))
    solz, rza, senz = data

    p_predicted = (
            (b00 + b10 * np.cos(solz) + b20 * np.cos(solz) * np.cos(solz)) +
            (b01 + b11 * np.cos(solz) + b21 * np.cos(solz) * np.cos(solz)) * (1 - np.cos(senz)) +
            (b02 + b12 * np.cos(solz) + b22 * np.cos(solz) * np.cos(solz)) * (1 - np.cos(senz)) * np.cos(
        np.pi - rza) +
            (b03 + b13 * np.cos(solz) + b23 * np.cos(solz) * np.cos(solz)) * (1 - np.cos(senz)) * np.cos(
        2 * (np.pi - rza))
    )
    return p_predicted


# 计算拟合参数
def calculate_fit_parameters(data, p_data):
    popt, _ = curve_fit(fit_func, data, p_data)
    return popt


# 主处理函数
def process_data(b3_folder_path, solz_folder_path, senz_folder_path, sola_folder_path, sena_folder_path):
    b3_dates, b3_results = process_tif_files(b3_folder_path)

    solz_data = process_solz_files(solz_folder_path)
    sena_data = process_solz_files(sena_folder_path)
    senz_data = process_solz_files(senz_folder_path)
    sola_data = process_solz_files(sola_folder_path)

    sola_data = np.array(sola_data)

    sena_data = np.array(sena_data)
    rza_data = np.abs(sola_data - sena_data)

    # 计算拟合参数
    solz_values = np.array([item[1] for item in solz_data]).flatten()
    senz_values = np.array([item[1] for item in senz_data]).flatten()
    rza_values = np.array([item[1] for item in rza_data]).flatten()

    input_data = (solz_values, senz_values, rza_values)
    # print(np.array(input_data).shape)

    fit_params = calculate_fit_parameters(input_data, b3_results)

    b00, b10, b20, b01, b11, b21, b02, b12, b22, b03, b13, b23 = fit_params
    print(b00, b10, b20, b01, b11, b21, b02, b12, b22, b03, b13, b23)

    # # 将 fit_params 中的数据分组为每三个一组
    # grouped_params = [fit_params[i:i + 3] for i in range(0, len(fit_params), 3)]
    #
    # # 遍历分组后的数据并输出
    # for group in grouped_params:
    #     b00, b01, b02 = group
    #     output_string = f"b00={b00} , b01={b01} , b02={b02} "
    #     print(output_string)

    # 计算p_predicted值

    p_predicted_results = []
    for solz_value, rza_value, senz_value in zip(*input_data):
        p_predicted_value = fit_func((solz_value, rza_value, senz_value),b00, b10, b20, b01, b11, b21, b02, b12, b22, b03, b13, b23)
        p_predicted_results.append(p_predicted_value)

    return b3_dates, b3_results, p_predicted_results


# 绘制TOA/TOA_Modeled
# def plot_graph(dates, results):
#     plt.figure(figsize=(10, 6))
#     plt.plot(dates, results, marker='o', linestyle='-', color='b')
#     plt.xlabel("Days from 20190101")
#     plt.ylabel("Normalized Value (B3_LeftTargetArea/TOA_Modeled)")
#     plt.title("20190101-20200229_TOA'/TOAModeled_LeftTargetArea ")
#     plt.grid(True)
#     plt.show()
# 绘制 B3和TOA_Modeled比较的图片
def plot_graph(dates, b3_results, toa_modeled_results):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, b3_results, marker='o', linestyle='-', color='b', label='B3')
    plt.plot(dates, toa_modeled_results, marker='o', linestyle='-', color='r', label='Modeled B3')
    plt.xlabel("Days from 20190101")
    plt.ylabel("Values")
    plt.title("20190101-20200229 Band3 TOA' & TOAModeled")
    plt.grid(True)
    plt.legend()  # 添加图例，显示不同数据集的标签
    plt.show()


# 绘制TOA/TOA_Modeled
def plot_TOA_TOA_Modeled(dates, TOA, TOA_Modeled):
    plt.figure(figsize=(10, 6))
    TOA = np.array(TOA)
    TOA_Modeled = np.array(TOA_Modeled)
    results = TOA / TOA_Modeled
    plt.scatter(dates, results, marker='o', linestyle='-', color='b')
    plt.xlabel("Days from 20190101")
    plt.ylabel("BRDF Norm Ref., B4")
    # plt.ylabel("BRDF Norm Ref., B4")
    plt.title("20190101_20230228_TOA'/TOAModeled_LeftTargetArea ")
    plt.grid(True)
    plt.ylim(0.8, 1.2)
    plt.show()


# 绘制B3TOA/B3TOA_Modeled
def plot_B3_TOA_TOA_Modeled(dates, TOA_left, TOA_Modeled_left, TOA_right, TOA_Modeled_right):
    plt.figure(figsize=(10, 6))
    TOA_left = np.array(TOA_left)
    TOA_Modeled_left = np.array(TOA_Modeled_left)

    TOA_right = np.array(TOA_right)
    TOA_Modeled_right = np.array(TOA_Modeled_right)
    leftresults = TOA_left / TOA_Modeled_left
    rightresults = TOA_right / TOA_Modeled_right

    # 多项式拟合
    left_fit = np.polyfit(dates, leftresults, 2)
    right_fit = np.polyfit(dates, rightresults, 2)

    # 合并左右拟合系数并计算整体拟合曲线
    combined_fit = np.mean([left_fit, right_fit], axis=0)
    combined_curve = np.polyval(combined_fit, dates)

    ax = plt.axes()

    # 绘制左侧数据，使用六边形标记
    plt.scatter(dates, leftresults, marker='H', facecolor=(102 / 255, 194 / 255, 165 / 255), edgecolors='black',
                linewidths=1,
                label='LeftTargetArea')

    # 绘制右侧数据，使用三角形标记
    plt.scatter(dates, rightresults, marker='^', facecolor=(252 / 255, 141 / 255, 98 / 255), edgecolors='black',
                linewidths=1,
                label='RightTargetArea')

    # 绘制合并的拟合曲线
    plt.plot(dates, combined_curve, color='blue', linestyle='-', linewidth=2, label='Combined Fit')

    plt.xlabel("Dates", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.ylabel("BRDF Norm. Ref., B3", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.xlim(0, 1600)
    plt.ylim(0.9, 1.1)  # 设置y轴坐标范围
    # 设置x轴刻度线
    plt.xticks(fontname="times new roman", fontsize=20)

    ax.set_xticks([365 * i for i in range(0, 5)])  # 设置刻度线位置
    ax.set_xticklabels(['2019/01/01', '2020/01/01', '2021/01/01', '2022/01/01', '2023/01/01'])  # 设置刻度线标签
    ax.xaxis.tick_bottom()  # 刻度线朝向内部
    # 旋转刻度标签
    plt.xticks(rotation=8)  # 适当调整旋转的角度
    # 设置y轴刻度线
    plt.yticks(fontname="times new roman", fontsize=20)
    ax.set_yticks([i * 0.01 for i in range(91, 111)], minor=True)  # 设置每0.01的刻度线位置
    ax.set_yticks([i * 0.05 for i in range(18, 23)])  # 设置每0.05的刻度线位置为次要刻度线
    ax.yaxis.tick_left()  # 刻度线朝向内部

    # 设置刻度线样式
    ax.tick_params(which='both', direction='in')  # 刻度线宽度
    ax.tick_params(which='major', width=1, length=10, direction='in')  # 主刻度线长度
    ax.tick_params(which='minor', width=1, length=5, direction='in')  # 次要刻度线长度

    # 添加图例，将拟合曲线放在左右两个区域之后
    font_props = {'family': 'times new roman', 'size': 20, 'weight': 'bold'}
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 0]  # 调整图例的顺序
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop=font_props)
    plt.tight_layout()
    plt.savefig("B3_BRDFNorm_原始公式.png", dpi=600)
    plt.show()


# 绘制B4TOA/B4TOA_Modeled
def plot_B4_TOA_TOA_Modeled(dates, TOA_left, TOA_Modeled_left, TOA_right, TOA_Modeled_right):
    plt.figure(figsize=(10, 6))
    TOA_left = np.array(TOA_left)
    TOA_Modeled_left = np.array(TOA_Modeled_left)

    TOA_right = np.array(TOA_right)
    TOA_Modeled_right = np.array(TOA_Modeled_right)
    leftresults = TOA_left / TOA_Modeled_left
    rightresults = TOA_right / TOA_Modeled_right

    # 多项式拟合
    left_fit = np.polyfit(dates, leftresults, 2)
    right_fit = np.polyfit(dates, rightresults, 2)

    # 合并左右拟合系数并计算整体拟合曲线
    combined_fit = np.mean([left_fit, right_fit], axis=0)
    combined_curve = np.polyval(combined_fit, dates)

    ax = plt.axes()

    # 绘制左侧数据，使用六边形标记
    plt.scatter(dates, leftresults, marker='H', facecolor=(102 / 255, 194 / 255, 165 / 255), edgecolors='black',
                linewidths=1,
                label='LeftTargetArea')

    # 绘制右侧数据，使用三角形标记
    plt.scatter(dates, rightresults, marker='^', facecolor=(252 / 255, 141 / 255, 98 / 255), edgecolors='black',
                linewidths=1,
                label='RightTargetArea')

    # 绘制合并的拟合曲线
    plt.plot(dates, combined_curve, color='blue', linestyle='-', linewidth=2, label='Combined Fit')

    plt.xlabel("Dates", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.ylabel("BRDF Norm. Ref., B4", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.xlim(0, 1600)
    plt.ylim(0.9, 1.1)  # 设置y轴坐标范围
    # 设置x轴刻度线
    plt.xticks(fontname="times new roman", fontsize=20)
    ax.set_xticks([365 * i for i in range(0, 5)])  # 设置刻度线位置
    ax.set_xticklabels(['2019/01/01', '2020/01/01', '2021/01/01', '2022/01/01', '2023/01/01'])  # 设置刻度线标签
    ax.xaxis.tick_bottom()  # 刻度线朝向内部
    # 旋转刻度标签
    plt.xticks(rotation=8)  # 适当调整旋转的角度

    # 设置y轴刻度线
    plt.yticks(fontname="times new roman", fontsize=20)
    ax.set_yticks([i * 0.01 for i in range(91, 111)], minor=True)  # 设置每0.01的刻度线位置
    ax.set_yticks([i * 0.05 for i in range(18, 23)])  # 设置每0.05的刻度线位置为次要刻度线
    ax.yaxis.tick_left()  # 刻度线朝向内部

    # 设置刻度线样式
    ax.tick_params(which='both', direction='in')  # 刻度线宽度
    ax.tick_params(which='major', width=1, length=10, direction='in')  # 主刻度线长度
    ax.tick_params(which='minor', width=1, length=5, direction='in')  # 次要刻度线长度

    # 添加图例，将拟合曲线放在左右两个区域之后
    font_props = {'family': 'times new roman', 'size': 20, 'weight': 'bold'}
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 0]  # 调整图例的顺序
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop=font_props)
    plt.tight_layout()
    plt.savefig("B4_BRDFNorm_原始公式.png", dpi=600)
    plt.show()


def plot_solz_brdfnorm(solz, leftresults, rightresults):
    plt.figure(figsize=(10, 6))
    # 绘制左侧数据
    # print(solz.shape,leftresults.shape,rightresults.shape)
    left_reg = sns.regplot(x=solz, y=leftresults, color='darkblue', scatter_kws={'s': 50}, ci=95, label='Band 3')

    # 获取左侧回归直线的斜率和截距
    left_slope, left_intercept, left_rvalue, _, _ = linregress(solz, leftresults)
    left_equation = f"Band 3 Fit: y = {left_slope:.10f}x + {left_intercept:.10f}, R-squared = {left_rvalue ** 2:.10f}"
    print(left_equation)

    # 绘制右侧数据
    right_reg = sns.regplot(x=solz, y=rightresults, color='red', scatter_kws={'s': 50}, ci=95, label='Band 4')

    # 获取右侧回归直线的斜率和截距
    right_slope, right_intercept, right_rvalue, _, _ = linregress(solz, rightresults)
    right_equation = f"Band 4 Fit: y = {right_slope:.10f}x + {right_intercept:.10f}, R-squared = {right_rvalue ** 2:.10f}"
    print(right_equation)

    plt.xlabel("Solar Zenith Angle (°)", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.ylabel("BRDF Norm. Ref.", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.xlim(57, 77)
    plt.ylim(0.9, 1.1)

    ax = plt.gca()
    ax.set_xticks(range(57, 78, 1), minor=True)
    ax.set_xticks(range(60, 76, 5))
    ax.xaxis.tick_bottom()

    ax.set_yticks([i * 0.01 for i in range(90, 111)], minor=True)
    ax.set_yticks([i * 0.05 for i in range(18, 23)])
    ax.yaxis.tick_left()

    # 设置刻度线样式
    ax.tick_params(which='both', direction='in')  # 刻度线宽度
    ax.tick_params(which='major', width=1, length=10, direction='in')  # 主刻度线长度
    ax.tick_params(which='minor', width=1, length=5, direction='in')  # 次要刻度线长度

    plt.xticks(fontname="times new roman", fontsize=20)
    plt.yticks(fontname="times new roman", fontsize=20)

    font_props = {'family': 'times new roman', 'size': 20, 'weight': 'bold'}
    legend = plt.legend(prop=font_props)
    legend.get_frame().set_alpha(0.5)
    plt.tight_layout()
    # plt.savefig('原始模型_Left_solz和brdf norm.ref.的关系图.png', dpi=600)
    plt.savefig('原始模型_Right_solz和 norm.ref.的关系图.png', dpi=600)
    plt.show()


# 示例用法
if __name__ == "__main__":
    b3_left_folder_path = "J:\\DomeC_data\\20190101_20230228_B3_Left"
    b4_left_folder_path = "J:\\DomeC_data\\20190101_20230228_B4_Left"
    left_solz_folder_path = "J:\\DomeC_data\\20190101_20230228_SolZ_Left"
    b3_right_folder_path = "J:\\DomeC_data\\20190101_20230228_B3_Right"
    b4_right_folder_path = "J:\\DomeC_data\\20190101_20230228_B4_Right"
    right_solz_folder_path = "J:\\DomeC_data\\20190101_20230228_SolZ_Right"

    left_senz_folder_path = "J:\\DomeC_data\\20190101_20230228_SenZ_Left"
    right_senz_folder_path = "J:\\DomeC_data\\20190101_20230228_SenZ_Right"
    left_sola_folder_path = "J:\\DomeC_data\\20190101_20230228_SolA_Left"
    right_sola_folder_path = "J:\\DomeC_data\\20190101_20230228_SolA_Right"
    left_sena_folder_path = "J:\\DomeC_data\\20190101_20230228_SenA_Left"
    right_sena_folder_path = "J:\\DomeC_data\\20190101_20230228_SenA_Right"

    dates, left_B3_TOA, left_B3_TOA_Modeled = process_data(b3_left_folder_path, left_solz_folder_path,
                                                           left_senz_folder_path, left_sola_folder_path,
                                                           left_sena_folder_path)
    dates, right_B3_TOA, right_B3_TOA_Modeled = process_data(b3_right_folder_path, right_solz_folder_path,
                                                             right_senz_folder_path, right_sola_folder_path,
                                                             right_sena_folder_path)
    _, left_B4_TOA, left_B4_TOA_Modeled = process_data(b4_left_folder_path, left_solz_folder_path,
                                                       left_senz_folder_path, left_sola_folder_path,
                                                       left_sena_folder_path)

    _, right_B4_TOA, right_B4_TOA_Modeled = process_data(b4_right_folder_path, right_solz_folder_path,
                                                         right_senz_folder_path, right_sola_folder_path,
                                                         right_sena_folder_path)
    # plot_B3_TOA_TOA_Modeled(dates, left_B3_TOA, left_B3_TOA_Modeled, right_B3_TOA, right_B3_TOA_Modeled)
    # plot_B4_TOA_TOA_Modeled(dates, left_B4_TOA, left_B4_TOA_Modeled, right_B4_TOA, right_B4_TOA_Modeled)

    #残差计算
    B3TOA_residuals = np.abs((np.array(right_B3_TOA) - np.array(right_B3_TOA_Modeled)) / np.array(right_B3_TOA) * 100)

    B3TOA_max_residuals = np.max(B3TOA_residuals)
    B3TOA_min_residuals = np.min(B3TOA_residuals)
    B3TOA_mean_residuals = np.mean(B3TOA_residuals)
    B4TOA_residuals = np.abs((np.array(right_B4_TOA) - np.array(right_B4_TOA_Modeled)) / np.array(right_B4_TOA) * 100)

    B4TOA_max_residuals = np.max(B4TOA_residuals)
    B4TOA_min_residuals = np.min(B4TOA_residuals)
    B4TOA_mean_residuals = np.mean(B4TOA_residuals)
    print(B3TOA_max_residuals, B3TOA_min_residuals, B3TOA_mean_residuals)
    print(B4TOA_max_residuals, B4TOA_min_residuals, B4TOA_mean_residuals)


    # plot_graph(dates, normalized_results)
    # plot_TOA_TOA_Modeled(dates, TOA, TOA_Modeled)
    # left_solz = process_solz_files(left_solz_folder_path)
    # left_solz = np.array(left_solz)
    #
    # left_solz_values = np.array([item[1] for item in left_solz]).flatten()
    #
    #
    # right_solz = process_solz_files(right_solz_folder_path)
    # right_solz = np.array(right_solz)
    # right_solz_values = np.array([item[1] for item in right_solz]).flatten()
    #
    # plot_solz_brdfnorm(left_solz_values / np.pi * 180, np.array(left_B3_TOA) / np.array(left_B3_TOA_Modeled),
    #                    np.array(left_B4_TOA) / np.array(left_B4_TOA_Modeled))
    #
    # plot_solz_brdfnorm(right_solz_values / np.pi * 180, np.array(right_B3_TOA) / np.array(right_B3_TOA_Modeled),
    #                    np.array(right_B4_TOA) / np.array(right_B4_TOA_Modeled))

    # # 计算衰减率 Band3
    # # 输入数据
    # t = np.array(dates)  # 时间数组，包括t1和t2
    # TOA_values = np.array(right_B3_TOA) / np.array(right_B3_TOA_Modeled)  # 对应的归一化后TOA值数组
    #
    #
    # # 定义拟合函数
    # def func(t, a0, a1, a2):
    #     return a0 + a1 * t + a2 * t ** 2
    #
    #
    # # 使用最小二乘法进行拟合
    # params, covariance = curve_fit(func, t, TOA_values)
    #
    # # 得到拟合后的参数
    # a0, a1, a2 = params
    # print("band3 a0, a1, a2=")
    # print(a0, a1, a2)
    # # 计算t1和t2对应的TOA值
    # TOA_t1_fit = func(0, a0, a1, a2)
    # TOA_t2_fit = func(1469, a0, a1, a2)
    #
    # # 计算D_total和D_annual
    # D_total = (TOA_t2_fit - TOA_t1_fit) / TOA_t1_fit
    # D_annual = D_total / (1469) * 365
    # print("Band3: D_total=")
    # print(D_total)
    # print("Band3: D_annual=")
    # print(D_annual)
    # # 计算衰减率 Band4
    # TOA_values = np.array(right_B4_TOA) / np.array(right_B4_TOA_Modeled)
    # params, covariance = curve_fit(func, t, TOA_values)
    #
    # # 得到拟合后的参数
    # a0, a1, a2 = params
    # print("band4 a0, a1, a2=")
    # print(a0, a1, a2)
    # # 计算t1和t2对应的TOA值
    # TOA_t1_fit = func(0, a0, a1, a2)
    # TOA_t2_fit = func(1469, a0, a1, a2)
    #
    # # 计算D_total和D_annual
    # D_total = (TOA_t2_fit - TOA_t1_fit) / TOA_t1_fit
    # D_annual = D_total / (1469) * 365
    # print("Band4: D_total=")
    # print(D_total)
    # print("Band4: D_annual=")
    # print(D_annual)
