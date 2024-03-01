import os
import numpy as np
import matplotlib.pyplot as plt

PLT_FIGURE = plt.figure(figsize=(10, 6))
import tifffile as tf
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
from matplotlib.lines import Line2D
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


# 处理tif文件
def process_tif_files(folder_path):
    tif_files = read_tif_files(folder_path)
    results = []
    dates = []
    for tif_file in tif_files:
        # 解析时间
        # print(tif_file)
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
def fit_func(solz_data, b0, b1, b2):
    return b0 + b1 * np.cos(solz_data) + b2 * np.cos(solz_data) * np.cos(solz_data)


# 计算拟合参数
def calculate_fit_parameters(solz_data, b3_data):
    x_data = solz_data
    y_data = b3_data
    popt, _ = curve_fit(fit_func, x_data, y_data)
    return popt


# 主处理函数
def process_data(b3_folder_path, solz_folder_path):
    b3_dates, b3_results = process_tif_files(b3_folder_path)
    solz_data = process_solz_files(solz_folder_path)

    # 计算拟合参数
    solz_values = np.array([item[1] for item in solz_data]).flatten()
    fit_params = calculate_fit_parameters(solz_values, np.array(b3_results))
    b0, b1, b2 = fit_params
    print("b0,b1,b2=")
    print(b0, b1, b2)

    # 计算TOA_Modeled值
    toa_modeled_results = []
    for solz_value in solz_values:
        toa_modeled_value = calculate_toa_modeled(solz_value, b0, b1, b2)
        toa_modeled_results.append(toa_modeled_value)
    print("b3_results, toa_modeled_results=")
    print(b3_results, toa_modeled_results)
    # 计算B3_LeftTargetArea/TOA_Modeled值
    normalized_results = np.array(b3_results) / np.array(toa_modeled_results)

    return b3_dates, normalized_results, b3_results, toa_modeled_results, solz_values


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
    plt.savefig("B3_BRDFNorm_简化公式.png", dpi=600)
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
    plt.savefig("B4_BRDFNorm_简化公式.png", dpi=600)
    plt.show()


# Function to convert days since 2019-01-01 to dates
def days_to_dates(days):
    base_date = datetime(2019, 1, 1)
    return [base_date + timedelta(days=int(day)) for day in days]


# Custom function to set ticks with increased density
# Custom function to set ticks with specific density

# 绘制B3
def plot_B3(dates, leftresults, rightresults):
    plt.figure(figsize=(10, 6))
    ax = plt.axes()

    # 绘制左侧数据，使用六边形标记
    plt.scatter(dates, leftresults, marker='H', facecolor=(102 / 255, 194 / 255, 165 / 255), edgecolors='black',
                linewidths=1,
                label='LeftTargetArea')

    # 绘制右侧数据，使用三角形标记
    plt.scatter(dates, rightresults, marker='^', facecolor=(252 / 255, 141 / 255, 98 / 255), edgecolors='black',
                linewidths=1,
                label='RightTargetArea')

    plt.xlabel("Dates", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.ylabel("Reflectance", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.title("Band3", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.xlim(0, 1600)
    plt.ylim(0.75, 0.95)  # 设置y轴坐标范围

    # 设置x轴刻度线
    plt.xticks(fontname="times new roman", fontsize=20)
    ax.set_xticks([365 * i for i in range(0, 5)])  # 设置刻度线位置
    ax.set_xticklabels(['2019/01/01','2020/01/01', '2021/01/01', '2022/01/01', '2023/01/01'])  # 设置刻度线标签
    ax.xaxis.tick_bottom()  # 刻度线朝向内部
    # 旋转刻度标签
    plt.xticks(rotation=8)  # 适当调整旋转的角度
    # 设置y轴刻度线
    plt.yticks(fontname="times new roman", fontsize=20)
    ax.set_yticks([i * 0.01 for i in range(75, 96)], minor=True)  # 设置每0.01的刻度线位置
    ax.set_yticks([i * 0.05 for i in range(15, 20)])  # 设置每0.05的刻度线位置为次要刻度线
    ax.yaxis.tick_left()  # 刻度线朝向内部

    # 设置刻度线样式
    ax.tick_params(which='both', direction='in')  # 刻度线宽度
    ax.tick_params(which='major', width=1, length=10, direction='in')  # 主刻度线长度
    ax.tick_params(which='minor', width=1, length=5, direction='in')  # 次要刻度线长度

    # 添加图例
    font_props = {'family': 'times new roman', 'size': 18, 'weight': 'bold'}
    plt.legend(prop=font_props)
    plt.tight_layout()
    plt.savefig("B3TOA.png", dpi=600)
    plt.show()


# 绘制B4
def plot_B4(dates, leftresults, rightresults):
    plt.figure(figsize=(10, 6))
    ax = plt.axes()

    # 绘制左侧数据，使用六边形标记
    plt.scatter(dates, leftresults, marker='H', facecolor=(102 / 255, 194 / 255, 165 / 255), edgecolors='black',
                linewidths=1,
                label='LeftTargetArea')

    # 绘制右侧数据，使用三角形标记
    plt.scatter(dates, rightresults, marker='^', facecolor=(252 / 255, 141 / 255, 98 / 255), edgecolors='black',
                linewidths=1,
                label='RightTargetArea')

    plt.xlabel("Dates", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.ylabel("Reflectance", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.title("Band4", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.xlim(0, 1600)
    plt.ylim(0.75, 0.93)  # 设置y轴坐标范围

    # 设置x轴刻度线
    plt.xticks(fontname="times new roman", fontsize=20)
    ax.set_xticks([365 * i for i in range(0, 5)])  # 设置刻度线位置
    ax.set_xticklabels(['2019/01/01','2020/01/01', '2021/01/01', '2022/01/01', '2023/01/01'])  # 设置刻度线标签
    ax.xaxis.tick_bottom()  # 刻度线朝向内部
    # 旋转刻度标签
    plt.xticks(rotation=8)  # 适当调整旋转的角度

    # 设置y轴刻度线
    plt.yticks(fontname="times new roman", fontsize=20)
    ax.set_yticks([i * 0.01 for i in range(75, 94)], minor=True)  # 设置每0.01的刻度线位置
    ax.set_yticks([i * 0.05 for i in range(15, 19)])  # 设置每0.05的刻度线位置为次要刻度线
    ax.yaxis.tick_left()  # 刻度线朝向内部

    # 设置刻度线样式
    ax.tick_params(which='both', direction='in')  # 刻度线宽度
    ax.tick_params(which='major', width=1, length=10, direction='in')  # 主刻度线长度
    ax.tick_params(which='minor', width=1, length=5, direction='in')  # 次要刻度线长度

    # 添加图例
    font_props = {'family': 'times new roman', 'size': 18, 'weight': 'bold'}
    plt.legend(prop=font_props)
    plt.tight_layout()
    plt.savefig("B4TOA.png", dpi=600)
    plt.show()


def plot_solz_ref(solz, leftresults, rightresults):
    plt.figure(figsize=(10, 6))
    # 绘制左侧数据
    left_reg = sns.regplot(x=solz, y=leftresults, color='darkblue', scatter_kws={'s': 50}, ci=95, label='Band 3')

    # 获取左侧回归直线的斜率和截距
    left_slope, left_intercept, left_rvalue, _, _ = linregress(solz, leftresults)
    left_equation = f"Band 3 Fit: y = {left_slope:.4f}x + {left_intercept:.4f}, R-squared = {left_rvalue ** 2:.4f}"
    print(left_equation)

    # 绘制右侧数据
    right_reg = sns.regplot(x=solz, y=rightresults, color='red', scatter_kws={'s': 50}, ci=95, label='Band 4')

    # 获取右侧回归直线的斜率和截距
    right_slope, right_intercept, right_rvalue, _, _ = linregress(solz, rightresults)
    right_equation = f"Band 4 Fit: y = {right_slope:.4f}x + {right_intercept:.4f}, R-squared = {right_rvalue ** 2:.4f}"
    print(right_equation)

    plt.xlabel("Solar Zenith Angle (°)", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.ylabel("Reflectance", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.xlim(57, 77)
    plt.ylim(0.75, 0.93)

    ax = plt.gca()
    ax.set_xticks(range(57, 78, 1), minor=True)
    ax.set_xticks(range(60, 76, 5))
    ax.xaxis.tick_bottom()

    ax.set_yticks([i * 0.01 for i in range(75, 94)], minor=True)
    ax.set_yticks([i * 0.05 for i in range(15, 19)])
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
    # plt.savefig('Left_solz和ref的关系图.png', dpi=600)
    plt.savefig('Right_solz和ref的关系图.png', dpi=600)
    plt.show()


def plot_solz_brdfnorm(solz, leftresults, rightresults):
    plt.figure(figsize=(10, 6))
    # 绘制左侧数据
    left_reg = sns.regplot(x=solz, y=leftresults, color='darkblue', scatter_kws={'s': 50}, ci=95, label='Band 3')

    # 获取左侧回归直线的斜率和截距
    left_slope, left_intercept, left_rvalue, _, _ = linregress(solz, leftresults)
    left_equation = f"Band 3 Fit: y = {left_slope:.4f}x + {left_intercept:.4f}, R-squared = {left_rvalue ** 2:.4f}"
    print(left_equation)

    # 绘制右侧数据
    right_reg = sns.regplot(x=solz, y=rightresults, color='red', scatter_kws={'s': 50}, ci=95, label='Band 4')

    # 获取右侧回归直线的斜率和截距
    right_slope, right_intercept, right_rvalue, _, _ = linregress(solz, rightresults)
    right_equation = f"Band 4 Fit: y = {right_slope:.4f}x + {right_intercept:.4f}, R-squared = {right_rvalue ** 2:.4f}"
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
    # plt.savefig('Left_solz和brdf norm.ref.的关系图.png', dpi=600)
    plt.savefig('Right_solz和 norm.ref.的关系图.png', dpi=600)
    plt.show()
# 绘制B3/B4
def plot_B3B4(dates, B3_left, B4_left, B3_right, B4_right):
    plt.figure(figsize=(10, 6))
    B3_left = np.array(B3_left)
    B4_left = np.array(B4_left)
    B3_right = np.array(B3_right)
    B4_right = np.array(B4_right)
    leftresults = B3_left / B4_left
    rightresults = B3_right / B4_right
    ax = plt.axes()
    # 绘制左侧数据，使用六边形标记
    plt.scatter(dates, leftresults, marker='H', facecolor=(102 / 255, 194 / 255, 165 / 255), edgecolors='black',
                linewidths=1,
                label='LeftTargetArea')

    # 绘制右侧数据，使用三角形标记
    plt.scatter(dates, rightresults, marker='^', facecolor=(252 / 255, 141 / 255, 98 / 255), edgecolors='black',
                linewidths=1,
                label='RightTargetArea')

    plt.xlabel("Dates", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.ylabel("Band Ratio, B3/B4", fontname="times new roman", fontsize=20, fontweight='bold')
    # plt.title("Band4", fontname="times new roman", fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.xlim(0, 1600)
    plt.ylim(0.94, 1.08)  # 设置y轴坐标范围

    # 设置x轴刻度线
    plt.xticks(fontname="times new roman", fontsize=20)
    ax.set_xticks([365 * i for i in range(0, 5)])  # 设置刻度线位置
    ax.set_xticklabels(['2019/01/01','2020/01/01', '2021/01/01', '2022/01/01', '2023/01/01'])  # 设置刻度线标签
    ax.xaxis.tick_bottom()  # 刻度线朝向内部

    # 旋转刻度标签
    plt.xticks(rotation=8)  # 适当调整旋转的角度

    # 设置y轴刻度线
    plt.yticks(fontname="times new roman", fontsize=20)
    ax.set_yticks([i * 0.01 for i in range(95, 109)], minor=True)  # 设置每0.01的刻度线位置
    ax.set_yticks([i * 0.05 for i in range(19, 22)])  # 设置每0.05的刻度线位置为次要刻度线
    ax.yaxis.tick_left()  # 刻度线朝向内部

    # 设置刻度线样式
    ax.tick_params(which='both', direction='in')  # 刻度线宽度
    ax.tick_params(which='major', width=1, length=10, direction='in')  # 主刻度线长度
    ax.tick_params(which='minor', width=1, length=5, direction='in')  # 次要刻度线长度

    # 添加图例
    font_props = {'family': 'times new roman', 'size': 20, 'weight': 'bold'}
    plt.legend(prop=font_props)
    plt.tight_layout()
    plt.savefig("B3除以B4.png", dpi=600)
    plt.show()


# 绘制Solz
def plot_Solz(dates, leftresults, rightresults):
    leftresults = leftresults / np.pi * 180
    rightresults = rightresults / np.pi * 180
    plt.figure(figsize=(10, 6))
    ax = plt.axes()
    # 绘制左侧数据，使用六边形标记
    plt.scatter(dates, leftresults, marker='H', facecolor=(102 / 255, 194 / 255, 165 / 255), edgecolors='black',
                linewidths=1,
                label='LeftTargetArea')

    # 绘制右侧数据，使用三角形标记
    plt.scatter(dates, rightresults, marker='^', facecolor=(252 / 255, 141 / 255, 98 / 255), edgecolors='black',
                linewidths=1,
                label='RightTargetArea')

    plt.xlabel("Dates", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.ylabel("Solar Zenith Angle (°)", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.title("SolZ", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.xlim(0, 1600)
    plt.ylim(55, 81)  # 设置y轴坐标范围

    # 设置x轴刻度线
    plt.xticks(fontname="times new roman", fontsize=20)
    ax.set_xticks([365 * i for i in range(0, 5)])  # 设置刻度线位置
    ax.set_xticklabels(['2019/01/01','2020/01/01', '2021/01/01', '2022/01/01', '2023/01/01'])  # 设置刻度线标签
    ax.xaxis.tick_bottom()  # 刻度线朝向内部
    # 旋转刻度标签
    plt.xticks(rotation=8)  # 适当调整旋转的角度
    # 设置y轴刻度线
    plt.yticks(fontname="times new roman", fontsize=20)
    ax.set_yticks(range(55, 82, 1), minor=True)  # 设置刻度线位置
    ax.set_yticks(range(55, 82, 5))  # 设置每500的刻度线为次要刻度线
    ax.yaxis.tick_left()  # 刻度线朝向内部

    # 设置刻度线样式
    ax.tick_params(which='both', direction='in')  # 刻度线宽度
    ax.tick_params(which='major', width=1, length=10, direction='in')  # 主刻度线长度
    ax.tick_params(which='minor', width=1, length=5, direction='in')  # 次要刻度线长度

    # 添加图例
    font_props = {'family': 'times new roman', 'size': 18, 'weight': 'bold'}
    plt.legend(prop=font_props)
    plt.tight_layout()
    plt.savefig("SOLZ.png", dpi=600)
    plt.show()

# 绘制Senz
def plot_Senz(dates, leftresults, rightresults):
    leftresults = leftresults / np.pi * 180
    rightresults = rightresults / np.pi * 180
    plt.figure(figsize=(10, 6))
    ax = plt.axes()
    # 绘制左侧数据，使用六边形标记
    plt.scatter(dates, leftresults, marker='H', facecolor=(102 / 255, 194 / 255, 165 / 255), edgecolors='black',
                linewidths=1,
                label='LeftTargetArea')

    # 绘制右侧数据，使用三角形标记
    plt.scatter(dates, rightresults, marker='^', facecolor=(252 / 255, 141 / 255, 98 / 255), edgecolors='black',
                linewidths=1,
                label='RightTargetArea')

    plt.xlabel("Dates", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.ylabel("Sensor Zenith Angle (°)", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.title("SenZ", fontname="times new roman", fontsize=20, fontweight='bold')
    plt.grid(True)
    plt.xlim(0, 1600)
    plt.ylim(0, 20)  # 设置y轴坐标范围

    # 设置x轴刻度线
    plt.xticks(fontname="times new roman", fontsize=20)
    ax.set_xticks([365 * i for i in range(0, 5)])  # 设置刻度线位置
    ax.set_xticklabels(['2019/01/01','2020/01/01', '2021/01/01', '2022/01/01', '2023/01/01'])  # 设置刻度线标签
    ax.xaxis.tick_bottom()  # 刻度线朝向内部
    # 旋转刻度标签
    plt.xticks(rotation=8)  # 适当调整旋转的角度

    # 设置y轴刻度线
    plt.yticks(fontname="times new roman", fontsize=20)
    ax.set_yticks(range(0, 20, 1), minor=True)  # 设置刻度线位置
    ax.set_yticks(range(0, 20, 5))  # 设置每500的刻度线为次要刻度线
    ax.yaxis.tick_left()  # 刻度线朝向内部

    # 设置刻度线样式
    ax.tick_params(which='both', direction='in')  # 刻度线宽度
    ax.tick_params(which='major', width=1, length=10, direction='in')  # 主刻度线长度
    ax.tick_params(which='minor', width=1, length=5, direction='in')  # 次要刻度线长度

    # 添加图例
    font_props = {'family': 'times new roman', 'size': 18, 'weight': 'bold'}
    plt.legend(prop=font_props)
    plt.tight_layout()
    plt.savefig("Senz.png", dpi=600)
    plt.show()

def plot_Senz_his(leftresults, rightresults):

    data_left = leftresults
    data_right = rightresults

    # 创建直方图
    plt.figure(figsize=(10, 6))

    # 使用Seaborn绘制直方图，并设置黑色边框
    sns.histplot(data_left, bins=10, alpha=1, element='step',color=(102 / 255, 194 / 255, 165 / 255), edgecolor='black')
    sns.histplot(data_right, bins=10, alpha=1, element='step',color=(252 / 255, 141 / 255, 98 / 255),
                 edgecolor='black')
    # 添加标签和标题
    plt.xlabel('Sensor Zenith Angle (°)', fontname="times new roman", fontsize=20, fontweight='bold')
    plt.ylabel('Frequency', fontname="times new roman", fontsize=20, fontweight='bold')
    plt.title('SenZ Histogram', fontname="times new roman", fontsize=20, fontweight='bold')
    plt.xticks(fontname="times new roman", fontsize=20)
    plt.yticks(fontname="times new roman", fontsize=20)
    plt.tick_params(axis='both', which='both', direction='in', length=6)
    # 添加图例
    font_props = {'family': 'times new roman', 'size': 18, 'weight': 'bold'}
    plt.legend(['LeftTargetArea', 'RightTargetArea'], prop=font_props)

    plt.savefig("Senz_his.png", dpi=600)
    # 显示图形
    plt.show()

# 绘制 B3和TOA_Modeled比较的图片
# def plot_graph(dates, b3_results, toa_modeled_results):
#     plt.figure(figsize=(10, 6))
#     plt.scatter(dates, b3_results, marker='o', linestyle='-', color='b', label='B3')
#     plt.scatter(dates, toa_modeled_results, marker='o', linestyle='-', color='r', label='Modeled B3')
#     plt.xlabel("Days from 20190101")
#     plt.ylabel("Values")
#     plt.title("20190101-20200229 Band3 TOA' & TOAModeled")
#     plt.grid(True)
#     plt.legend()  # 添加图例，显示不同数据集的标签
#     plt.show()

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

    dates, left_B3_normalized_results, left_B3_TOA, left_B3_TOA_Modeled, left_solz = process_data(b3_left_folder_path,
                                                                                                  left_solz_folder_path)

    _, left_B4_normalized_results, left_B4_TOA, left_B4_TOA_Modeled, _ = process_data(b4_left_folder_path,
                                                                                      left_solz_folder_path)

    dates, right_B3_normalized_results, right_B3_TOA, right_B3_TOA_Modeled, right_solz = process_data(
        b3_right_folder_path, right_solz_folder_path)

    _, right_B4_normalized_results, right_B4_TOA, right_B4_TOA_Modeled, _ = process_data(b4_right_folder_path,
                                                                                         right_solz_folder_path)

    _, _, _, _, left_senz = process_data(b3_left_folder_path,left_senz_folder_path)
    _, _, _, _, right_senz = process_data(b3_right_folder_path, right_senz_folder_path)


    # # 计算残差
    #
    # B3TOA_residuals =np.abs((np.array(right_B3_TOA) - np.array(right_B3_TOA_Modeled))/np.array(right_B3_TOA) *100)
    #
    # B3TOA_max_residuals=np.max(B3TOA_residuals)
    # B3TOA_min_residuals = np.min(B3TOA_residuals)
    # B3TOA_mean_residuals = np.mean(B3TOA_residuals)
    #
    # # 计算残差
    # B4TOA_residuals =np.abs((np.array(right_B4_TOA) - np.array(right_B4_TOA_Modeled))/np.array(right_B4_TOA) *100)
    #
    # B4TOA_max_residuals = np.max(B4TOA_residuals)
    # B4TOA_min_residuals = np.min(B4TOA_residuals)
    # B4TOA_mean_residuals = np.mean(B4TOA_residuals)
    # print("B3TOA_max_residuals,B3TOA_min_residuals,B3TOA_mean_residuals)=")
    # print(B3TOA_max_residuals,B3TOA_min_residuals,B3TOA_mean_residuals)
    # print("B4TOA_max_residuals,B4TOA_min_residuals,B4TOA_mean_residuals=")
    # print(B4TOA_max_residuals,B4TOA_min_residuals,B4TOA_mean_residuals)
    #
    # #计算衰减率 Band3
    # # 输入数据
    # t = np.array(dates)  # 时间数组，包括t1和t2
    # TOA_values = np.array(right_B3_TOA)/np.array(right_B3_TOA_Modeled)  # 对应的归一化后TOA值数组
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
    # print("band3 a0,a1,a2=")
    # print(a0,a1,a2)
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
    # TOA_values = np.array(right_B4_TOA)/np.array(right_B4_TOA_Modeled)
    # params, covariance = curve_fit(func, t, TOA_values)
    #
    # # 得到拟合后的参数
    # a0, a1, a2 = params
    # print("band4 a0,a1,a2=")
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

    # 出图
    # # plot_B3(dates, left_B3_TOA, right_B3_TOA)
    # # plot_B4(dates, left_B4_TOA, right_B4_TOA)
    # # plot_Solz(dates, left_solz, right_solz)
    # plot_Senz(dates, left_senz, right_senz)

    #
    plot_Senz_his(-left_senz/ np.pi * 180, right_senz/ np.pi * 180)

    # # plot_B3B4(dates, left_B3_TOA, left_B4_TOA,right_B3_TOA,right_B4_TOA)
    # plot_B3_TOA_TOA_Modeled(dates, left_B3_TOA,left_B3_TOA_Modeled,right_B3_TOA,right_B3_TOA_Modeled)
    # plot_B4_TOA_TOA_Modeled(dates, left_B4_TOA, left_B4_TOA_Modeled, right_B4_TOA, right_B4_TOA_Modeled)
    # plot_TOA_TOA_Modeled(dates, B4_TOA, B4_TOA_Modeled)
    # plot_solz_brdfnorm(left_solz / np.pi * 180,np.array(left_B3_TOA)/np.array(left_B3_TOA_Modeled),np.array(left_B4_TOA)/np.array(left_B4_TOA_Modeled))
    # plot_solz_brdfnorm(right_solz / np.pi * 180, np.array(right_B3_TOA) / np.array(right_B3_TOA_Modeled),
    #                    np.array(right_B4_TOA) / np.array(right_B4_TOA_Modeled))
    # plot_solz_ref(right_solz / np.pi * 180, right_B3_TOA_Modeled, right_B4_TOA_Modeled)




