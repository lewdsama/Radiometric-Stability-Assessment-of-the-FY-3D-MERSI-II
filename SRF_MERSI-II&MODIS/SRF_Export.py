import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
# 文件夹路径
fy3d_folder_path = r'F:\基于FY的MOA\fy3d_srf\srf_for_export\fy3d'
modis_folder_path = r'F:\基于FY的MOA\fy3d_srf\srf_for_export\modis'

# FY-3D的颜色
fy3d_color = 'b'  # 蓝色

# MODIS的颜色
modis_color = 'r'  # 红色

# 绘制FY-3D的前四个曲线
fy3d_legend_label = 'FY-3D MERSI-II'  # FY-3D曲线的图例标签
for filename in os.listdir(fy3d_folder_path)[:4]:
    if filename.endswith('.txt'):
        file_path = os.path.join(fy3d_folder_path, filename)

        with open(file_path, 'r') as file:
            lines = file.readlines()[4:]

        wavelengths = []
        responses = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                wavelength, response = map(float, parts)
                wavelengths.append(1 / wavelength * 10000)
                responses.append(response)

        plt.plot(wavelengths, responses, color=fy3d_color, label=fy3d_legend_label)

# 绘制MODIS的前四个曲线
modis_legend_label = 'MODIS'  # MODIS曲线的图例标签
for filename in os.listdir(modis_folder_path)[:4]:
    if filename.endswith('.txt'):
        file_path = os.path.join(modis_folder_path, filename)

        with open(file_path, 'r') as file:
            lines = file.readlines()[4:]

        wavelengths = []
        responses = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                wavelength, response = map(float, parts)
                wavelengths.append(1 / wavelength * 10000)
                responses.append(response)

        plt.plot(wavelengths, responses, color=modis_color, label=modis_legend_label)


# 添加标签、标题和图例等
plt.xlabel('Wavelength/μm', fontname='Times New Roman', labelpad=9, fontsize=14)
plt.ylabel('Reflectance', fontname='Times New Roman', fontsize=14)
plt.title('Spectral Response Function', fontname='Times New Roman', fontsize=14)

# 创建FY-3D和MODIS的图例，并设置位置为右上角，水平布局
font = fm.FontProperties(family="Times New Roman", size=10)
fy3d_legend = plt.Line2D([0], [0], color=fy3d_color, linewidth=2, label=fy3d_legend_label)
modis_legend = plt.Line2D([0], [0], color=modis_color, linewidth=2, label=modis_legend_label)
first_legend=plt.legend(handles=[fy3d_legend], bbox_to_anchor=(0.38, -0.130),loc='lower center',frameon=False,prop=font)
ax=plt.gca().add_artist(first_legend)
plt.legend(handles=[modis_legend], bbox_to_anchor=(0.67, -0.130),loc='lower center',facecolor='none',frameon=False,prop=font)
plt.xticks(fontproperties = 'Times New Roman', size = 10)
plt.yticks(fontproperties = 'Times New Roman', size = 10)
plt.show()