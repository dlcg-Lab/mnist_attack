import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import os

config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
from matplotlib import rcParams

rcParams.update(config)
sns.set(style='whitegrid', font='Serif')


def fig_hist(params, loss_hist, metric_hist, config=None):
    """
    绘制训练过程中loss值和准确率的变化
    """
    epochs = params["epochs"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.lineplot(x=[*range(1, epochs + 1)], y=loss_hist["train"], ax=ax[0], label='loss_hist["train"]')
    sns.lineplot(x=[*range(1, epochs + 1)], y=loss_hist["val"], ax=ax[0], label='loss_hist["val"]')
    sns.lineplot(x=[*range(1, epochs + 1)], y=metric_hist["train"], ax=ax[1], label='metric_hist["train"]')
    sns.lineplot(x=[*range(1, epochs + 1)], y=metric_hist["val"], ax=ax[1], label='metric_hist["val"]')
    plt.title('Convergence History')

    plt.savefig('figs/log_{}.png'.format(config.DATA.name), dpi=200)


def sub_fig(image1, image2, label_1, label_2, label_gt, index=0, config=None):
    image1 = np.squeeze(image1)
    image2 = np.squeeze(image2)
    # 创建一个有两个子图的窗口
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    ax1.grid(False)  # 关闭网格线显示
    ax2.grid(False)  # 关闭网格线显示
    ax1.axis('off')  # 隐藏所有坐标轴
    ax2.axis('off')  # 隐藏所有坐标轴

    # 在子图1中显示第一张图片
    ax1.imshow(image1)
    ax1.set_title('Clean Examples Predict Label:{}'.format(label_1[0, 0]))

    # 在子图2中显示第二张图片
    ax2.imshow(image2)
    ax2.set_title('Adversarial Example Predict Label:{}'.format(label_2[0, 0]))

    # 设置窗口的主标题
    fig.suptitle('Original Example & Adversarial Example \n Ground Truth : {} \n'.format(label_gt))
    create_folders(config.attack.save_fig_path.format(config.DATA.name))
    # 显示窗口
    plt.savefig(config.attack.save_fig_path.format(config.DATA.name) + str(index) + '.png')


def create_folders(path):
    try:
        os.makedirs(path)
        print(f"Created folder: {path}")
    except FileExistsError:
        pass
    except OSError as e:
        print(f"Failed to create folder: {path} ({e})")
    else:
        # 继续递归创建子文件夹
        for root, dirs, files in os.walk(path):
            for dir_name in dirs:
                create_folders(os.path.join(root, dir_name))


def _grid(config=None, auto=False, DE=False):
    if DE:
        res = np.loadtxt(
            config.auto.csv_path.format(config.DATA.name) + 'DE_' + config.IosForest.processing + '_{}'.format(
                config.DATA.name) + '.csv',
            delimiter=',')
    else:
        res = np.loadtxt(
            config.auto.csv_path.format(config.DATA.name) + 'iForest_' + config.IosForest.processing + '_{}'.format(
                config.DATA.name) + '.csv', delimiter=',')
    ax = sns.heatmap(res, cmap="RdBu_r", linewidths=0.3, annot=True, cbar=False, vmin=0, vmax=1)

    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.ax.set_ylabel('Colorbar Label', rotation=-90, va="bottom")

    ax.set_xticklabels([str(x) for x in config.auto.pixel_num_list])

    ylabel = [str(y) for y in config.auto.beta_list]
    ylabel.reverse()
    ax.set_yticklabels(ylabel)

    plt.xlabel("Number of perturbed pixels")
    plt.ylabel("\u03B2")

    if DE:
        plt.savefig('./figs/auto_{}_{}'.format('DE', config.DATA.name), dpi=200)
    else:
        plt.savefig('./figs/auto_{}_{}_{}'.format('iForest', config.IosForest.processing, config.DATA.name), dpi=200)
    if auto == False:
        plt.show()
    plt.close()


from scipy.ndimage.filters import gaussian_filter


def generate_heatmap(coords, img_size, sigma=5):
    """Generate a Gaussian heatmap centered on the given coordinates."""
    heatmap = np.zeros(img_size)
    for coord in coords:
        x, y = coord
        x = min(max(x, 0), img_size[1] - 1)
        y = min(max(y, 0), img_size[0] - 1)
        heatmap[int(y), int(x)] = 1
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    heatmap /= np.max(heatmap)
    return heatmap


def CPv(config=None):
    # 读取PNG图像并将其转换为NumPy数组
    image = np.array(Image.open("dataset/{}/raw/test/1.jpg".format(config.DATA.name)).convert('L'))

    t_list = config.auto.feature_list

    # 读取包含二维坐标的txt文件
    for t in t_list:
        with open(config.IosForest.sensitive_points_save_path + '_' + t + '.csv', "r") as f:
            coordinates = []
            for line in f:
                x, y = line.split(',')
                coordinates.append([int(x), int(y)])

        cam = generate_heatmap(coordinates, (config.DATA.shape_in_h, config.DATA.shape_in_w))

        fig, ax = plt.subplots()
        plt.axis('off')
        ax.imshow(image)
        ax.imshow(cam, alpha=0.5, cmap='jet')
        plt.savefig('./figs/CAM_{}_{}.png'.format(t, config.DATA.name), dpi=200)
        plt.close()

        # 可视化图像和坐标
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        plt.axis('off')
        ax.scatter([coord[1] for coord in coordinates], [coord[0] for coord in coordinates], s=20, c='red')
        plt.savefig('./figs/CPv_{}_{}.png'.format(t, config.DATA.name), dpi=200)
