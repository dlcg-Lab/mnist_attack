import sys
import torch
import numpy as np

sys.path.append("..")
from Dataset import get_dataset
from network import Network
from utils import sub_fig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_rows(arr, num_rows):
    return arr[np.random.choice(arr.shape[0], num_rows, replace=False)]


def perturb_dir(arr, x, y, config):
    val = arr[0, x, y]
    neighbor_size = config.attack.neighbor_size

    i_min = max(x - neighbor_size // 2, 0)
    i_max = min(x + neighbor_size // 2 + 1, arr.shape[1])
    j_min = max(y - neighbor_size // 2, 0)
    j_max = min(y + neighbor_size // 2 + 1, arr.shape[2])

    pad_width = ((0, 0), (neighbor_size // 2, neighbor_size // 2), (neighbor_size // 2, neighbor_size // 2))
    padded_arr = np.pad(arr, pad_width=pad_width, mode='edge')

    avg = np.mean(padded_arr[0, i_min:i_max + neighbor_size, j_min:j_max + neighbor_size])

    gradient = val - avg

    if gradient > 0:
        return "down"
    else:
        return "up"


def eval_accuracy(test_dataset, model):
    len_dataset = len(test_dataset)
    acc_c = 0
    with torch.no_grad():
        for x, y in test_dataset:
            y_gt = y
            y_out = model(x.to(device))
            pred = y_out.argmax(dim=1, keepdim=True)
            y_pred = pred.cpu().numpy()
            if y_pred[0, 0] == y_gt:
                acc_c += 1
    return acc_c / len_dataset


def attack_sps(arr, sps, epsilon, config):
    for x, y in sps:
        direction = perturb_dir(arr=arr, x=x, y=y, config=config)
        if direction == 'up':
            arr[0, x, y] += epsilon
        else:
            arr[0, x, y] -= epsilon
        arr[0, x, y] = np.clip(arr[0, x, y], 0, 1)
    return arr


def sp_attack(config=None, cnn_model=None):
    img_num = 0
    train_dataset, test_dataset = get_dataset(config=config)

    if cnn_model == None:
        cnn_model = Network(config=config).to(device)
        cnn_model.load_state_dict(torch.load(config.MODEL.weight_path).format(config.DATA.name))
    cnn_model.eval()
    sp = np.loadtxt(config.IosForest.sensitive_points_save_path + '_' + config.IosForest.processing + '_{}'.format(
        config.DATA.name) + '.csv', delimiter=',', dtype=int)

    epsilon = 1 * config.attack.beta

    acc_c_ori = 0
    acc_c_adv = 0
    with torch.no_grad():
        for index, args in enumerate(test_dataset):
            x, y = args
            y_gt = y
            y_pred_0 = None
            if config.auto.auto == False:
                y_out_0 = cnn_model(x.to(device))
                pred_0 = y_out_0.argmax(dim=1, keepdim=True)
                y_pred_0 = pred_0.cpu().numpy()

            x = x.numpy()
            sps = random_rows(sp, config.attack.pixel_num)
            x_adv = attack_sps(x.copy(), sps, epsilon, config)
            x_adv = torch.as_tensor(x_adv)

            y_out_1 = cnn_model(x_adv.to(device))
            x_adv = x_adv.cpu().numpy()
            pred_1 = y_out_1.argmax(dim=1, keepdim=True)
            y_pred_1 = pred_1.cpu().numpy()

            if img_num < config.attack.save_fig_max_num and y_pred_1[0, 0] != y_gt:
                sub_fig(image1=x, image2=x_adv, label_1=y_pred_0, label_2=y_pred_1, label_gt=y_gt, index=index,
                        config=config)
                img_num += 1
            if config.auto.auto == False:
                if y_pred_0[0, 0] == y_gt:
                    acc_c_ori += 1
            if y_pred_1[0, 0] == y_gt:
                acc_c_adv += 1

    acc_ori = acc_c_ori / len(test_dataset)
    acc_adv = acc_c_adv / len(test_dataset)
    if config.auto.auto == False:
        print('Test accuracy on clean examples: %0.2f%%\n' % (acc_ori * 100))
        print('Test accuracy on adversarial examples: %0.2f%%\n' % (acc_adv * 100))
    return acc_ori, acc_adv
