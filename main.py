import sys
import copy
import torch
from config import get_config
from utils import _grid
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch import optim
from Dataset import get_dataset_dl
from network import Network
from torchsummary import summary
from script.ilf_mnist import IsoForestSpotter
from script.SparseAttack import sp_attack
from utils import fig_hist
from utils import CPv
from OPA import OPA
import warnings
import platform

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = get_config('configs/configFashionMNIST.yaml')

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('Agg')

# 定义模型参数
params_model = {
    "shape_in": (config.DATA.shape_in_c, config.DATA.shape_in_h, config.DATA.shape_in_w),
    "initial_filters": config.MODEL.initial_filters,
    "num_fc1": config.MODEL.num_fc,
    "dropout_rate": config.MODEL.dropout_rate,
    "num_classes": config.TRAIN.epoch_num,
    "weight_path": config.MODEL.weight_path
}

params_train = {
    "train": None,
    "val": None,
    "epochs": 100,
    "optimiser": None,
    "lr_change": None,
    "f_loss": nn.NLLLoss(reduction="sum"),
    "weight_path": config.MODEL.weight_path,
}


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def loss_epoch(model, loss_func, dataset_dl, opt=None):
    run_loss = 0.0
    t_metric = 0.0
    len_data = len(dataset_dl.dataset)
    # internal loop over dataset
    for xb, yb in dataset_dl:
        # move batch to device
        xb = torch.as_tensor(xb)
        yb = torch.as_tensor(yb)
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)  # get model output
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)  # get loss per batch
        run_loss += loss_b  # update running loss

        if metric_b is not None:  # update running metric
            t_metric += metric_b

    loss = run_loss / float(len_data)  # average loss value
    metric = t_metric / float(len_data)  # average metric value

    return loss, metric


def loss_batch(loss_func, output, target, opt=None):
    # 计算损失
    loss = loss_func(output, target)

    # 获取预测的类别
    pred = output.argmax(dim=1, keepdim=True)

    # 计算性能指标
    metric_b = pred.eq(target.view_as(pred)).sum().item()

    # 如果提供了优化器，则进行梯度下降
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


def train_val(model, params, verbose=False):
    # Get the parameters
    epochs = params["epochs"]
    loss_func = params["f_loss"]
    opt = params["optimiser"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]

    # loss_history和metric_history用于绘图
    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    ''' Train Model n_epochs '''

    for epoch in tqdm(range(epochs)):

        ''' Get the Learning Rate '''
        current_lr = get_lr(opt)
        if verbose:
            print('Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))

        '''

        Train Model Process

        '''

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)

        # collect losses
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        '''

        Evaluate Model Process

        '''

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)

        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            # store weights into a local file
            torch.save(model.state_dict(), weight_path.format(config.DATA.name))
            if verbose:
                print("Copied best model weights!")

        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        # learning rate schedule
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            if verbose:
                print("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        if verbose:
            print(f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100 * val_metric:.2f}")
            print("-" * 10)

            # load best model weights
    model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history


def inference(model, dataset_dl, device, config):
    len_data = float(len(dataset_dl) * config.TEST.batch_size)
    model = model.to(device)  # move model to device
    metric_c = 0
    model.eval()
    with torch.no_grad():
        with tqdm(total=100) as pbar:
            for xb, yb in dataset_dl:
                xb = torch.as_tensor(xb)
                yb = torch.as_tensor(yb)
                xb = xb.to(device)
                yb = yb.to(device)
                output = model(xb)
                pred = output.argmax(dim=1, keepdim=True)
                metric_b = pred.eq(yb.view_as(pred)).sum().item()
                metric_c += metric_b
                pbar.update(100 / len(dataset_dl))

    return float(metric_c / len_data) * 100


def main_training():
    train_dl, val_dl = get_dataset_dl(config=config)

    cnn_model = Network(config).to(device=device)
    if config.TRAIN.OPTIMIZER.NAME == 'Adam':
        opt = optim.Adam(cnn_model.parameters(), lr=config.TRAIN.lr)
    else:
        opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
    params_train['train'] = train_dl
    params_train['val'] = val_dl
    params_train['optimiser'] = opt
    params_train['lr_change'] = ReduceLROnPlateau(opt,
                                                  mode='min',
                                                  factor=0.5,
                                                  patience=20,
                                                  verbose=False)
    summary(cnn_model, input_size=params_model['shape_in'], device=device.type)
    cnn_model, loss_hist, metric_hist = train_val(cnn_model, params_train)
    fig_hist(params=params_train, loss_hist=loss_hist, metric_hist=metric_hist, config=config)
    main_testing()


def main_testing():
    train_set, val_set = get_dataset_dl(config=config)
    print(len(val_set), 'samples found')
    cnn_model = Network(config).to(device=device)
    cnn_model.load_state_dict(torch.load(params_model["weight_path"].format(config.DATA.name)))
    acc = inference(model=cnn_model, dataset_dl=val_set, device=device, config=config)
    print('Test accuracy on test set: %0.2f%%\n' % acc)
    # print('Test accuracy on adversarial examples: %0.2f%%\n' % acc)


def autoHP(config=None):
    config.auto.auto = True
    model = Network(config=config).to(device)
    model.load_state_dict(torch.load(config.MODEL.weight_path.format(config.DATA.name)))
    # print('runing : ')
    # print(' >> PCA -> iForese -> F-Principle')
    # print(' >> tsfel -> iForese -> F-Principle')
    # with tqdm(total=100) as pbar:
    #     for feature in config.auto.feature_list:
    #         config.IosForest.processing = feature
    #         IsoForestSpotter(config=config, auto=True)
    #         res = np.zeros((len(config.auto.beta_list), len(config.auto.pixel_num_list)))
    #         for i, beta in enumerate(config.auto.beta_list):
    #             config.attack.beta = beta
    #             for j, pixel_num in enumerate(config.auto.pixel_num_list):
    #                 config.attack.pixel_num = pixel_num
    #                 acc_ori, acc_adv = sp_attack(config=config, cnn_model=model)
    #                 pbar.update(100 / (len(config.auto.feature_list) * len(config.auto.pixel_num_list) * len(
    #                     config.auto.beta_list)))
    #                 res[i, j] = acc_adv
    #         res = res[::-1]
    #         np.savetxt(
    #             config.auto.csv_path.format(config.DATA.name) + 'iForest_' + config.IosForest.processing + '_{}'.format(
    #                 config.DATA.name) + '.csv', res, delimiter=',',
    #             fmt='%.2f')
    #         _grid(config, auto=True, DE=False)

    print('runing : ')
    print(' >> ORI -> DE')
    train_set, test_loader = get_dataset_dl(config=config)
    res = np.zeros((len(config.auto.beta_list), len(config.auto.pixel_num_list)))
    with tqdm(total=100) as pbar:
        for i, beta in enumerate(config.auto.beta_list):
            config.attack.beta = beta
            for j, pixel_num in enumerate(config.auto.pixel_num_list):
                config.attack.pixel_num = pixel_num
                acc_adv = OPA(config=config, model=model, test_loader=test_loader)
                pbar.update(100 / (len(config.auto.beta_list) * len(config.auto.pixel_num_list)))
                res[i, j] = acc_adv
    res = res[::-1]
    np.savetxt(config.auto.csv_path.format(config.DATA.name) + 'DE_' + config.IosForest.processing + '_{}'.format(
        config.DATA.name) + '.csv',
               res, delimiter=',',
               fmt='%.2f')
    _grid(config, auto=True, DE=True)

    # CPv(config=config)


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        main_training()
    elif sys.argv[1] == 'test':
        main_testing()
    elif sys.argv[1] == 'sp':
        IsoForestSpotter(config=config)
    elif sys.argv[1] == 'attack':
        sp_attack(config=config)
    elif sys.argv[1] == 'DE':
        OPA(config=config)
    elif sys.argv[1] == 'auto':
        autoHP(config=config)
    elif sys.argv[1] == 'heatmap':
        _grid(config=config)
    elif sys.argv[1] == 'CPv':
        CPv(config=config)
