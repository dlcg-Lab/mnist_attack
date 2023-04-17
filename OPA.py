import torch
from network import Network
from Dataset import get_dataset_dl
from onepixel import OnePixel
# import torchattacks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def OPA(config=None, model=None, test_loader=None):
    if test_loader == None:
        train_set, test_loader = get_dataset_dl(config=config)

    if model == 'None':
        model = Network(config=config).to(device)
        model.load_state_dict(torch.load(config.MODEL.weight_path))
    model.eval()
    epsilon = 1 * config.attack.beta
    # 定义一像素攻击对象
    attack = OnePixel(model, pixels=config.attack.pixel_num, steps=config.DE.step,
                      popsize=config.DE.popsize, inf_batch=config.TEST.batch_size,epsilon= epsilon)

    len_data = float(len(test_loader) * config.TEST.batch_size)
    # 对MNIST数据集进行攻击并计算攻击成功率
    metric_c = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 对图像进行攻击
            adv_images = attack(images, labels)

            # 利用预训练模型对攻击后的图像进行分类
            output = model(adv_images).to(device)
            # print('ok')
            pred = output.argmax(dim=1, keepdim=True)
            metric_b = pred.eq(labels.view_as(pred)).sum().item()
            metric_c += metric_b
    res = float(metric_c / len_data) * 100

    return res
