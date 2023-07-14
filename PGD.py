from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
import numpy as np
from tqdm import tqdm
import torch
from network import Network
from Dataset import get_dataset_dl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def PGD(config=None, AT_flag=False):
    train_set, test_loader = get_dataset_dl(config=config)
    cnn_model = Network(config=config).to(device)
    if AT_flag:
        cnn_model.load_state_dict(torch.load(config.MODEL.weight_path.format(config.DATA.name + '_AT')))
    else:
        cnn_model.load_state_dict(torch.load(config.MODEL.weight_path.format(config.DATA.name)))
    cnn_model.eval()

    len_data = float(len(test_loader) * config.TEST.batch_size)
    metric_c = 0
    cnn_model.eval()
    # with torch.no_grad():
    with tqdm(total=100) as pbar:
        for xb, yb in test_loader:
            xb = torch.as_tensor(xb)
            yb = torch.as_tensor(yb)
            xb = xb.to(device)
            yb = yb.to(device)
            xb = projected_gradient_descent(cnn_model, xb, 1 * config.attack.beta, 0.01, 40, np.inf)
            output = cnn_model(xb)  # get model output
            pred = output.argmax(dim=1, keepdim=True)
            metric_b = pred.eq(yb.view_as(pred)).sum().item()
            metric_c += metric_b
            pbar.update(100 / len(test_loader))
    res = float(metric_c / len_data) * 100
    print(res)
    return res
