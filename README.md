# 基于数据特征的无监督对抗攻击方法

## Abstract

得益于大规模数据集的涌现和硬件计算能力的提升，深度学习技术在社会各界得以广泛应用。然而，随着深度学习应用的普及，深度学习模型的不安全性逐渐显露出来，这使得对抗样本成为了一个备受关注的领域。数据集本身的结构是任何深度模型学习到有辨别力特征的基础，也是对抗样本构造的主要依据。受此启发，我们尝试从数据集内部特征出发，提取与位置相关的异常点，基于这些异常点应用频率定律实现对抗样本的构造。本文方法通过控制扰动的$L_0$和$L_{\infty}$实现对攻击扰动点数和幅度的限制以达到稀疏攻击。本文方法利用特征提取和主成分分析方法对数据集各坐标进行降维，使用无监督算法搜索数据集中敏感点的坐标。然后根据频率定理破坏敏感点的高频信息生成对抗样本。本文方法在图像和时序数据集上进行了大量实验，实验结果表明了本文方法的有效性。

## Approach

![](figs\lct.png)
 <div align="center"> <img src="output_img\684.png" width = 500 height = 300 /> </div>

## Data

The data used in this project comes from the [MNIST](http://yann.lecun.com/exdb/mnist/) and [FASHION-MNIST](https://github.com/zalandoresearch/fashion-mnist)

## Prerequisites

All python packages needed are listed in [pip-requirements.txt](D:/Project/mnist_attack/requirements.txt) file and can be installed simply using the pip command.

* torch : 2.0.0
* numpy : 1.24.2
* pandas : 1.5.3
* scipy : 1.9.1
* tqdm : 4.65.0
* tsfel : 0.1.5
* torchsummary : 1.5.1
* yacs : 0.1.8

Note that we copied some of the code from the [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch) and modified it to adapt it

## Code

To get dataset run [get_mnist.py](script/get_mnist.py) first.

```
python get_mnist.py
```

Select config files from [configs](configs) in [main.py](main.py)

```python
config = get_config('configs/configFashionMNIST.yaml')
```

Traing target model

```python
python main.py train
```

Test the metric of target model

```
python main.py test
```

Generate sensitive points

```
python main.py sp
```

Attack on sensitive points

```
python main.py attack
```

Attack on original dataset by differential evolution

```
python main.py DE
```

Automatically test the attack success rate of this method under various parameters, You can customize these parameters in  [config.py](config.py) or [configs](configs/)

```
python main.py auto
```

