# An unsupervised adversarial attack method based on data features


## Abstract

Thanks to the emergence of large-scale datasets and the improvement of hardware computing capabilities, deep learning technology has been widely applied in various fields. However, as deep learning applications become more popular, the insecurity of deep learning models gradually becomes apparent, making adversarial samples a highly concerned area. The structure of the dataset itself is the basis for any deep model to learn discriminative features and is also the main basis for constructing adversarial samples. Inspired by this, we attempt to extract position-related anomalies from internal features of datasets and construct adversarial samples based on frequency laws. Our method limits the number and amplitude of attack perturbations by controlling L0 and Linf perturbations to achieve sparse attacks. Our method uses feature extraction and principal component analysis methods to reduce dimensions for each coordinate in a dataset, using unsupervised algorithms to search for sensitive point coordinates in a dataset. Then we generate adversarial samples by destroying high-frequency information at sensitive points based on frequency theory. We conducted extensive experiments on image and time-series datasets, which demonstrate that our approach is effective.

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

##  Author

[Yuyao Ge](https://github.com/GeYuYao-hub)

Yuyao Ge is currently studying computer science and technology as an undergraduate at North China University of Technology, with a research focus on adversarial attacks and deep learning.

