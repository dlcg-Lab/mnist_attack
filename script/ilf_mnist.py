import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import tsfel
import pandas as pd

cgf_file_1 = tsfel.get_features_by_domain("statistical")
cgf_file_2 = tsfel.get_features_by_domain("temporal")
cgf_file_3 = tsfel.get_features_by_domain("spectral")


def IsoForestSpotter(config, auto=False):
    folder_path = config.IosForest.dataset_path.format(config.DATA.name)

    file_names = os.listdir(folder_path)
    images = []

    for file_name in file_names:
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            image_array = np.array(image)
            images.append(image_array)

    images = np.array(images)
    images = np.transpose(images, (1, 2, 0))  # 将第一个维度移动到最后一个维度
    # 加载数据并将其转换为二维数组
    data = images.reshape(-1, 10000)

    if config.IosForest.processing == 'ori':
        X_reduced = data
    elif config.IosForest.processing == 'PCA':
        pca = PCA(n_components=50)
        X_reduced = pca.fit_transform(data)
    elif config.IosForest.processing == 'tsfel':
        data_ = []
        for per_data in data:
            per_data = np.squeeze(per_data)
            df = pd.DataFrame(per_data)
            X = tsfel.time_series_features_extractor(cgf_file_1, df, verbose=0)
            X1 = np.squeeze(X)
            X = tsfel.time_series_features_extractor(cgf_file_2, df, verbose=0)
            X2 = np.squeeze(X)
            X = tsfel.time_series_features_extractor(cgf_file_3, df, verbose=0)
            X3 = np.squeeze(X)
            X = np.concatenate([X1, X2, X3], axis=0)
            data_.append(X)
        X_reduced = np.array(data_)
    else:
        X_reduced = data

    # 使用IsolationForest模型进行拟合
    clf = IsolationForest(n_estimators=config.IosForest.n_estimators, max_samples='auto',
                          contamination=config.IosForest.contamination,
                          max_features=1.0, bootstrap=False, n_jobs=-1, random_state=None, verbose=0)
    clf.fit(X_reduced)

    # 找到所有异常序列的坐标
    outliers = np.where(clf.predict(X_reduced) == -1)

    if auto == False:
        print('{} sensitive points have been generated'.format(len(outliers[0])))

    results = []
    # 打印所有异常序列的坐标
    for idx in outliers[0]:
        x = idx // config.DATA.shape_in_w
        y = idx % config.DATA.shape_in_w
        # print("({}, {})".format(x, y))
        results.append([x, y])
    results = np.array(results).astype('int8')
    np.savetxt(config.IosForest.sensitive_points_save_path + '_' + config.IosForest.processing + '_{}'.format(
        config.DATA.name) + '.csv', results,
               delimiter=',', fmt='%d')
