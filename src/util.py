import os
import numpy as np
from matplotlib import pyplot as plt
import torch, argparse, imageio

def save2img_rgb(img_data, img_fn):
    plt.figure(figsize=(img_data.shape[1]/10., img_data.shape[0]/10.))
    plt.axes([0, 0, 1, 1])
    plt.imshow(img_data, )
    plt.savefig(img_fn, facecolor='black', edgecolor='black', dpi=10)
    plt.close()

def save2img(d_img, fn):
    if fn[-4:] == 'tiff': 
        img_norm = d_img.copy()
    else:
        _min, _max = d_img.min(), d_img.max()
        if _max == _min:
            img_norm = d_img - _max
        else:
            img_norm = (d_img - _min) * 255. / (_max - _min)
        img_norm = img_norm.astype('uint8')
    imageio.imwrite(fn, img_norm)

def scale2uint8(_img):
    _min, _max = _img.min(), _img.max()
    if _max == _min:
        _img_s = _img - _max
    else:
        _img_s = (_img - _min) * 255. / (_max - _min)
    _img_s = _img_s.astype('uint8')
    return _img

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def cosine_decay(epoch, warmup=100, max_epoch=10000):
    if epoch <= warmup:
        return (epoch / warmup)
    else:
        return 0.5 * (1 + np.cos((epoch - warmup)/max_epoch * np.pi))

def str2list(s):
    return s.split(':')


"""
以下是用于CVAE的一些工具函数
"""
def array3Dto2Dmat(data):
    """
    Flatten 3D data to 2D for training.
    :param data: numpy array, shape (n_samples, height, width)
    :return: numpy array, shape (n_samples, height * width)
    """
    return data.reshape(data.shape[0], -1)


def mat2Dto3D(data, lat=224, lon=224):
    """
    Reshape 2D data back to 3D for visualization.
    :param data: numpy array, shape (n_samples, lat * lon)
    :param lat: int, latitude resolution
    :param lon: int, longitude resolution
    :return: numpy array, shape (n_samples, lat, lon)
    """
    return data.reshape(data.shape[0], lat, lon)

def predict_cVAE_fullSet(X, model, modelName):

    '''
    Compute the prediction given a X
    '''

    # Load model state dict
    model.load_state_dict(torch.load(os.path.expanduser('./models/' +
                                                        modelName + '.pt')))
    model = model.cuda()

    # Subset Y
    X = torch.from_numpy(X).float()

    yPred = model.predictX(X.cuda())

    yPred = yPred.detach().cpu().numpy()

    return yPred


def computeRData(x_test, y_test, model, modelName):
    """
    Compute predictions and save them as .npy files
    """
    nGenerations = 8
    for i in range(1, nGenerations + 1):
        # 模型预测
        y_pred = predict_cVAE_fullSet(X=x_test, model=model, modelName=modelName)

        # 将 2D 矩阵还原为 3D
        y_pred_3D = mat2Dto3D(y_pred, lat=224, lon=224)

        # 保存为 .npy 文件
        np.save(f"yPred_{i}.npy", y_pred_3D)

def predict_cVAE(X, index, model, modelName):

    '''
    Compute the prediction given some X (return random conditioned samples)
    '''

    # Load model state dict
    model.load_state_dict(torch.load(os.path.expanduser('./models/' + modelName + '.pt')))
    model = model.cuda()

    # Subset Y
    X = torch.from_numpy(X).float()
    X = X[index, :].unsqueeze(0)

    numSamples = 9

    yPred = model.predictX(X.cuda())

    for i in range(numSamples-1):
        yPrime =  model.predictX(X.cuda())
        yPred = torch.cat((yPred, yPrime), 0)

    yPred = yPred.detach().cpu().numpy()

    return yPred

def plotPrediction_cVAE(X, Y, index, model, modelName):

    '''
    Plot generated prediction (3x3) for a specific X
    '''

    yGen = predict_cVAE(X, index, model, modelName)
    yGen = mat2Dto3D(yGen)

    # Y = mat2Dto3D(Y)

    # Plot
    fig, ax = plt.subplots(nrows = 4, ncols = 3, figsize = (100, 100))
    fig.delaxes(ax[0, 0])
    fig.delaxes(ax[0, 2])

    minValue = np.nanmin(Y[index, :, :])
    maxValue = np.nanmax(Y[index, :, :])

    if (index == 1819):
        fig.suptitle('Prediction for 25-12-2007', fontsize = 150)
    elif (index == 1936):
        fig.suptitle('Prediction for 20-04-2008', fontsize = 150)
    elif (index == 2023):
        fig.suptitle('Prediction for 16-07-2008', fontsize = 150)

    ax[0, 1].imshow(np.flip(np.flip(Y[index, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[0, 1].set_title('Y test', fontsize = 90)

    ax[1, 0].imshow(np.flip(np.flip(yGen[0, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[1, 1].imshow(np.flip(np.flip(yGen[1, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[1, 2].imshow(np.flip(np.flip(yGen[2, :, :]), 1),
                    vmin = minValue, vmax = maxValue)

    ax[2, 0].imshow(np.flip(np.flip(yGen[3, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[2, 1].imshow(np.flip(np.flip(yGen[4, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[2, 2].imshow(np.flip(np.flip(yGen[5, :, :]), 1),
                    vmin = minValue, vmax = maxValue)

    ax[3, 0].imshow(np.flip(np.flip(yGen[6, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[3, 1].imshow(np.flip(np.flip(yGen[7, :, :]), 1),
                    vmin = minValue, vmax = maxValue)
    ax[3, 2].imshow(np.flip(np.flip(yGen[8, :, :]), 1),
                    vmin = minValue, vmax = maxValue)

    plt.savefig('./figures/prediction_' + str(index) + '.png')
    plt.close()

