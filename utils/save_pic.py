"""
@Env: /anaconda3/python3.10
@Time: 2023/7/14-18:27
@Auth: karlieswift
@File: save_pic.py
@Desc: 
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
def tensor2pic(x,path,std,mean,dpi=300):
    """
    x:tensor.shape=(3,224,224)  x 为标准化的图片
    path: save path pics
    dpi:像素密度
    """
    x = x.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1)).numpy()
    im = np.transpose(x, (1, 2, 0))
    fig, ax = plt.subplots()
    ax.imshow(im, aspect='equal')
    plt.axis('off')
    height, width, channels = im.shape

    fig.set_size_inches(width / dpi, height / dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(path, dpi=dpi)