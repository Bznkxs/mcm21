import torch
import matplotlib.pyplot as pyplot
plt = pyplot
import numpy as np
from tqdm import tqdm
from array2gif import write_gif
from scipy.signal import convolve2d
from mpl_toolkits.mplot3d import Axes3D
# conv

def lifegame(a, t):
    w = torch.tensor([[[[1,1,1],[1,0,1],[1,1,1]]]])
    def trans(a):
        return (a[0] * 255).repeat(3,1,1).numpy()
    outs = [trans(a)]
    for i in tqdm(range(t)):
        sums = torch.conv2d(a, w, stride=1, padding=1)
        x = (sums == 2) * a
        y = torch.ones(a.shape, dtype=torch.int64) * (sums == 3)
        a = x | y
        outs.append(trans(a))
    return outs

def draw():
    k = 500
    t = 1600
    ski = torch.zeros(1, 1, k, k, dtype=torch.int64)
    ski_back = torch.tensor([[0,1,0],[0,0,1],[1,1,1]])
    ski[0][0][:3, :3][ski_back > 0] = 1
    outs = lifegame(ski, t)
    write_gif(outs, 'output.gif')

# draw()


def lifegame1(a, t):
    w = np.array([[1,1,1],[1,0,1],[1,1,1]])
    reshape_ = a.shape[0]
    def trans(a):
        g = a * 255
        g = g.repeat(3)
        return g.reshape(3,reshape_,reshape_)
    outs = [trans(a)]
    for i in tqdm(range(t)):
        sums = convolve2d(a, w, mode='same', fillvalue=0)
        x = (sums == 2) * a
        y = np.ones(a.shape, dtype=int) * (sums == 3)
        a = x | y
        outs.append(trans(a))
    return outs

def draw1():
    k = 500
    t = 1600
    ski = np.zeros((k, k), dtype=int)
    ski_back = np.array([[0,1,0],[0,0,1],[1,1,1]])
    ski[:3, :3][ski_back > 0] = 1
    outs = lifegame1(ski, t)
    write_gif(outs, 'output1.gif')

draw1()