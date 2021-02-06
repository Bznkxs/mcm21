import torch
import matplotlib.pyplot as pyplot
plt = pyplot
import numpy as np
from tqdm import tqdm
#from array2gif import write_gif
from scipy.signal import convolve2d
import scipy.sparse as sparse
from mpl_toolkits.mplot3d import Axes3D
# conv
device = 'cpu'
dfloat = torch.float32 # torch.float16
dint = torch.int8
if torch.has_cuda:
    device = 'cuda'
    dfloat = torch.float16 # torch.float16
    dint = torch.int8
d_birth = 3  # must be odd
d_death = 3

variance = 1



w_b = torch.ones(d_birth, d_birth, dtype=dfloat, device=device)
w_b[d_birth//2, d_birth//2] = 0
w_b = w_b.reshape(1, 1, d_birth, d_birth)

w_d = torch.ones(d_death, d_death, dtype=dfloat, device=device)
w_d[d_death//2, d_death//2] = 0
w_d = w_d.reshape(1, 1, d_death, d_death)


tr = 1  # timestep per sec
expected_expectancy = 10 * tr  # in timesteps

# death rate
b = 1. / expected_expectancy
death_rate = b

birth_rate = 1.2 / tr  # (fungi per sec) / (timestep per sec) = (fungi per timestep)
r = birth_rate - death_rate

K = d_death * d_death * 0.741

k = 100
t = 100 * tr  # in timesteps

init_p = 0.01

er = False
def debug(*args, **kwargs):
    if er:
        print(*args, **kwargs)


def fungisim(a, age, t):


    def trans(a):
        out = a[0].cpu().numpy()
        pyplot.imsave(f'outputs/{i+1}.png', out[0])
        return np.sum(out[0]) # (a[0] * 255).repeat(3, 1, 1).cpu().numpy()
    i = -1
    outs = [trans(a)]
    for i in tqdm(range(t)):
        debug(i,":")
        debug("a\n", a)
        debug("ages\n", age)
        # dying stage
        age -= 1
        a[age <= 0] = 0

        sums = torch.conv2d(a, w_d, padding=d_death//2)
        sums_ = torch.conv2d(a, w_b, padding=d_birth//2)
        debug("sums\n", sums)

        death_rates = r * sums / K  # float
        debug("death\n", death_rates)
        death_rates[death_rates > 1.] = 1.  # death rate exceed
        rr = torch.rand((1, 1, k, k), device=device)
        debug("rr, k\n", rr, k)
        debug("sample_1\n", death_rates + rr)
        sample = (death_rates + rr).to(dint)
        debug("sample\n", sample)
        a = a - (a * sample)
        debug("after a\n", a)
        age[a < 0.1] = 0
        #  birth stage

        birth_rates = sums_ * birth_rate / d_birth / d_birth  # float
        debug("births\n", birth_rates)
        birth_rates[birth_rates > 1.] = 1.  # birth rate exceed
        debug("births_after\n", birth_rates)
        rr = torch.rand((1, 1, k, k), device=device)
        debug("rr\n", rr)
        debug("sample1\n", birth_rates + rr)
        sample = (birth_rates + rr).to(dint)
        sample = ((1-a) * sample)
        age[sample > 0] = (torch.randn(age[sample > 0].shape, device=device) * expected_expectancy)
        a += sample
        debug("at last\n", a)
        outs.append(trans(a))
    return outs

def draw():

    ski = torch.zeros(1, 1, k, k, dtype=dfloat, device=device)
    rand_map = torch.zeros(1,1,k,k)#torch.rand((1, 1, k, k), device=device)
    rand_map[0,0,k//2,k//2] = 1
    ski[rand_map >= init_p] = 1

    age = ski * (torch.randn((1, 1, k, k), device=device) * variance + expected_expectancy)
    nums = fungisim(ski, age, t)
    fig = pyplot.figure()
    #pyplot.ion()

    pyplot.plot(nums)
    pyplot.show()

        #pyplot.pause(0.01)
    #pyplot.ioff()
    #pyplot.show()
    # write_gif(outs, 'output.gif')



# draw()
draw()