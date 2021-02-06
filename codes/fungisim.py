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
d_birth = 9  # must be odd
d_death = 9

variance = 1


def make_circular(mat):
    h, d = mat.shape[-2], mat.shape[-1]
    ox, oy = h // 2, d // 2
    a, b = ox, oy
    for i in range(h):
        for j in range(d):
            x = i - ox
            y = j - oy
            if x ** 2 / a ** 2 + y ** 2 / b ** 2 <= 1:
                #print(i, j)
                mat[i, j] = 1
            else:
                mat[i, j] = 0
    #print("over",mat)



w_b = torch.ones(d_birth, d_birth, dtype=dfloat, device=device)
make_circular(w_b)
#print(w_b)
w_b = w_b.reshape(1, 1, d_birth, d_birth)

w_d = torch.ones(d_death, d_death, dtype=dfloat, device=device)
make_circular(w_d)
w_d = w_d.reshape(1, 1, d_death, d_death)

print(w_b)
print(w_d)

tr = 20  # timestep per sec
expected_expectancy = 1000 * tr  # in timesteps

# death rate
b = 1. / expected_expectancy
death_rate = b

birth_rate = 1.2 / tr  # (fungi per sec) / (timestep per sec) = (fungi per timestep)
r = birth_rate - death_rate

K = w_d[0, 0].sum() * 0.741

k = 200
t = 80 * tr  # in timesteps

init_p = 0.003

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



        sums = torch.conv2d(a, w_d, stride=1, padding=d_death//2)

        debug("sums\n", sums)
        # sums[sums == 0] = 0.1
        death_rates = (sums > K) * (1 - K / sums)  # float
        debug("death\n", death_rates)
        death_rates[death_rates > 1.] = 1.  # death rate exceed
        rr = torch.rand((1, 1, k, k), device=device)
        debug("rr, k\n", rr, k)
        debug("sample_1\n", death_rates + rr)
        sample = (death_rates + rr).to(dint)
        debug("sample\n", sample)
        a = a - (a * sample)
        debug("after a\n", a)
        outs.append(trans(a))
        #age[a < 0.1] = 0

                #  birth stage
        sums_ = torch.conv2d(a, w_b, padding=d_birth//2)
        birth_rates = sums_ * birth_rate / d_birth / d_birth  # float
        debug("births\n", birth_rates)
        birth_rates[birth_rates > 1.] = 1.  # birth rate exceed
        debug("births_after\n", birth_rates)
        rr = torch.rand((1, 1, k, k), device=device)
        debug("rr\n", rr)
        debug("sample1\n", birth_rates + rr)
        sample = (birth_rates + rr).to(dint)
        sample = ((1-a) * sample)
        #age[sample > 0] = (torch.randn(age[sample > 0].shape, device=device) * expected_expectancy)
        a += sample
        # outs.append(trans(a))
        #debug("ages\n", age)
        # dying stage
        #age -= 1
        #a[age <= 0] = 0

        debug("at last\n", a)
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
    print(nums)


# draw()
draw()