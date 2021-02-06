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

# devices
device = 'cpu'
dfloat = torch.float32 # torch.float16
dint = torch.int8
if torch.has_cuda:
    device = 'cuda'
    dfloat = torch.float16 # torch.float16
    dint = torch.int8

# debugging
er = False
logs = []
def debug(*args, **kwargs):
    if er:
        print(*args, **kwargs)
    else:
        logs.append((args, kwargs))
        if len(logs) > 20:
            logs.pop(0)

# utils
def make_circular(mat, h=None, d=None):
    if h is None:
        h = mat.shape[-2]
        d = mat.shape[-1]
    elif d is None:
        d = h

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


# env consts
d_birth = 9  # must be odd
d_death = 9

variance = 1
n_fungi = 3

_birth_rates = torch.tensor([[
    1.2, 2.5, 1.5
][:n_fungi]]).reshape((1, n_fungi, 1, 1))  # 1xnx1x1
capacities = torch.tensor([[
    0.741, 0.541, 0.6
][:n_fungi]]).reshape((1, n_fungi, 1, 1))  # 1xnx1x1

# exp. settings
tr = 10  # timestep per sec
k = 100  # width = height
tot_time = 100  # total time span in sec
init_p = 0.01  # initial density

# calculated env. consts
w_b = torch.ones(n_fungi, n_fungi, d_birth, d_birth, dtype=dfloat, device=device)
for i in range(n_fungi):
    make_circular(w_b[i, i])

w_d = torch.ones(1, n_fungi, d_death, d_death, dtype=dfloat, device=device)
for i in range(n_fungi):
    make_circular(w_b[0, i])

death_range = w_d[0, 0].sum()
birth_range = w_b[0, 0].sum()
K = death_range * capacities
birth_rate = _birth_rates / tr  # (fungi per sec) / (timestep per sec) = (fungi per timestep)
r = birth_rate

# calculated exp. settings
t = tot_time * tr  # in timesteps

# colors
colors = np.array([
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 1]
][:n_fungi]).reshape((n_fungi, 1, 1, 3))

# simulation
def fungisim(a, age, t):


    def trans(a):
        out = a[0].cpu().numpy().reshape((n_fungi, k, k, 1))
        #debug("out", a[0])
        pic = np.zeros((k, k, 3))
        for j in range(n_fungi):
            pic += colors[j] * out[j]
        #print("pic",pic)
        pyplot.imsave(f'outputs/{i+1}.png', pic)
        return [np.sum(out[j]) for j in range(n_fungi)]  # (a[0] * 255).repeat(3, 1, 1).cpu().numpy()
    i = -1
    outs = [trans(a)]
    for i in tqdm(range(t)):
        debug(i,":")
        debug("a\n", a)

        sums = torch.conv2d(a, w_d, padding=d_death//2)  # (1x1xkxk)
        debug("sums\n", sums)
        sums[sums < 0.1] = 1
        death_rates = (sums > K) * (1 - K / sums)  # float, (1xnxkxk)
        debug("death\n", death_rates)
        death_rates[death_rates > 1] = 1  # death rate exceed
        rr = torch.rand((1, n_fungi, k, k), device=device)
        debug("rr, k\n", rr, k)
        debug("sample_1\n", death_rates + rr)
        sample = (death_rates + rr).to(dint)
        sample[a < 0.1] = 0
        debug("sample\n", sample)
        a = a - sample
        debug("after a\n", a)
        if torch.max(a.sum(dim=1)) > 1 or torch.min(a) < 0:
            for args, kwargs in logs:
                print(*args, **kwargs)
            exit(-1)
        #age[a < 0.1] = 0


        #  birth stage
        sums_ = torch.conv2d(a, w_b, padding=d_birth//2)
        debug('sums\n', sums_)
        birth_rates = sums_ * birth_rate / birth_range  # float
        debug("births\n", birth_rates)
        # normalize
        normfacts = birth_rates.sum(1)
        normfacts[normfacts < 1] = 1
        birth_rates /= normfacts
        # roulette
        rr = torch.rand(k, k, device=device, dtype=dfloat)
        debug("rr\n", rr)
        selected = torch.zeros(k, k, device=device, dtype=torch.bool)
        sample = torch.zeros(1, n_fungi, k, k, device=device, dtype=torch.bool)
        selected[birth_rates[0].sum(0) == 0] = True
        selected[a[0].sum(0) > 0.1] = True
        for j in range(n_fungi):
            debug("selected\n", selected)
            sample[0, j][(birth_rates[0, j] > rr) & (~selected)] = True

            selected |= sample[0, j]
            rr -= birth_rates[0, j]

        debug("births_after\n", birth_rates)


        a += sample
        if torch.max(a.sum(dim=1)) > 1 or torch.min(a) < 0:
            for args, kwargs in logs:
                print(*args, **kwargs)
            exit(-2)
        debug("at last\n", a)
        outs.append(trans(a))

    return outs

def draw():
    ski = torch.zeros(1, n_fungi, k, k, dtype=dfloat, device=device)
    rand_map = torch.rand((1, n_fungi, k, k), device=device)
    max_indices = rand_map.view(n_fungi, -1).argmax(0)
    print(max_indices)
    rand_map = torch.rand((k, k), device=device)
    max_indices = torch.stack(
        [
            max_indices,
            torch.repeat_interleave(torch.arange(k), k),
            torch.arange(k).repeat(k)
        ]
    )
    print(max_indices)
    mask = torch.sparse_coo_tensor(
        max_indices,
        torch.ones(len(max_indices[0])),
        dtype=dint,
        device=device
    )
    mask = mask.to_dense().to(device)
    ski[0, :, rand_map <= init_p] = 1
    ski = ski * mask
    print(ski.sum())
    nums = fungisim(ski, None, t)  # num a 2-d list
    nums = np.array(nums).T  # a matrix

    fig = pyplot.figure()
    #pyplot.ion()

    print(nums)

    for _i, num in enumerate(nums):
        pyplot.plot(np.arange(0, t + 1) / tr, num)
    pyplot.show()

        #pyplot.pause(0.01)
    #pyplot.ioff()
    #pyplot.show()
    # write_gif(outs, 'output.gif')



# draw()
draw()