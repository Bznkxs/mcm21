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



eps = 1e-5

# devices
device = 'cpu'
dfloat = torch.float32 # torch.float16
dint = torch.int8
if torch.has_cuda:
    device = 'cuda'
    dfloat = torch.float32 # torch.float16
    dint = torch.int8


# debugging
plotting_period = 1
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
d_death =7

variance = 1
n_fungi = 3

_birth_rates = torch.tensor([[
    2.22, 2.25, 2.15
][:n_fungi]], dtype=dfloat, device=device).reshape((1, n_fungi, 1, 1))  # 1xnx1x1
capacities = torch.tensor([[
    0.741, 0.541, 0.6
][:n_fungi]], dtype=dfloat, device=device).reshape((1, n_fungi, 1, 1))  # 1xnx1x1

# exp. settings
tr = 10  # timestep per sec
k = 2000  # width = height
tot_time = 250  # total time span in sec
init_p = 0.01  # initial density

# calculated env. consts
w_b = torch.zeros(n_fungi, n_fungi, d_birth, d_birth, dtype=dfloat, device=device)
for i in range(n_fungi):
    make_circular(w_b[i, i])

w_d = torch.zeros(n_fungi, n_fungi, d_death, d_death, dtype=dfloat, device=device)
for i in range(n_fungi):
    make_circular(w_d[i, i])



death_range = w_d[0, 0].sum()
birth_range = w_b[0, 0].sum()
K = death_range * capacities
birth_rate = _birth_rates / tr  # (fungi per sec) / (timestep per sec) = (fungi per timestep)
r = birth_rate

# calculated exp. settings
t = tot_time * tr  # in timesteps

# colors
colors = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
][:n_fungi]).reshape((n_fungi, 1, 1, 3))

# simulation
def fungisim_macro(a, age, t):


    def trans(a):
        out = a[0].cpu().numpy().reshape((n_fungi, k, k, 1))
        #debug("out", a[0])


        #print("pic",pic)
        if i % 5 == 0:
            pic = np.zeros((k, k, 3))
            for j in range(n_fungi):
                pic += colors[j] * out[j]
            # print("pic")
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
            # print(torch.max(a.sum(dim=1)), torch.min(a) < 0)
            import json
            fp=open('./o.txt', 'w')
            json.dump(a.tolist(), fp=fp)
            fp.close()
            for args, kwargs in logs:
                print(*args, **kwargs)
            a[a.sum(dim=1) > 1] = 0
            a[a < 0] = 0
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
            print(torch.max(a.sum(dim=1)), torch.min(a) < 0)

            import json
            fp=open('./o.txt', 'w')
            json.dump(a.tolist(), fp=fp)
            fp.close()
            for args, kwargs in logs:
                print(*args, **kwargs)
            a[a.sum(dim=1) > 1] = 0
            a[a < 0] = 0
            # exit(-2)
        debug("at last\n", a)
        outs.append(trans(a))

    return outs

# in micro we need:
# - material transfer matrix M_t: [ Me x Ms ] num of enzyme & num of substances
# - resource generation matrix M_g: [ n x Me ] num of fungi types & num of enzyme
# - resource reduction matrix M_r: [ n x Ms ] num of fungi types & num of substances (needed to survive)
# - toxicity matrix M_tx: [n x Ms] describes toxicity of each substance with regard to each type of fungi
#
# assumptions:
# - fungi secrete enzymes to their neighborhood
# - enzymes decomposite nutrition in a linear manner with regard to their concentration
# - fungi only absorb substances right below their grids
# model:
# - sum[l, i, j] calculates sum of fungi l in neighborhood of (i, j) [n x k x k]
# - calculates enzyme generation  enz = (sum.permute(1,2,0).matmul(M_g)) [ k x k x Me ]
#    - in-place generation
#    - convolve to get diffused generation? do not convolve in the first place
# - calculates resource generation (the most difficult step)
#     - materials = enz.matmul(M_t)  [ k x k x Ms ]
#     - an activation function
# - now we get resources that each grid can get
# - if not enough, die at a calculated probability
# - toxic: let it be a sigmoid function

Me = 3
Ms = 3
M_t = torch.tensor([
    [1, 0, 0],
    [0, 1, 0,],
    [0, 0, 1.,],
], device=device, dtype=dfloat)[:Me, :Ms]  # Me x Ms
M_g = torch.tensor([
    [2, 0, 0],
    [0, 3, 0],
    [0, 0, 3]
], device=device, dtype=dfloat)[:n_fungi, :Me] / tr



M_need = torch.tensor([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], device=device, dtype=dfloat)[:n_fungi, :Ms] / tr

M_toxic = torch.tensor([
    [0, 0, 1.0],
    [1.0, 0, 0],
    [0, 1.0, 0]
], device=device, dtype=dfloat)[:n_fungi, :Ms] / tr

M_toxic[M_toxic > 0] = 1/M_toxic[M_toxic > 0]
M_r = M_need - M_toxic

M_absorb = M_need + ((M_toxic > 0) & (M_need == 0))


w_enzgen = torch.zeros(n_fungi, n_fungi, d_death, d_death, dtype=dfloat, device=device)
for i in range(n_fungi):
    make_circular(w_enzgen[i, i])
w_enzgen /= death_range

w_absorb = torch.zeros(Ms, n_fungi, d_death, d_death, dtype=dfloat, device=device)
for j in range(Ms):
    for i in range(n_fungi):
        make_circular(w_absorb[j, i])
        w_absorb[j, i] *= M_absorb[i, j]

act_weight = torch.tensor([0.6, 1, 1],
    device=device, dtype=dfloat)[:Me] / tr # Me


def s_curve(x):  # -1 <------> 1
    return (torch.sigmoid(x * 2) * 2 - 1)
def act(x):
    # k x k x Me

    return (s_curve(x / act_weight) * (act_weight))
    # exit(1)
def fungisim_micro(a, age, t):
    def trans(a):
        out = a[0].cpu().numpy().reshape((n_fungi, k, k, 1))
        #debug("out", a[0])
        if i % plotting_period == 0:
            pic = np.zeros((k, k, 3))
            for j in range(n_fungi):
                pic += colors[j] * out[j]
            #print("pic")
            pyplot.imsave(f'outputs/{i+1}.png', pic)
        return [np.sum(out[j]) for j in range(n_fungi)]  # (a[0] * 255).repeat(3, 1, 1).cpu().numpy()
    i = -1
    outs = [trans(a)]
    for i in tqdm(range(t)):
        debug(i,":")
        debug("a\n", a)

        # death stage
        sums = torch.conv2d(a, w_enzgen, padding=d_death//2).reshape(n_fungi, k, k)  # (nxkxk)

        debug("sums\n", sums)
        # print(sums.shape)
        enz = (sums.permute(1, 2, 0).matmul(M_g))  # kxkxMe
        debug("enz\n", ((enz*100).to(torch.int32).to(dfloat)/100).permute(2,0,1))
        # convolve to get diffused generation
        # pass through an activation function
        enz = act(enz)
        materials = enz.matmul(M_t)  # kxkxMs
        materials[materials < 0] = 0
        debug("act\n", ((enz*100).to(torch.int32).to(dfloat)/100).permute(2,0,1))
        debug("mat\n", materials.permute(2,0,1))
        materials = materials.reshape(k, k, 1, Ms)
        ########### conv ##########
        materials = materials.permute(3, 2, 0, 1)  # Ms, 1, k, k
        needs = torch.conv2d(a, w_absorb, padding=d_death//2)  # 1, Ms, k, k
        needs = needs.permute(1, 0, 2, 3)
        needs[(needs < 1e-08)] = 1
        debug("Needs\n", needs)
        materials /= needs  # Ms, 1, k, k
        debug("Material\n", materials)
        #print(materials.shape)
        materials = torch.conv2d(materials, w_d[0:1, 0:1], padding=d_death//2)  # Ms, 1, k, k
        #print(materials.shape)

        materials = materials.permute(2, 3, 1, 0)  # k,k,1,Ms
        ########## conv ###########
        debug("mat\n", materials.permute(3, 2,0,1))
        debug("Mr\n", M_r)
        fertilize = materials * M_absorb / M_r  # k,k,n,Ms
        fertilize[fertilize > 1] = 1

        fertilize[fertilize != fertilize] = 1  # deal with NaNs

        toxics = ((materials >= 0) & (M_r < 0))
        #debug("toxics", fertilize[toxics])
        sig = torch.sigmoid((fertilize[toxics] + 1) * 10.)
        #debug("sig", sig)
        fertilize[toxics] = sig
        fertilize = fertilize.permute(3,2,0,1)  # M, n, k, k

        debug("fertilize\n", fertilize)
        #debug("toxic\n", toxics)

        survive_rates = fertilize.prod(dim=0)  # n, k, k
        debug("sur\n", survive_rates)
        death_rates = 1 - survive_rates
        death_rates = death_rates.reshape(1, n_fungi, k, k)

        debug("death\n", death_rates)
        death_rates[death_rates > 1] = 1  # death rate exceed
        rr = torch.rand((1, n_fungi, k, k), device=device) * (1 - eps)
        debug("rr, k\n", rr, k)
        debug("sample_1\n", death_rates + rr)
        sample = (death_rates + rr).to(dint)
        sample[a < 0.1] = 0
        debug("sample\n", sample)
        a = a - sample
        debug("after a\n", a)
        if torch.max(a.sum(dim=1)) > 1 or torch.min(a) < 0:
            print(torch.max(a.sum(dim=1)), torch.min(a) < 0)

            import json
            fp=open('./o.txt', 'w')
            json.dump(a.tolist(), fp=fp)
            fp.close()
            for args, kwargs in logs:
                print(*args, **kwargs)
            a = a.permute(1, 0, 2, 3)
            a[:, a.sum(dim=0) > 1] = 0
            a = a.permute(1, 0, 2, 3)
            a[a < 0] = 0
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
            print(torch.max(a.sum(dim=1)), torch.min(a) < 0)

            import json
            fp=open('./o.txt', 'w')
            json.dump(a.tolist(), fp=fp)
            fp.close()
            for args, kwargs in logs:
                print(*args, **kwargs)
            a = a.permute(1, 0, 2, 3)
            a[:, a.sum(dim=0) > 1] = 0
            a = a.permute(1, 0, 2, 3)
            a[a < 0] = 0
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
            torch.repeat_interleave(torch.arange(k), k).to(device),
            torch.arange(k).repeat(k).to(device)
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
    print([ski[0, i].sum() for i in range(n_fungi)])
    nums = fungisim_micro(ski, None, t)  # num a 2-d list
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


import sys

# feeee = open('C:\mcm21\codes\output.txt', 'w')

#sys.stdout = feeee
#sys.stderr = feeee
# draw()
draw()

#feeee.close()
