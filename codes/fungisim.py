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
from preprocess import data_conversion, c_curve


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
d_birth = 11  # must be odd
d_death = 7
# exp. settings
tr = 10  # timestep per sec
k = 100  # width = height
tot_time = 100  # total time span in sec
init_p = 0.03  # initial density
# calculated exp. settings
t = tot_time * tr  # in timesteps

n_fungi = 3

_birth_rates = torch.tensor([[
   2.22, 2.25, 2.15
][:n_fungi]], dtype=dfloat, device=device).reshape((1, n_fungi, 1, 1))  # 1xnx1x1

capacities = torch.tensor([[
    0.741, 0.541, 0.6
][:n_fungi]], dtype=dfloat, device=device).reshape((1, n_fungi, 1, 1))  # 1xnx1x1

# micro
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
concentration_subs = torch.tensor([0.6, 1, 1], device=device, dtype=dfloat)[:Me] / tr # Me

fungi_list, tpcurve, decurve, battles = data_conversion()
n_fungi = len(fungi_list)
# birth_rates: we assume they have the same birth rate.
_birth_rates = torch.ones(n_fungi, dtype=dfloat, device=device).reshape(1, n_fungi, 1, 1)
Me = n_fungi
Ms = n_fungi + 1
# generation matrix.
M_g = torch.eye(Me, device=device, dtype=dfloat) / tr
# colors

colors = np.random.random((n_fungi, 1, 1, 3)) * 0.9 + 0.1
# toxics


M_toxic = torch.zeros(Me, Ms, device=device, dtype=dfloat)
consider_interaction = True
if consider_interaction:
    for i, s1 in enumerate(fungi_list):
        for j, s2 in enumerate(fungi_list):
            M_toxic[i, j] = battles[s2][s1]


M_toxic /= tr
rev_m_toxic = torch.zeros(Me, Ms, device=device, dtype=dfloat)
rev_m_toxic[M_toxic > 0] = 1/M_toxic[M_toxic > 0]


# requirement in std environment
M_need = torch.zeros(Me, Ms, device=device, dtype=dfloat)
for i, fungi in enumerate(fungi_list):
    M_need[i, -1] = decurve(fungi, tpcurve(fungi, 22, -0.5)) / 1.5
M_need /= tr

def load_settings(temperature, moisture):
    global n_fungi, _birth_rates, capacities, Me, Ms, M_t, M_g, M_need, M_toxic, concentration_subs, colors
    # we get hyphal extension rate using temperature and moisture
    hyphal_extension_rate = {}
    for fungi in fungi_list:
        hyphal_extension_rate[fungi] = tpcurve(fungi, temperature, moisture)
    # we convert hyphal extension rate to decomposition rate
    decomposition_rate = {}
    for fungi in fungi_list:
        decomposition_rate[fungi] = decurve(fungi, hyphal_extension_rate[fungi])
    # we represent decomposition rate as enzymatic activity

    # enzyme transition matrix.
    M_t = torch.zeros(Me, Ms, device=device, dtype=dfloat)
    for i, fungi in enumerate(fungi_list):

        M_t[i, i] = decomposition_rate[fungi]
        M_t[i, -1] = decomposition_rate[fungi]

    # print(M_t)
    # print(M_g)
    # print(M_need)
    # print(M_toxic)



    #times.append(time())
    # concentration of substrate.
    concentration_subs = torch.ones(Me, device=device, dtype=dfloat) * c_curve(torch.tensor(temperature),
                                                                               torch.tensor(moisture)) / tr
    #times.append(time())

    # times.append(time())
    # spend = []
    # for i in range(len(times) - 1):
    #     spend.append( times[i+1]-times[i])
    # print(spend)
    # print(">",times[-1]-times[0])
    #print(times)

setting_data = [(
######### setting1 ########
        [22, 26, 26],
        [-0.5, -0.5, -1],
        [0, 30, 60]
    ),
######### setting2 ########
    (
        [25],
        [-0.4],
        [0]
    ),
######### setting3 ########
    (
        [25],
        [-3.0],
        [0]
    ),
######## setting4 #########
    (
        [20],
        [-1.0],
        [0]
    ),
######### setting5 ########
    (
        [12],
        [-1.5],
        [0]
    ),
######### setting6 ########
    (
        [10],
        [-2.0],
        [0]
    ),
######## setting7 #########
    (
        [0],
        [-1.0],
        [0]
    ),
]

setting_no = 0
temp, moist, changes = setting_data[setting_no]

for i in range(len(changes)):
    changes[i] *= tr

def load_environment():
    pass





K = 1.
birth_rate = 1.
r = birth_rate
M_r = torch.Tensor()
M_r2 = torch.Tensor()
M_absorb = torch.Tensor()

w_b = torch.zeros(n_fungi, n_fungi, d_birth, d_birth, dtype=dfloat, device=device)
for i in range(n_fungi):
    make_circular(w_b[i, i])

w_d = torch.zeros(n_fungi, n_fungi, d_death, d_death, dtype=dfloat, device=device)
for i in range(n_fungi):
    make_circular(w_d[i, i])

death_range = w_d[0, 0].sum()
birth_range = w_b[0, 0].sum()

w_absorb0 = torch.zeros(Ms, n_fungi, d_death, d_death, dtype=dfloat, device=device)
for j in range(Ms):
    for i in range(n_fungi):
        make_circular(w_absorb0[j, i])

w_absorb = torch.zeros(Ms, n_fungi, d_death, d_death, dtype=dfloat, device=device)

w_enzgen = torch.zeros(n_fungi, n_fungi, d_death, d_death, dtype=dfloat, device=device)
for i in range(n_fungi):
    make_circular(w_enzgen[i, i])
w_enzgen /= death_range

def setting_calculation():
    #import time
    global w_b, w_d, death_range, birth_range, K, birth_rate, r, t, M_toxic, M_r, M_absorb, w_enzgen, w_absorb, M_r2
    # calculated env. consts
    #tw = time.time()
    K = death_range * capacities
    birth_rate = _birth_rates / tr  # (fungi per sec) / (timestep per sec) = (fungi per timestep)
    r = birth_rate

    #t00 = time.time()



    #t001 = time.time()

    M_r = M_need# - rev_m_toxic
    M_r2 = M_need - rev_m_toxic
    #t01 = time.time()
    M_absorb = M_need # + ((M_toxic > 0) & (M_need == 0)) / tr
    #print("absorb")
    #print(M_absorb)


    #t0=time.time()
    for j in range(Ms):
        for i in range(n_fungi):
            w_absorb[j, i] = w_absorb0[j, i] * M_absorb[i, j]
    #t1=time.time()
    #print(t00-tw, t001-t00, t01-t001, t0-t01, t1-t0, t1-tw)



def s_curve(x):  # -1 <------> 1
    return (torch.sigmoid(x * 2) * 2 - 1)
def act(x):
    # k x k x Me

    return (s_curve(x / concentration_subs) * (concentration_subs))

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

plotting_period = 100000000000000
    # exit(1)
def fungisim_micro(a, age, t):
    def trans(a):
        w1 = dr # a[0].sum()

        out = a[0].cpu().numpy().reshape((n_fungi, k, k, 1))
        #debug("out", a[0])
        if i % plotting_period == 0:
            pic = np.zeros((k, k, 3))
            for j in range(n_fungi):
                pic += colors[j] * out[j]
            #print("pic")
            pyplot.imsave(f'outputs/{i+1}.png', pic)

        return [np.sum(out[j]) for j in range(n_fungi)] + [float(w1)] # (a[0] * 255).repeat(3, 1, 1).cpu().numpy()
    i = -1
    dr = 0

    outs = [trans(a)]
    p = 0
    for i in tqdm(range(t)):
        #import time
        #t0 = time.time()
        if len(changes) > p and changes[p] == i:
            load_settings(temperature=temp[p], moisture=moist[p])
            setting_calculation()
            p+=1
        #t1 = time.time()
        #print(t1-t0)
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
        dr = materials.sum() / k / k
        debug("act\n", ((enz*100).to(torch.int32).to(dfloat)/100).permute(2,0,1))
        debug("mat\n", materials.permute(2,0,1))
        materials = materials.reshape(k, k, 1, Ms)
        gmaterials = materials
        ########### conv ##########
        materials = materials.permute(3, 2, 0, 1)  # Ms, 1, k, k
        # tdr = materials.sum()
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
        debug("mat\n", materials.permute(3, 2, 0, 1))
        debug("Mr\n", M_r)
        #print("sdfsdfsdf")
        # quest = (M_absorb/M_r)
        # print(quest[quest<0].cpu().numpy().tolist())
        # print(len(quest[quest<-1]))
        #print("...")
        #print(len(materials[materials>1]))
        fertilize = materials * M_absorb / M_r  # k,k,n,Ms
        fertilize[fertilize > 1] = 1
        fertilize[fertilize != fertilize] = 1  # deal with NaNs
        fertilize1 = -gmaterials * rev_m_toxic # / 10


        # print(fertilize1.cpu().numpy().tolist())
        ques = materials / M_r2
        toxic = (ques < 0)
        # print("toxic", len(fertilize1[toxic]))

        #print(len(fertilize1[ddd]))
        #print("ccccc")
        #debug("toxics", fertilize[toxics])
        sig = torch.sigmoid((fertilize1 + 23) * 10.)
        # print("sig", torch.min(sig))
        #debug("sig", sig)
        fertilize[toxic] = sig[toxic]
        #print(len(fertilize[fertilize > 1]))

        fertilize = fertilize.permute(3,2,0,1)  # M, n, k, k

        debug("fertilize\n", fertilize)
        #debug("toxic\n", toxics)

        survive_rates = fertilize.prod(dim=0)  # n, k, k
        debug("sur\n", survive_rates)
        death_rates = 1 - survive_rates
        death_rates /= np.sqrt(tr)
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
  #  print(max_indices)
    rand_map = torch.rand((k, k), device=device)
    max_indices = torch.stack(
        [
            max_indices,
            torch.repeat_interleave(torch.arange(k), k).to(device),
            torch.arange(k).repeat(k).to(device)
        ]
    )
   # print(max_indices)
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
    def getcolor(color):
        r, g, b = color[0], color[1], color[2]
        def h(x):
            x = int(x * 255)
            hh = hex(x)
            hh = hh[2:]
            if len(hh) == 1:
                hh = '0' + hh
            return hh
        return '#' + h(r) + h(g) + h(b)
    print(fungi_list)

    final_population = []
    tot = 0
    for _i, num in enumerate(nums[:-1]):
        pyplot.plot(np.arange(0, t + 1) / tr, num, c=getcolor(colors[_i, 0, 0]) )
        final_population.append((fungi_list[_i], num[-1]))
        tot += num[-1]

    final_population.sort(key=lambda x: x[1], reverse=True)
    print("-----final population-----")
    print("\ttot:\t", tot)
    for i, (fungus, pop) in enumerate(final_population):
        print(f"#{i+1}\t{fungus}:\t{pop}")

    print("-----population ratio-----")
    for i, (fungus, pop) in enumerate(final_population):
        print(f"#{i+1}\t{fungus}:\t{pop / tot}")
    pyplot.legend(fungi_list)
    pyplot.xlabel('time')
    pyplot.ylabel('population')

    pyplot.show()

    pyplot.plot(np.arange(0, t + 1) / tr, nums[-1], c='black', label='TDR')
    pyplot.xlabel('time')
    pyplot.ylabel('TDR')
    #pyplot.title('Graph of TDR-time')
    pyplot.show()

        #pyplot.pause(0.01)
    #pyplot.ioff()
    #pyplot.show()
    # write_gif(outs, 'output.gif')




if __name__ == '__main__':
    draw()

#feeee.close()
