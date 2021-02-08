import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import scipy.stats as st
from scipy.interpolate import interp1d

fn_mois = 'data/Fungi_moisture_curves.csv'
fn_temp = 'data/Fungi_temperature_curves.csv'
fn_dtemp = 'data/decomp_temperature_curves.csv'
fn_etemp = 'data/ext_temperature_curves.csv'
fn_tdata = 'data/Fungal_trait_data.csv'
fn_battle = 'data/fungi_battle.csv'  # https://github.com/dsmaynard/diversity_begets_diversity
fn0 = 'data/fungi_list.json'
fn1 = 'data/processed_temp_mois_curve.json'
fn2 = 'data/processed_decomp_ext_curve.json'
fn3 = 'data/battle_results.json'

def process_env_curves():

    # read from file


    mois = pd.read_csv(fn_mois)
    temp = pd.read_csv(fn_temp)
    dtemp = pd.read_csv(fn_dtemp, header=None, names=[str(i) for i in range(5)])
    etemp = pd.read_csv(fn_etemp, header=None, names=[str(i) for i in range(4)])
    battles = pd.read_csv(fn_battle)


    tdata = pd.read_csv(fn_tdata)
    print(len(tdata))

    # get table
    fungi_dict = {}  # long to short
    fungi_lookup = {}  # short to long
    name2, name3 = 'gen_name2', 'name3'
    for data in tdata.iterrows():
        n2, n3 = data[1][name2], data[1][name3]
        fungi_dict[n2] = n3
        fungi_lookup[n3] = n2


    # get real data
    seta = set(mois['species'])
    setb = set(temp['species'])
    setc = set(fungi_dict[i] for i in set(dtemp['0']))
    setd = set(battles['s1'])
    sete = set(battles['s2'])

    fungi = seta.intersection(setb).intersection(setc).intersection(setd).intersection(sete)

    FUNGI = set(fungi_lookup[i] for i in fungi)
    print(len(fungi))

    sfp0 = open(fn0, 'w')
    json.dump(list(fungi), sfp0)
    sfp0.close()


    mois = mois[mois['species'].isin(fungi)]
    temp = temp[temp['species'].isin(fungi)]
    dtemp = dtemp[dtemp['0'].isin(FUNGI)]
    etemp = etemp[etemp['0'].isin(FUNGI)]
    battles = battles[battles['s1'].isin(fungi)]
    battles = battles[battles['s2'].isin(fungi)]

    # fit

    tpcurves = {}

    # save fungi curves

    if False:
        for fungus in (fungi):
            print(fungus)
            d1 = temp[temp['species'] == fungus]
            d2 = mois[mois['species'] == fungus]

            f1 = {}
            for data in d1.iterrows():
                if data[1]['type'] == 'smoothed':
                    f1[(data[1]['temp_c'])] = (data[1]['hyphal_rate'])
            f2 = {}
            for data in d2.iterrows():
                if data[1]['type'] == 'smoothed':
                    f2[(data[1]['matric_pot'])] = (data[1]['hyphal_rate'])
            tpcurves[fungus] = [f1, f2]

            std_t, std_p = 22, -0.5
            base = (f1[std_t] + f2[std_p]) / 2


            def f(x, y):
                return f1[x] * f2[y] / base

            # values
            vals = [[], [], []]
            vals[0] = [i for i in f1]
            vals[1] = [i for i in f2]
            for i in f1:
                for j in f2:
                    vals[2].append(f1[i] * f2[j] / base)
            vals[2] = np.array(vals[2]).reshape((len(f1), len(f2))).T
            # meshgrid
            x,y = np.meshgrid(vals[0], vals[1])
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            #ax.plot_wireframe(x,y, vals[2])

            # SUBPLOTS
            #fig, (ax1, ax2) = plt.subplots(1, 2)
            #ax1.plot([i for i in f1], [f1[i] for i in f1])
            #ax2.plot([i for i in f2], [f2[i] for i in f2])
            #plt.show()
        #sfp1 = open(fn1, 'w')
        #json.dump(tpcurves, sfp1)
        #sfp1.close()

    # deal with decomposition and extension rate
    if True:
        decurves = {}
        for fungus in fungi:
          #  print(fungus)
            FUNGUS = fungi_lookup[fungus]
            d1 = dtemp[dtemp['0'] == FUNGUS]
            d2 = etemp[etemp['0'] == FUNGUS]
            # print(d1[0, :])
            xs, ys = [], []
            xes, yes = [], []
            for i in range(1, 4):
                x, xerr = ((d1.iloc[0][str(i)]).split('±'))
                y, yerr = ((d2.iloc[0][str(i)]).split('±'))
                x = float(x)
                y = float(y)
           #     print(x, xerr, y, yerr)
                xerr, yerr = float(xerr), float(yerr)
                # x = np.log(x)
                xs.append(x)
                ys.append(y)
                xes.append(xerr)
                yes.append(yerr)
            # xs0 = xs[0] - (xs[1] - xs[0]) * 100
            # ys0 = ys[0] - (ys[1] - ys[0]) * 100
            # xs2 = xs[2] - (xs[1] - xs[2]) * 100
            # ys2 = ys[2] - (ys[1] - ys[2]) * 100
            # xs = [xs0] + xs + [xs2]
            # ys = [ys0] + ys + [ys2]
            print(xs, ys)
            decurves[fungus] = (xs, ys)  # dtemp, etemp

            # interpolate
            xs, ys = ys, xs

            ws = np.polyfit(xs, ys, 1)
            # f = interp1d(xs, ys, kind='linear')
            x_s = np.arange(100) / 100 * (xs[-1] - xs[0]) + xs[0]
            y_fit = np.polyval(ws, x_s)
            plt.plot(x_s, y_fit)
            plt.plot(xs, ys, '.')
        plt.show()

        sfp2 = open(fn2, 'w')
        json.dump(decurves, sfp2)
        sfp2.close()

    # fungi battles
    if False:
        battle_record = {i: {j: 0 for j in fungi} for i in fungi}
        for data in battles.iterrows():
            battle = data[1]
            s1, s2 = battle['s1'], battle['s2']
            a, b, c, d, e = battle['s1.rate'], battle['s2.rate'], battle['s1.win'], battle['s1.loss'], battle['draw']
            if a < 0.1:
                print(data[0])
            if b < 0.1:
                print(data[0])
            battle_record[s1][s2] += c + e * min(max((a - b) / 15, 0), 1)
            battle_record[s2][s1] += d + e * min(max((b - a) / 15, 0), 1)

        sfp3 = open(fn3, 'w')
        json.dump(battle_record, sfp3)
        sfp3.close()


def data_conversion():
    # get fungi list
    ret = []
    with open(fn0) as sfp0:
        fungus = json.load(sfp0)
        # fungus = ['p.flav.s', 'p.flav.n', 'm.trem.n', 'm.trem.s', 'p.robin.n', 'p.rufa.acer.s']
        ret.append(fungus)


    # get tp
    with open(fn1) as sfp1:
        tpcurves = json.load(sfp1)
        for fung in fungus:
            f1, f2 = tpcurves[fung]
            f1 = {float(k): v for k, v in f1.items()}
            f2 = {float(k): v for k, v in f2.items()}
            tpcurves[fung] = (f1, f2)
        std_t, std_p = 22, -0.5

        def f(fung, x, y):
            f1, f2 = tpcurves[fung]
            base = (f1[std_t] + f2[std_p]) / 2
            return f1[x] * f2[y] / base

        ret.append(f)


    # get de
    with open(fn2) as sfp2:
        de = json.load(sfp2)
        def f(fung, decomp_rate):
            ys, xs = de[fung]  # xs = extension rate, ys = decomp rate

            ws = np.polyfit(xs, ys, 1)
          #  print(decomp_rate, type(decomp_rate), 'in')
          #  print(type(fp(decomp_rate)), 'out')
            return float(np.polyval(ws, decomp_rate))
        ret.append(f)

    # get battle
    with open(fn3) as sftp3:
        battles = json.load(sftp3)
        ret.append(battles)

    return ret

import torch
def c_curve(t, p):
    k, k2 = 2, 10
    return torch.sigmoid((p - 0.5 + 1.36 * k) / k) * torch.sigmoid((t - 22 + 1.36 * k2) / k2)

if __name__ == '__main__':
    process_env_curves()


    # t = torch.arange(-20,70,0.1)
    # p = torch.arange(-5, 0.5, 0.01)
    # t, p = torch.meshgrid(t, p)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe(t, p, c_curve(t, p))
    # plt.show()
