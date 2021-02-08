import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import scipy.stats as st
from scipy.interpolate import interp1d
def process_env_curves():

    # read from file
    fn_mois = 'data/Fungi_moisture_curves.csv'
    fn_temp = 'data/Fungi_temperature_curves.csv'
    fn_dtemp = 'data/decomp_temperature_curves.csv'
    fn_etemp = 'data/ext_temperature_curves.csv'
    fn_tdata = 'data/Fungal_trait_data.csv'

    mois = pd.read_csv(fn_mois)
    temp = pd.read_csv(fn_temp)
    dtemp = pd.read_csv(fn_dtemp, header=None, names=[str(i) for i in range(5)])
    etemp = pd.read_csv(fn_etemp, header=None, names=[str(i) for i in range(4)])


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

    fungis = seta.intersection(setb).intersection(setc)

    FUNGIS = set(fungi_lookup[i] for i in fungis)
    print(len(fungis))


    mois = mois[mois['species'].isin(fungis)]
    temp = temp[temp['species'].isin(fungis)]
    dtemp = dtemp[dtemp['0'].isin(FUNGIS)]
    etemp = etemp[etemp['0'].isin(FUNGIS)]

    # fit

    tpcurves = {}

    # save fungi curves

    if False:
        for fungi in (fungis):
            print(fungi)
            d1 = temp[temp['species'] == fungi]
            d2 = mois[mois['species'] == fungi]

            f1 = {}
            for data in d1.iterrows():
                if data[1]['type'] == 'smoothed':
                    f1[(data[1]['temp_c'])] = (data[1]['hyphal_rate'])
            f2 = {}
            for data in d2.iterrows():
                if data[1]['type'] == 'smoothed':
                    f2[(data[1]['matric_pot'])] = (data[1]['hyphal_rate'])
            tpcurves[fungi] = [f1, f2]

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
        #sfp1 = open('data/processed_temp_mois_curve.json', 'w')
        #json.dump(tpcurves, sfp1)
        #sfp1.close()



    # deal with decomposition and extension rate
    for fungi in fungis:
        print(fungi)
        FUNGI = fungi_lookup[fungi]
        d1 = dtemp[dtemp['0'] == FUNGI]
        d2 = etemp[etemp['0'] == FUNGI]
        # print(d1[0, :])
        xs, ys = [], []
        xes, yes = [], []
        for i in range(1, 4):
            x, xerr = ((d1.iloc[0][str(i)]).split('±'))
            y, yerr = ((d2.iloc[0][str(i)]).split('±'))
            x = float(x)
            y = float(y)
            print(x, xerr, y, yerr)
            xerr, yerr = float(xerr), float(yerr)
            # x = np.log(x)
            xs.append(x)
            ys.append(y)
            xes.append(xerr)
            yes.append(yerr)
        xs = np.array(xs)  # decomposition rate
        ys = np.array(ys)  # extension rate

        xes = np.array(xes)
        yes = np.array(yes)
        xs, ys = ys, xs
        xes, yes = yes, xes
        f = interp1d(xs, ys, kind=5)
        ys = np.log(ys)
        coef = np.polyfit(xs, ys, 1)
        x_s = np.arange(101) / 100 * (xs[-1] - xs[0]) + xs[0]
        #y_fit = np.polyval(coef, x_s)
        # if coef[0] < 0:
        #    lgy = np.log(ys)
        #    coeflg = np.ployfit(xs, lgy, 1)
        y_fit = f(x_s)
        plt.plot(x_s, y_fit)
        plt.plot(xs, ys, '.')
        # plt.errorbar(xs, ys, linestyle='None', xerr=xes, yerr=yes, )
    plt.show()





if __name__ == '__main__':
    process_env_curves()