#!/usr/bin/env python

import preprocessing_largegrid
import plotting
import numpy as np
import multiprocessing as mp
import matplotlib as mpl
from functools import partial
import os
import pandas as pd
import time
import numba
import re

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go
import plotly


def way_counter(k, valley_inds):
    """
    This function calculates the number of ways that an electron can scatter to the L valley by first determining which
    k points correspond to the Gamma and L valleys and then counting the number of ways (Gaussian energy weight over
    some limit) that a Gamma electron can couple to the L states as a function of energy.
    """
    spread = 3 * dx

    def gaussian(x, mu, sigma=spread):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)

    g_df = pd.read_parquet(chunk_loc + 'k{:05d}.parquet'.format(k))
    print(r'Loaded k={:d}'.format(k))
    # ems_weight = np.multiply(np.multiply(g_df['BE'].values + 1 - g_df['k+q_FD'].values, g_df['g_element'].values),
    #                          g_df['ems_gaussian'])
    # abs_weight = np.multiply(np.multiply((g_df['BE'].values + g_df['k+q_FD'].values), g_df['g_element'].values),
    #                          g_df['abs_gaussian'])
    #
    # totweight = ems_weight + abs_weight
    totweight = g_df['ems_gaussian'] + g_df['abs_gaussian']

    # if np.sum(totweight) > 0:
    intervalleys = np.isin(g_df['k+q_inds'].values, valley_inds)
    ivweight = np.sum(totweight[intervalleys])
    thistotweight = np.sum(totweight)
    istart = int(np.maximum(np.floor((enk[k-1] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    iend = int(np.minimum(np.floor((enk[k-1] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    intervalley_weight[istart:iend] += ivweight * gaussian(en_axis[istart:iend], enk[k-1])
    total_delta_weight[istart:iend] += thistotweight * gaussian(en_axis[istart:iend], enk[k-1])

    # binind = int(np.floor((enk[k-1] - en_axis[0]) / dx))
    # intervalley_weight[binind] += ivweight
    # numratio = len(g_df.loc[np.in1d(g_df['k+q_inds'], valley_inds)])/len(g_df)
    # print(r'For k={:d}, the L valley momentum coupling weight is {:.14E}'.format(k, num))
    # else:
    #     ivweights = 0
    #     numratio = 0
    # print(r'For k={:d}, the L valley energy+momentum coupling weight is {:.14E}'.format(k, ratio))

    # return ivweights, numratio
    # return ivweight


if __name__ == '__main__':
    # data_loc = 'D:/Users/AlexanderChoi/GaAs_300K_10_19/k200-0.4eV/'
    # chunk_loc = 'D:/Users/AlexanderChoi/GaAs_300K_10_19//chunked/'
    # recip_loc = 'D:/Users/AlexanderChoi/GaAs_300K_10_19/chunked/recips/'
    data_loc = '/home/peishi/nvme/k200-0.4eV/'
    chunk_loc = '/home/peishi/nvme/k200-0.4eV/chunked/'

    _, kpts_df, enk_df, qpts_df, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)
    con = preprocessing_largegrid.PhysicalConstants()
    cart_kpts_df = preprocessing_largegrid.load_vel_data(data_loc, con)

    do_valley_indexing = True
    if do_valley_indexing:
        print('Sorting into Gamma vs L valley by k point magnitude')

        reciplattvecs = np.concatenate((con.b1[np.newaxis, :], con.b2[np.newaxis, :], con.b3[np.newaxis, :]), axis=0)
        fbzcartkpts = preprocessing_largegrid.translate_into_fbz(cart_kpts_df.to_numpy()[:, 2:5], reciplattvecs)
        fbzcartkpts = pd.DataFrame(data=fbzcartkpts, columns=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'])

        # points = fbzcartkpts
        kmag = np.sqrt((fbzcartkpts['kx [1/A]'] / (2 * np.pi / con.a))**2
                       + (fbzcartkpts['ky [1/A]'] / (2 * np.pi / con.a))**2
                       + (fbzcartkpts['kz [1/A]'] / (2 * np.pi / con.a))**2)
        valley_key = kmag > 0.3
        valley_inds = np.array(cart_kpts_df.loc[valley_key, 'k_inds'])
        inverse_key = np.logical_not(valley_key)
        print('There are {:d} kpoints in the L valley'.format(np.count_nonzero(valley_key)))
        # np.save('D:/Users/AlexanderChoi/Dropbox (Personal)/Alex Choi/Research/Code/Linear_Boltzmann_Eqn/October_2019/outfile', valley_key)
        # np.save('D:/Users/AlexanderChoi/Dropbox (Personal)/Alex Choi/Research/Code/Linear_Boltzmann_Eqn/October_2019/cartkpts', np.array(cart_kpts_df['k_inds']))

        # Below saves data but then it doesn't get loaded later?
        np.save(data_loc+'valleykey', valley_key)
        np.save(data_loc+'cartkpts', np.array(cart_kpts_df['k_inds']))

        # plotting.bz_3dscatter(con,fbzcartkpts[valley_key],enk_df[valley_key])
        # plotting.bz_3dscatter(con,cart_kpts_df[valley_key],enk_df[valley_key])
        gamma_kinds = np.array(cart_kpts_df.loc[inverse_key,'k_inds'])
        # plotting.bz_3dscatter(con,cart_kpts_df[inverse_key],enk_df[inverse_key])

    # I think the proper way to do the counting is a kernel density estimate since you have to sum the contributions for
    # a given energy range for intervalleys, so I added that to the way_counter code.

    enk = cart_kpts_df['energy'].values

    npts = 200  # number of bins
    intervalley_weight = np.zeros(npts)
    total_delta_weight = np.zeros(npts)
    en_axis = np.linspace(enk.min(), enk.max() + 0.1, npts)
    dx = (en_axis.max() - en_axis.min()) / npts

    count_intervalleys = True
    if count_intervalleys:
        # gamma_ratio = np.zeros((np.count_nonzero(inverse_key), 1))
        # number = np.zeros((np.count_nonzero(inverse_key), 1))
        # print('The number of kpoints in the Gamma valley is {:d}'.format(np.count_nonzero(inverse_key)))

        start = time.time()
        for i1 in range(len(gamma_kinds)):
            way_counter(gamma_kinds[i1], valley_inds)
            point_end = time.time()
            print('Way step number={:d} at {:.2f} seconds'.format(i1, point_end - start))
        end = time.time()
        print('Way counting took {:.2f} seconds'.format(end - start))
        fractioniv = np.divide(intervalley_weight, total_delta_weight)
        np.save(data_loc + 'intervalley_fraction', fractioniv)
    else:
        if os.path.isfile(data_loc + 'intervalley_weight_by_en.npy'):
            intervalley_weight = np.load('intervalley_weight_by_en.npy')
        elif os.path.isfile(data_loc + 'intervalley_fraction.npy'):
            fractioniv = np.load('intervalley_fraction.npy')
        else:
            exit('Couldn''t find intervalley kernel density data')

    # Plotting
    font = {'size': 14}
    mpl.rc('font', **font)

    # ivw200k = np.load('intervalley_weight_by_en_200K.npy')

    fig = plt.figure(figsize=(6, 5))
    plt.axes([0.2, 0.14, 0.7, 0.7])
    plt.plot(en_axis, fractioniv, linewidth=2.5, color='darkred', label='300 K')
    # plt.plot(en_axis, ivw200k, linewidth=2.5, color='red', label='200 K')
    plt.xlim([enk.min(), enk.min() + 0.4])
    plt.xlabel('Energy (eV)')
    plt.ylabel('Fraction of scattering intervalley')
    plt.legend()

    saveloc = '/home/peishi/calculations/first-principles-fluctuations/'
    plt.savefig(saveloc+'fraction_intervalley.png', dpi=400)

    plt.figure(figsize=(6, 5))
    plt.semilogy(en_axis, intervalley_weight, label='Intervalley', linewidth=2)
    plt.semilogy(en_axis, total_delta_weight, label='Total', linewidth=2)
    plt.ylim([1E5, 1.6E9])
    plt.xlim([enk.min(), enk.min() + 0.4])
    plt.xlabel('Energy (eV)')
    plt.ylabel('Phase space volume (arb.)')
    plt.legend()
    plt.savefig(saveloc + 'phasespace.png', dpi=400)

    plt.show()

    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    # # Turn off axis lines and ticks of the big subplot
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['right'].set_color('none')
    # ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    #
    # ax1.scatter(cart_kpts_df.loc[inverse_key, 'energy'], gamma_ratio)
    # ax2.scatter(cart_kpts_df.loc[inverse_key, 'energy'], number)
    # plt.xlabel('Energy (eV)')
    #
    # # Set common labels
    # ax.set_xlabel('Energy (eV)')
    # ax1.set_ylim([np.min(gamma_ratio), 1.5*np.max(gamma_ratio)])
    #
    # ax1.set_title('Energy + Momentum Weighting Normalized to Self')
    # ax2.set_title('Momentum Weighting')
    #
    # plt.savefig('common_labels.png', dpi=400)
    # plt.show()

    # fig = plt.figure(2)
    # ax = fig.add_subplot(111)
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    # # Turn off axis lines and ticks of the big subplot
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['right'].set_color('none')
    # ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    #
    # ax1.scatter(cart_kpts_df.loc[inverse_key, 'energy'], np.divide(gamma_ratio, total))
    # ax2.scatter(cart_kpts_df.loc[inverse_key, 'energy'], number)
    # plt.xlabel('Energy (eV)')
    #
    # # Set common labels
    # ax.set_xlabel('Energy (eV)')
    # ax1.set_ylim([np.min(gamma_ratio), 1.5*np.max(gamma_ratio)])
    #
    # ax1.set_title('Energy + Momentum Weighting Normalized to Total')
    # ax2.set_title('Momentum Weighting')
    #
    # plt.savefig('common_labels2.png', dpi=400)
    # plt.show()













