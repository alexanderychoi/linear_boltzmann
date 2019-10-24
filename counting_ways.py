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

def way_counter(k,valley_inds):
    """
    This function calculates the number of ways that an electron can scatter to the L valley by first determining which
    k points correspond to the Gamma and L valleys and then counting the number of ways (Gaussian energy weight over
    some limit) that a Gamma electron can couple to the L states as a function of energy.
    """

    g_df = pd.read_parquet(chunk_loc + 'k{:05d}.parquet'.format(k))
    print(r'Loaded k={:d}'.format(k))
    ems_weight = np.multiply(np.multiply(g_df['BE'].values + 1 - g_df['k+q_FD'].values, g_df['g_element'].values),
                             g_df['ems_gaussian'])
    abs_weight = np.multiply(np.multiply((g_df['BE'].values + g_df['k+q_FD'].values), g_df['g_element'].values),
                             g_df['abs_gaussian'])

    g_df['weight'] = ems_weight+abs_weight

    if np.sum(g_df['weight']) > 0:
        ratio = np.sum(g_df.loc[np.in1d(g_df['k+q_inds'],valley_inds), 'weight'])
        num = len(g_df.loc[np.in1d(g_df['k+q_inds'],valley_inds)])/len(g_df)
        print(r'For k={:d}, the L valley momentum coupling weight is {:.14E}'.format(k, num))

    else:
        ratio = 0
        num = 0
    print(r'For k={:d}, the L valley energy+momentum coupling weight is {:.14E}'.format(k, ratio))


    return ratio,num



if __name__ == '__main__':
    data_loc = 'D:/Users/AlexanderChoi/GaAs_300K_10_19/k200-0.4eV/'
    chunk_loc = 'D:/Users/AlexanderChoi/GaAs_300K_10_19//chunked/'
    recip_loc = 'D:/Users/AlexanderChoi/GaAs_300K_10_19/chunked/recips/'

    _, kpts_df, enk_df, qpts_df, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)
    con = preprocessing_largegrid.PhysicalConstants()
    cart_kpts_df = preprocessing_largegrid.load_vel_data(data_loc, con)

    do_valley_indexing = True
    if do_valley_indexing:
        print('Sorting into Gamma vs L valley by k point magnitude')

        reciplattvecs = np.concatenate((con.b1[np.newaxis, :], con.b2[np.newaxis, :], con.b3[np.newaxis, :]), axis=0)
        fbzcartkpts = preprocessing_largegrid.translate_into_fbz(cart_kpts_df.to_numpy()[:, 2:5], reciplattvecs)
        fbzcartkpts = pd.DataFrame(data=fbzcartkpts, columns=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'])

        points = fbzcartkpts
        points = np.sqrt((points['kx [1/A]'] / (2 * np.pi / con.a))**2+(points['ky [1/A]'] / (2 * np.pi / con.a))**2+(points['kz [1/A]'] / (2 * np.pi / con.a))**2)
        valley_key = points > 0.3
        valley_inds = np.array(cart_kpts_df.loc[valley_key,'k_inds'])
        inverse_key = points < 0.3
        print(sum(valley_key))
        np.save('D:/Users/AlexanderChoi/Dropbox (Personal)/Alex Choi/Research/Code/Linear_Boltzmann_Eqn/October_2019/outfile',valley_key)
        np.save('D:/Users/AlexanderChoi/Dropbox (Personal)/Alex Choi/Research/Code/Linear_Boltzmann_Eqn/October_2019/cartkpts',np.array(cart_kpts_df['k_inds']))

        # plotting.bz_3dscatter(con,fbzcartkpts[valley_key],enk_df[valley_key])
        # plotting.bz_3dscatter(con,cart_kpts_df[valley_key],enk_df[valley_key])
        k_inds = np.array(cart_kpts_df.loc[inverse_key,'k_inds'])
        # plotting.bz_3dscatter(con,cart_kpts_df[inverse_key],enk_df[inverse_key])

    gamma_ratio = np.zeros((sum(inverse_key),1))
    number = np.zeros((sum(inverse_key),1))
    print(len(gamma_ratio))
    start = time.time()

    for i1 in range(len(gamma_ratio)):
        gamma_ratio[i1],number[i1] = way_counter(k_inds[i1],valley_inds)
        point_end = time.time()
        print('Way step number={:d} at {:.2f} seconds'.format(i1,point_end - start))

    end = time.time()
    print('Way counting took {:.2f} seconds'.format(end - start))

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    ax1.scatter(cart_kpts_df.loc[inverse_key,'energy'], gamma_ratio)
    ax2.scatter(cart_kpts_df.loc[inverse_key, 'energy'], number)
    plt.xlabel('Energy (eV)')

    # Set common labels
    ax.set_xlabel('Energy (eV)')
    ax1.set_ylim([np.min(gamma_ratio), 1.5*np.max(gamma_ratio)])

    ax1.set_title('Energy + Momentum Weighting Normalized to Self')
    ax2.set_title('Momentum Weighting')

    plt.savefig('common_labels.png', dpi=600)
    plt.show()

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
    # ax1.scatter(cart_kpts_df.loc[inverse_key,'energy'], np.divide(gamma_ratio,total))
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
    # plt.savefig('common_labels2.png', dpi=600)
    # plt.show()













