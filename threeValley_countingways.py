#!/usr/bin/env python

import preprocessing_largegrid
import numpy as np
import matplotlib as mpl
import os
import pandas as pd
import time

import matplotlib.pyplot as plt

import problemparameters as pp
import matrix_plotter

def way_counter(k, valley_inds_L, valley_inds_X = [], do_X = False):
    """
    This function calculates the number of ways that an electron can scatter to the L valley by first determining which
    k points correspond to the Gamma and L valleys and then counting the number of ways (Gaussian energy weight over
    some limit) that a Gamma electron can couple to the L states as a function of energy.
    """
    spread = 3 * dx

    # def gaussian(x, mu, vmag, stdev=spread):
    #     sigma = stdev - (vmag/1E6) * 0.9 * stdev
    #     vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
    #     return vals

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

    intervalley_L = np.isin(g_df['k+q_inds'].values, valley_inds_L)
    ivweight_L = np.sum(totweight[intervalley_L])
    thistotweight = np.sum(totweight)
    istart = int(np.maximum(np.floor((enk[k-1] - en_axis[0]) / dx) - (4 * spread / dx), 0))
    iend = int(np.minimum(np.floor((enk[k-1] - en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
    intervalley_weight_L[istart:iend] += ivweight_L * gaussian(en_axis[istart:iend], enk[k-1])
    total_delta_weight[istart:iend] += thistotweight * gaussian(en_axis[istart:iend], enk[k-1])

    if do_X:
        intervalley_X = np.isin(g_df['k+q_inds'].values, valley_inds_X)
        ivweight_X = np.sum(totweight[intervalley_X])
        intervalley_weight_X[istart:iend] += ivweight_X * gaussian(en_axis[istart:iend], enk[k-1])

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
    # data_loc = '/home/peishi/nvme/k200-0.4eV/'
    # chunk_loc = '/home/peishi/nvme/k200-0.4eV/chunked/'
    # plots_loc = '/home/peishi/Dropbox (Minnich Lab)/Papers-Proposals-Plots/analysis-noise/'
    data_loc = pp.inputLoc
    # chunk_loc = pp.inputLoc + 'chunked3/'
    chunk_loc = pp.inputLoc + 'chunked-hdd/chunked/'
    outLoc = pp.outputLoc

    _, kpts_df, enk_df, qpts_df, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)
    con = preprocessing_largegrid.PhysicalConstants()
    cart_kpts_df = preprocessing_largegrid.load_vel_data(data_loc, con)

    do_valley_indexing = True
    if do_valley_indexing:
        print('Sorting into Gamma vs L valley by k point magnitude')

        reciplattvecs = np.concatenate((con.b1[np.newaxis, :], con.b2[np.newaxis, :], con.b3[np.newaxis, :]), axis=0)
        fbzcartkpts = preprocessing_largegrid.translate_into_fbz(cart_kpts_df.to_numpy()[:, 2:5], reciplattvecs)
        fbzcartkpts = pd.DataFrame(data=fbzcartkpts, columns=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'])
        fbzcartkpts['energy'] = cart_kpts_df['energy'].values

        # points = fbzcartkpts
        # kmag = np.sqrt((fbzcartkpts['kx [1/A]'] / (2 * np.pi / con.a))**2
        #                + (fbzcartkpts['ky [1/A]'] / (2 * np.pi / con.a))**2
        #                + (fbzcartkpts['kz [1/A]'] / (2 * np.pi / con.a))**2)

        kmag = np.sqrt(fbzcartkpts['kx [1/A]'].values ** 2 + fbzcartkpts['ky [1/A]'].values ** 2 +
                                         fbzcartkpts['kz [1/A]'].values ** 2)

        kx = fbzcartkpts['kx [1/A]'].values
        # valley_key_L = np.array(kmag > 0.3)
        # valley_key_X = np.array(kmag > 0.79)
        valley_key_L = np.array(kmag > 0.3) & np.array(abs(kx) > 0.25) & np.array(abs(kx) < 0.75)
        inverse_key = np.array(kmag < 0.3)
        valley_key_X = np.invert(valley_key_L) &  np.invert(inverse_key)

        valley_inds_L = np.array(cart_kpts_df.loc[valley_key_L, 'k_inds'])
        valley_inds_X = np.array(cart_kpts_df.loc[valley_key_X, 'k_inds'])
        # inverse_key = valley_key_L | valley_key_X
        print(r'There are {:d} kpoints in the $\Gamma$ valley'.format(np.count_nonzero(inverse_key)))
        print('There are {:d} kpoints in the L valley'.format(np.count_nonzero(valley_key_L)))
        print('There are {:d} kpoints in the X valley'.format(np.count_nonzero(valley_key_X)))
        np.save(outLoc+'valley_inds_L',valley_key_L)
        np.save(outLoc+'valley_inds_X',valley_key_X)
        np.save(outLoc+'valley_inds_G',inverse_key)

        gamma_df = fbzcartkpts.loc[inverse_key]
        l_df = fbzcartkpts.loc[valley_key_L]
        x_df = fbzcartkpts.loc[valley_key_X]
        matrix_plotter.bz_3dscatter(gamma_df, True, False)
        matrix_plotter.bz_3dscatter(l_df, True, False)
        matrix_plotter.bz_3dscatter(x_df, True, False)
        matrix_plotter.bz_3dscatter(fbzcartkpts, True, False)

        # np.save('D:/Users/AlexanderChoi/Dropbox (Personal)/Alex Choi/Research/Code/Linear_Boltzmann_Eqn/October_2019/outfile', valley_key)
        # np.save('D:/Users/AlexanderChoi/Dropbox (Personal)/Alex Choi/Research/Code/Linear_Boltzmann_Eqn/October_2019/cartkpts', np.array(cart_kpts_df['k_inds']))

        # Below saves data but then it doesn't get loaded later?
        # np.save(data_loc+'valleykey', valley_key)
        # np.save(data_loc+'cartkpts', np.array(cart_kpts_df['k_inds']))

        # plotting.bz_3dscatter(con,fbzcartkpts[valley_key],enk_df[valley_key])
        # plotting.bz_3dscatter(con,cart_kpts_df[valley_key],enk_df[valley_key])
        gamma_kinds = np.array(cart_kpts_df.loc[inverse_key,'k_inds'])
        # plotting.bz_3dscatter(con,cart_kpts_df[inverse_key],enk_df[inverse_key])

    # I think the proper way to do the counting is a kernel density estimate since you have to sum the contributions for
    # a given energy range for intervalleys, so I added that to the way_counter code.

    enk = cart_kpts_df['energy'].values

    npts = 200  # number of bins
    intervalley_weight_L = np.zeros(npts)
    intervalley_weight_X = np.zeros(npts)
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
            way_counter(gamma_kinds[i1], valley_inds_L, valley_inds_X, True)
            point_end = time.time()
            print('Way step number={:d} at {:.2f} seconds'.format(i1, point_end - start))
        end = time.time()
        print('Way counting took {:.2f} seconds'.format(end - start))
        fractioniv_L = np.divide(intervalley_weight_L, total_delta_weight)
        fractioniv_X = np.divide(intervalley_weight_X, total_delta_weight)
        np.save(data_loc + 'intervalley_fraction_L', fractioniv_L)
        np.save(data_loc + 'intervalley_fraction_X', fractioniv_X)

    else:
        if os.path.isfile(data_loc + 'intervalley_weight_by_en.npy'):
            intervalley_weight = np.load(data_loc + 'intervalley_weight_by_en.npy')
        elif os.path.isfile(data_loc + 'intervalley_fraction.npy'):
            fractioniv = np.load(data_loc + 'intervalley_fraction.npy')
        else:
            exit('Couldn''t find intervalley kernel density data')

    # Plotting
    font = {'size': 12}
    mpl.rc('font', **font)

    # ivw200k = np.load('intervalley_weight_by_en_200K.npy')
    # ivw300k = np.load('intervalley_weight_by_en_300K.npy')
    #
    # plt.figure(figsize=(5, 4.5))
    # ax = plt.axes([0.22, 0.15, 0.73, 0.73])
    # cmap = plt.cm.get_cmap('Oranges', 20)
    # plt.plot(en_axis - enk.min(), ivw200k, linewidth=2, color=cmap(12)[:3], label='200 K')
    # plt.plot(en_axis - enk.min(), ivw300k, linewidth=2, color=cmap(15)[:3], label='300 K')
    # plt.xlim([0, 0.385])
    # plt.xlabel('Energy above CBM (eV)')
    # plt.yticks([])
    # # plt.annotate(r'$\sum_q\delta(\epsilon_k \pm \hbar\omega_q - \epsilon_{k+q})$', (0.025, 9000))
    # # plt.ylabel(r'Intervalley scattering strength ($\sum_q(\delta(\epsilon_k \pm \hbar\omega_q - \epsilon_{k+q}))$)')
    # plt.ylabel(r'Intervalley scattering strength (arb.)')
    # plt.savefig(plots_loc + 'intervalley wrt temp.png', dpi=300)
    # # plt.legend()

    plt.figure(figsize=(6, 5))
    plt.semilogy(en_axis, intervalley_weight_L, label='Intervalley L', linewidth=2)
    plt.semilogy(en_axis, intervalley_weight_X, label='Intervalley X', linewidth=2)
    plt.semilogy(en_axis, total_delta_weight, label='Total', linewidth=2)
    plt.ylim([1E-1, 1.6E9])
    plt.xlim([enk.min(), enk.min() + 0.45])
    plt.xlabel('Energy (eV)')
    plt.ylabel('Coupling strength (arb.)')
    plt.legend()

    plt.figure(figsize=(6, 5))
    plt.plot(en_axis, fractioniv_L, label='Intervalley L', linewidth=2)
    plt.plot(en_axis, fractioniv_X, label='Intervalley X', linewidth=2)
    plt.xlim([enk.min(), enk.min() + 0.45])
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intervalley transitionM N` probability')
    plt.legend()
    # plt.savefig(saveloc + 'phasespace.png', dpi=400)

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