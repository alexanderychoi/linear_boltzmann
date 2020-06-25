import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd
import noise_solver
import occupation_plotter
from matplotlib.font_manager import FontProperties
import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import material_plotter

if __name__ == '__main__':
    fields = pp.fieldVector
    freq = pp.freqGHz
    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    material_plotter.bz_3dscatter(electron_df, True, False)

    df = electron_df.loc[(electron_df['kx [1/A]']<-0.5)&(electron_df['ky [1/A]']>-0.5)].sort_values(['k_inds','energy [eV]'])
    # df = electron_df.loc[(electron_df['kz [1/A]']>0.5)&(np.abs(electron_df['kx [1/A]'])<0.5)].sort_values(['k_inds','energy [eV]'])
    df['kx [1/A]'] = np.around(df['kx [1/A]'].values,11)
    df['ky [1/A]'] = np.around(df['ky [1/A]'].values,11)
    df['kz [1/A]'] = np.around(df['kz [1/A]'].values,11)

    # band_vec = np.zeros(len(df))
    # kinds_vec = np.unique(df['k_inds'].values)
    # for i1 in range(len(kinds_vec)):
    #     inds = np.where(df['k_inds'].values== kinds_vec[i1])
    #     inds = inds[0]
    #     band_vec[inds[0]] = 1
    #     band_vec[inds[1]] = 2
    # df['band'] = band_vec
    # print('Done assigning band index.')


    plt.figure()
    plt.plot(df['kx [1/A]'],df['energy [eV]'],'.')
    plt.ylabel('Energy [eV]')
    plt.xlabel('kx [1/A]')

    df2 = df.loc[df['bands'] ==1]

    fig, ax = plt.subplots()
    uniq_yz = np.unique(df2[['ky [1/A]', 'kz [1/A]']].values, axis=0)
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        slice_df = df2.loc[(df['ky [1/A]'] == ky) & (df2['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],
                                                                                         ascending=True)
        ax.plot(slice_df['kx [1/A]'],slice_df['k_FD'],color='black',linewidth=0.5)
        ax.plot(slice_df['kx [1/A]'],slice_df['k_FD'],'.',color='black')
    plt.yscale('log')


    fig, ax = plt.subplots()
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        slice_df = df2.loc[(df2['ky [1/A]'] == ky) & (df2['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],
                                                                                         ascending=True)
        ax.plot(slice_df['kx [1/A]'],slice_df['energy [eV]'],color='black',linewidth=0.5)
        ax.plot(slice_df['kx [1/A]'],slice_df['energy [eV]'],'.',color='black')

    fig, ax = plt.subplots()
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        slice_df = df2.loc[(df2['ky [1/A]'] == ky) & (df2['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],
                                                                                          ascending=True)
        ax.plot(slice_df['kx [1/A]'], slice_df['vx [m/s]'], '.', color='black')


    df2 = df.loc[df['bands'] ==2]

    fig, ax = plt.subplots()
    uniq_yz = np.unique(df2[['ky [1/A]', 'kz [1/A]']].values, axis=0)
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        slice_df = df2.loc[(df['ky [1/A]'] == ky) & (df2['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],
                                                                                         ascending=True)
        ax.plot(slice_df['kx [1/A]'],slice_df['k_FD'],color='black',linewidth=0.5)
        ax.plot(slice_df['kx [1/A]'],slice_df['k_FD'],'.',color='black')
    plt.yscale('log')


    fig, ax = plt.subplots()
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        slice_df = df2.loc[(df2['ky [1/A]'] == ky) & (df2['kz [1/A]'] == kz)].sort_values(by=['kx [1/A]'],
                                                                                         ascending=True)
        ax.plot(slice_df['kx [1/A]'],slice_df['energy [eV]'],color='black',linewidth=0.5)
        ax.plot(slice_df['kx [1/A]'],slice_df['energy [eV]'],'.',color='black')
    plt.show()