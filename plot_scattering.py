#!/usr/bin/python3
import preprocessing_largegrid

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import problemparameters as pp
import utilities

def plot_scattering_rates(data_dir, energies):
    rates = np.load(data_dir + 'scattering_rates.npy')
    rates = rates * (2*np.pi)**2
    plt.figure()
    plt.plot(energies, rates, '.', MarkerSize=3, label = 'Parallel RTA')
    plt.xlabel('Energy [eV]')
    plt.ylabel(r'Scattering rate [ps$^{-1}$]')
    plt.title(r'160^3 Grid, 0.4 eV Cutoff')
    print(rates[0])

    return rates


def plot_scattering_rates_matrix(inLoc,df,applyscmFac=False,simplelin=True):
    """Takes chi solutions which are already calculated and plots drift velocity vs field
    Parameters:
        inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
        linearization by default.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        applyscmFac (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.

    Returns:
        Nothing. Just the plots.
    """
    if applyscmFac:
        scmfac = (2*np.pi)**2
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1
    nkpts = len(df)
    scm = np.memmap(inLoc + 'scattering_matrix.mmap', dtype='float64', mode='r', shape=(nkpts, nkpts))
    if simplelin:
        rates = (-1) * np.diag(scm) * scmfac * 1E-12
    else:
        chi2psi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
        rates = (-1) * np.diag(scm) * scmfac * 1E-12 / chi2psi
    plt.plot(df['energy'], rates, '.', MarkerSize=3, label = 'Matrix Diagonal')
    plt.xlabel('Energy [eV]')
    plt.ylabel(r'Scattering rate [ps$^{-1}$]')
    print(rates[0])

    return rates


if __name__ == '__main__':
    # data_Loc = pp.inputLo
    data_Loc = 'C:/Users/TheDingDongDiddler/Desktop/DownloadFromComet/'
    electron_df = pd.read_pickle(data_Loc+'electron_df.pkl')
    electron_df = utilities.fermi_distribution(electron_df)
    chi2psi = np.squeeze(electron_df['k_FD'] * (1 - electron_df['k_FD']))

    other = electron_df['energy'].sort_values(ascending=True)
    other_inds = other.index
    enk_array = np.loadtxt(data_Loc+'gaas.enk')
    enk = pd.DataFrame(data=enk_array, columns=['k_inds', 'band_inds', 'energy [Ryd]'])
    enk[['k_inds', 'band_inds']] = enk[['k_inds', 'band_inds']].astype(int)

    ems_c = np.load(data_Loc+'ems_c.npy')
    ems_s = np.load(data_Loc+'ems_s.npy')
    ems_w = np.load(data_Loc+'ems_w.npy')
    plt.figure()
    plt.plot(ems_s/chi2psi,Label='Simple')
    plt.plot(ems_c/chi2psi,Label='Canonical')
    plt.plot(ems_w/chi2psi,label='Wu Li')
    plt.ylabel('Emission weight/(f0*(1-f0))')
    plt.legend()

    abs_c = np.load(data_Loc+'abs_c.npy')
    abs_s = np.load(data_Loc+'abs_s.npy')
    plt.figure()
    plt.plot(abs_s/chi2psi,Label='Simple')
    plt.plot(abs_c/chi2psi,Label='Canonical',alpha=0.4)
    plt.ylabel('Sum over absorption weight')
    plt.legend()

    plt.figure()
    plt.plot(abs_s/abs_c)
    plt.ylabel('abs_s/abs_c')
    plt.xlabel('kpt index')
    plt.title(r'160^3 Grid, 0.4 eV Cutoff, 200 K')

    plt.figure()
    plt.plot(ems_s/ems_c)
    plt.ylabel('ems_s/ems_c')
    plt.xlabel('kpt index')
    plt.title(r'160^3 Grid, 0.4 eV Cutoff, 200 K')

    plt.figure()
    plt.plot(electron_df['energy'].sort_values(ascending=True),ems_w[other_inds]/ems_c[other_inds])
    plt.ylabel('ems_w/ems_c')
    plt.xlabel('Energy [eV]')
    plt.title(r'160^3 Grid, 0.4 eV Cutoff, 200 K')

    plt.figure()
    plt.plot(electron_df['energy'].sort_values(ascending=True),chi2psi[other_inds])
    plt.ylabel('f0*(1-f0)')
    plt.xlabel('Energy [eV]')
    plt.title(r'160^3 Grid, 0.4 eV Cutoff, 200 K')

    plt.figure()
    plt.plot(ems_w/ems_s)
    plt.ylabel('ems_w/ems_s')
    plt.xlabel('kpt index')
    plt.title(r'160^3 Grid, 0.4 eV Cutoff, 200 K')

    plt.figure()
    plt.plot(abs_s/abs_c)
    plt.ylabel('Sum over absorption weight ratio')
    plt.legend()


    rates_diag = plot_scattering_rates(data_Loc,enk['energy [Ryd]']* 13.6056980659)
    rates_matrix = plot_scattering_rates_matrix(data_Loc, electron_df, True, False)
    plt.legend()

    plt.figure()
    plt.plot(rates_diag/rates_matrix)

    chi2psi = np.squeeze(electron_df['k_FD'] * (1 - electron_df['k_FD']))
    plt.figure()
    plt.plot(1/chi2psi)
    plt.xlabel('kpt')
    plt.ylabel('1/(f0(1-f0))')
    plt.show()


