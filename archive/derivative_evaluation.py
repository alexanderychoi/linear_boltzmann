import numpy as np
import time
import constants as c
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
import numpy.linalg
from scipy.sparse import linalg
import occupation_solver
import preprocessing
import material_plotter


if __name__ == '__main__':
    # Create electron and phonon dataframes
    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    electron_df['df/dkx'] = -c.hbar_joule/c.kb_joule/pp.T*electron_df['vx [m/s]']*electron_df['k_FD']*(1-electron_df['k_FD'])

    fields = np.array([1e2,1e3,1e4,4e4,1e5])

    plt.figure()
    plt.plot(electron_df['kx [1/A]'],electron_df['df/dkx'],'.')
    # plt.plot(electron_df['kx [1/A]'],electron_df['k_FD'],'.')


    plt.figure()
    plt.plot(electron_df['kx [1/A]'],electron_df['k_FD'],'.')
    nkpts = len(electron_df)
    fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix2.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))

    ylimit = [1e-9,1e6]

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'Field applied in (1 0 0)'+'\n'+pp.fdmName
    fig, ax = plt.subplots()

    for ee in fields:
        _, _, _, matrix_fd = occupation_solver.gaas_gamma_fdm(fdm, electron_df, ee)
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        electron_df['dfc/dkx'] = np.abs(np.dot(matrix_fd,chi)*c.hbar_joule/c.e/ee)
        ax.plot(electron_df['kx [1/A]'], electron_df['dfc/dkx']/ np.abs(electron_df['df/dkx']), '.',label='{:.3f} kV/cm'.format(ee/1e5))
        print('yeet')
    plt.yscale('log')
    plt.ylim(ylimit)
    plt.legend()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    plt.title(pp.title_str)
    plt.ylabel(r'$\frac{\partial \Delta f}{\partial kx} / \frac{\partial f_0}{\partial kx}$',fontsize=12)
    plt.xlabel('kx [1/A]')

    plt.figure()
    for ee in fields:
        _, _, _, matrix_fd = occupation_solver.gaas_gamma_fdm(fdm, electron_df, ee)
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        electron_df['dfc/dkx'] = np.abs(np.dot(matrix_fd,chi)*c.hbar_joule/c.e/ee)
        plt.plot(electron_df['energy [eV]'], electron_df['dfc/dkx']/ np.abs(electron_df['df/dkx']), '.',label='{:.3f} kV/cm'.format(ee/1e5))
    plt.yscale('log')
    plt.ylim(ylimit)
    plt.legend()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    plt.title(pp.title_str)
    plt.ylabel(r'$\frac{\partial \Delta f}{\partial kx} / \frac{\partial f_0}{\partial kx}$',fontsize=12)
    plt.xlabel('Energy [eV]')

    electron_df['k_FD'] = electron_df['k_FD']*8
    plt.figure()
    for ee in fields:
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        plt.plot(electron_df['kx [1/A]'], np.abs(chi)/electron_df['k_FD'], '.',label='{:.3f} kV/cm'.format(ee/1e5))
    plt.yscale('log')
    plt.ylim(ylimit)
    plt.legend()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    plt.title(pp.title_str)
    plt.ylabel(r'$\frac{\Delta f}{f_0}$',fontsize=12)
    plt.xlabel('kx [1/A]')

    plt.figure()
    for ee in fields:
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        plt.plot(electron_df['energy [eV]'], np.abs(chi)/electron_df['k_FD'], '.',label='{:.3f} kV/cm'.format(ee/1e5))
    plt.yscale('log')
    plt.ylim(ylimit)
    plt.legend()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    plt.title(pp.title_str)
    plt.ylabel(r'$\frac{\Delta f}{f_0}$',fontsize=12)
    plt.xlabel('Energy [eV]')


    plt.show()