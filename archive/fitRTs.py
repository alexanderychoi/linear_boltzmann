import numpy as np
import problem_parameters as pp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import constants as c
from scipy import integrate

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def func_powerlaw(x, m, c):
    return x**(-m) * c

def func_exponential(x, m, c):
    return np.exp(x*-m) * c

if __name__ == '__main__':
    dat_300 = np.loadtxt(pp.inputLoc+'relaxation_time_300K.dat', skiprows=1)
    dat_77 = np.loadtxt(pp.inputLoc+'relaxation_time_77K.dat', skiprows=1)

    enk_300 = dat_300[:, 3]  # eV
    taus_300 = dat_300[:, 4]  # fs
    rates_300 = dat_300[:, 5]  # THz

    enk_77 = dat_77[:, 3]  # eV
    taus_77 = dat_77[:, 4]  # fs
    rates_77 = dat_77[:, 5]  # THz

    plt.figure()
    plt.semilogy(enk_300-np.min(enk_300), taus_300, '.', markersize=3,label='300 K')
    plt.semilogy(enk_77-np.min(enk_77), taus_77, '.', markersize=3,label='77 K')
    plt.ylabel('Relaxation times (fs)')
    plt.xlabel('Energy above CBM (eV)')
    plt.legend()

    FD_300K = (np.exp((enk_300 * c.e - 6.043 * c.e) / (c.kb_joule * 300)) + 1) ** (-1)
    FD_77K = (np.exp((enk_77 * c.e - 6.043 * c.e) / (c.kb_joule * 77)) + 1) ** (-1)
    sortedInds_300K = np.argsort(enk_300)
    sortedInds_77K = np.argsort(enk_77)
    integrated_300K = integrate.cumtrapz(FD_300K[sortedInds_300K], enk_300[sortedInds_300K] * c.e- 6.043 * c.e, initial=0)
    integrated_77K = integrate.cumtrapz(FD_77K[sortedInds_77K], enk_77[sortedInds_77K] * c.e- 6.043 * c.e, initial=0)

    plt.figure()
    plt.semilogy(enk_300-np.min(enk_300), FD_300K, '.', markersize=3,label='300 K')
    plt.semilogy(enk_77-np.min(enk_77), FD_77K, '.', markersize=3,label='77 K')
    plt.ylabel('Fermi Dirac Distribution (unitless)')
    plt.xlabel('Energy above CBM (eV)')
    plt.legend()

    plt.figure()
    plt.plot(enk_300[sortedInds_300K]-np.min(enk_300), integrated_300K/np.max(integrated_300K), '-', markersize=3,label='300 K')
    plt.plot(enk_77[sortedInds_77K]-np.min(enk_77), integrated_77K/np.max(integrated_77K), '-', markersize=3,label='77 K')
    plt.axhline(1,linestyle='--',c='black')
    plt.ylabel('Population fraction')
    plt.xlabel('Energy above CBM (eV)')
    plt.xlim(0,0.1)
    plt.legend()

    plt.figure()
    plt.plot(enk_300[sortedInds_300K]-np.min(enk_300), taus_300[sortedInds_300K],'-',label='300 K')
    plt.plot(enk_77[sortedInds_77K]-np.min(enk_77), taus_77[sortedInds_77K],'-',label='77 K')
    plt.plot(enk_300[sortedInds_300K]-np.min(enk_300), taus_300[sortedInds_300K],'.',color='black')
    plt.plot(enk_77[sortedInds_77K]-np.min(enk_77), taus_77[sortedInds_77K],'.',color='black')
    plt.yscale('log')
    plt.ylabel('Relaxation times (fs)')
    plt.xlabel('Energy above CBM (eV)')
    plt.xlim([0,0.1])
    plt.legend()


    plt.figure()
    plt.scatter(enk_300-np.min(enk_300), taus_300, s=3,label='300 K',c=integrated_300K/np.max(integrated_300K))
    plt.scatter(enk_77-np.min(enk_77), taus_77, s=3,label='77 K',c=integrated_77K/np.max(integrated_77K))
    plt.yscale('log')
    plt.ylabel('Relaxation times (fs)')
    plt.xlabel('Energy above CBM (eV)')
    plt.legend()

    # plt.figure()
    # plt.plot(enk, rates, '.', markersize=3)
    # plt.ylabel('Scattering rate (THz)')
    # plt.xlabel('Energy (eV)')
    plt.show()