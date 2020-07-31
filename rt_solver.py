import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import preprocessing
import occupation_solver
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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



def relaxation_solver(df, ee):
    """Calculates momentum and energy relaxation times using a definition established in Fischetti in terms of the .
    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
    Returns:
        mobility (double): The value of the mobility carrier energy in m^2/V-s.
    """
    Nuc = pp.kgrid ** 3
    nkpts = len(df)
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    df['df/dkx'] = -c.hbar_joule/c.kb_joule/pp.T*df['vx [m/s]']*df['k_FD']*(1-df['k_FD'])
    f0 = df['k_FD']
    thermal_energy = utilities.mean_energy(np.zeros(len(df)), df)

    print('Starting calculation of RTsE = {:.1e} kV/cm'.format(ee/1e5))
    fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix.mmap', dtype='float64', mode='w+',
                    shape=(nkpts, nkpts))
    _, _, _, fdm = occupation_solver.gaas_gamma_fdm(fdm, df, ee)
    _, _, _, fdm = occupation_solver.gaas_l_fdm(fdm, df, ee)
    chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
    b = np.dot(fdm, chi_3_i)
    del fdm
    pre = ee*c.e/c.hbar_joule
    f = chi_3_i + f0
    momRT_1 = -(np.sum((b+pre*df['df/dkx'])*df['kx [1/A]'])/np.sum(f*df['kx [1/A]']))**(-1)
    momRT_2 = -(np.sum(np.dot(scm,chi_3_i)*df['kx [1/A]'])/np.sum(f*df['kx [1/A]']))**(-1)

    # enRT_1 = -(np.sum((b+pre*df['df/dkx'])*(df['energy [eV]']-thermal_energy))/np.sum(f*(df['energy [eV]']-thermal_energy)))**(-1)
    enRT_1a = -(np.sum((b + pre * df['df/dkx']) * (df['energy [eV]']-np.min(df['energy [eV]']))) / np.sum(
        f * (df['energy [eV]']-np.min(df['energy [eV]'])))) ** (-1)
    enRT_1b = -(np.sum((b + pre * df['df/dkx']) * (df['energy [eV]']-thermal_energy)) / np.sum(
        f * (df['energy [eV]']-thermal_energy))) ** (-1)
    enRT_1c = -(np.sum((b + pre * df['df/dkx']) * (df['energy [eV]']-pp.mu)) / np.sum(
        f * (df['energy [eV]']-pp.mu))) ** (-1)
    enRT_2 = -(np.sum(np.dot(scm,chi_3_i)*(df['energy [eV]']-thermal_energy))/np.sum(f*(df['energy [eV]']-thermal_energy)))**(-1)
    vd_3 = utilities.mean_velocity(chi_3_i, df)

    denom = ee*np.array(vd_3)
    excess_energy_d = utilities.mean_energy(chi_3_i,df) - thermal_energy
    excess_energy_e = utilities.mean_energy(chi_3_i,df) - np.min(df['energy [eV]'])

    enRT_1d = excess_energy_d/denom
    enRT_1e = excess_energy_e/denom

    # momRT_1 = -np.sum((pre*df['df/dkx'])*df['kx [1/A]'])/np.sum(f0*df['kx [1/A]'])
    # momRT_2 = np.sum(np.dot(scm,f0)*df['kx [1/A]'])/np.sum(f0*df['kx [1/A]'])
    print(momRT_1)
    print(momRT_2)

    print(enRT_1a)
    print(enRT_2)
    return momRT_1,enRT_1a, enRT_1b, enRT_1c,enRT_1d,enRT_1e


def plot_relaxation(df,fieldVector):
    momRTs = []
    enRTs_a = []
    enRTs_b = []
    enRTs_c = []
    enRTs_d = []
    enRTs_e = []

    for ee in fieldVector:
        momRT,enRT_a,enRT_b,enRT_c,enRT_d,enRT_e = relaxation_solver(df,ee)
        momRTs.append(momRT)
        enRTs_a.append(enRT_a)
        enRTs_b.append(enRT_b)
        enRTs_c.append(enRT_c)
        enRTs_d.append(enRT_d)
        enRTs_e.append(enRT_e)

    Vcm = np.array(fieldVector)/1e2
    momRTs_fs = np.array(momRTs)*1e15
    enRTs_fs_a = np.array(enRTs_a)*1e15
    enRTs_fs_b = np.array(enRTs_b)*1e15
    enRTs_fs_c = np.array(enRTs_c)*1e15
    enRTs_fs_d = np.array(enRTs_d)*1e15
    enRTs_fs_e = np.array(enRTs_e)*1e15

    plt.figure()
    plt.plot(Vcm,momRTs_fs,'.',label = 'Momentum')
    plt.plot(Vcm,enRTs_fs_a,'.',label = 'Fischetti Energy above CBM')
    plt.plot(Vcm,enRTs_fs_b,'.',label = 'Fischetti Energy above Thermal')
    plt.plot(Vcm,enRTs_fs_c,'.',label = 'Fischetti Energy above Fermi')
    plt.plot(Vcm,enRTs_fs_d,'.',label = 'Harnagel Energy above Thermal')
    plt.plot(Vcm,enRTs_fs_e,'.',label = 'Hartnagel Energy above CBM')

    plt.ylabel(r'Relaxation time (fs)')
    plt.xlabel(r'Electric field ($V \, cm^{-1})$')
    plt.yscale('log')
    plt.legend()


def single_lorentzian_RT(f, A, tau):
    return A/(1+(f*2*np.pi*tau)**2)


def double_lorentzian_RT(f, A1, tau1, A2, tau2):
    return -A1/(1+(f*2*np.pi*tau1)**2) + A2/(1+(f*2*np.pi*tau2)**2)


def fit_single_lorentzian_rt(freq,quant):
    popt, pcov = curve_fit(single_lorentzian_RT, freq, quant,bounds = (0,np.inf))
    fit_quant = single_lorentzian_RT(freq, *popt)
    rt = popt[1]*1e6  # Assuming freq is in GHz, return RT in fs
    A = popt[0]  # The zero frequency limit of quant
    return fit_quant, rt, A


def fit_double_lorentzian_rt(freq,quant,A2,tau2):
    tau2 = tau2*0.85
    tau1 = tau2*10
    popt, pcov = curve_fit(lambda f, A1: double_lorentzian_RT(f,A1,tau1,A2,tau2), freq, quant)
    fit_quant = double_lorentzian_RT(freq, *popt,tau1,A2,tau2)
    rt1 = popt[1]*1e6  # Assuming freq is in GHz, return RT in fs
    rt2 = tau2*1e6  # Assuming freq is in GHz, return RT in fs

    A1 = popt[0]  # The zero frequency limit of quant
    return fit_quant, rt1, rt2, A1, A2


if __name__ == '__main__':
    # Create electron and phonon dataframes
    preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    fields = np.geomspace(1e2,4e4,20)
    freqs = pp.freqVector

    plot_relaxation(electron_df, fields)
    plt.show()