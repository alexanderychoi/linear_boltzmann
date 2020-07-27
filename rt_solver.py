import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import preprocessing
import occupation_solver
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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

    enRT_1 = -(np.sum((b+pre*df['df/dkx'])*(df['energy [eV]']-thermal_energy))/np.sum(f*(df['energy [eV]']-thermal_energy)))**(-1)
    enRT_2 = -(np.sum(np.dot(scm,chi_3_i)*(df['energy [eV]']-thermal_energy))/np.sum(f*(df['energy [eV]']-thermal_energy)))**(-1)

    # momRT_1 = -np.sum((pre*df['df/dkx'])*df['kx [1/A]'])/np.sum(f0*df['kx [1/A]'])
    # momRT_2 = np.sum(np.dot(scm,f0)*df['kx [1/A]'])/np.sum(f0*df['kx [1/A]'])
    print(momRT_1)
    print(momRT_2)

    print(enRT_1)
    print(enRT_2)
    return momRT_1,enRT_1


def plot_relaxation(df,fieldVector):
    momRTs = []
    enRTs = []
    for ee in fieldVector:
        momRT,enRT = relaxation_solver(df,ee)
        momRTs.append(momRT)
        enRTs.append(enRT)
    Vcm = np.array(fieldVector)/1e2
    momRTs_fs = np.array(momRTs)*1e15
    enRTs_fs = np.array(enRTs)*1e15

    plt.figure()
    plt.plot(Vcm,momRTs_fs,'.',label = 'Momentum RT')
    plt.plot(Vcm,enRTs_fs,'.',label = 'Energy RT')
    plt.ylabel(r'Relaxation time (fs)')
    plt.xlabel(r'Electric field ($V \, cm^{-1}$')
    plt.yscale('log')


def lorentzian_RT(f, A, tau):
    return A/(1+(f*2*np.pi*tau)**2)


def fit_lorentzian_rt(freq,quant):
    popt, pcov = curve_fit(lorentzian_RT, freq, quant)
    fit_quant = lorentzian_RT(freq, *popt)
    rt = popt[1]*1e6  # Assuming freq is in GHz, return RT in fs
    A = popt[0]  # The zero frequency limit of quant
    return fit_quant, rt, A


if __name__ == '__main__':
    # Create electron and phonon dataframes
    preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    fields = np.geomspace(1e2,4e4,20)
    freqs = pp.freqVector

    plot_relaxation(electron_df, fields)
    plt.show()