import numpy as np
import time
import constants as c
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
import numpy.linalg
from scipy.sparse import linalg
import occupation_solver

# Set the parameters for the paper figures
SMALL_SIZE = 9
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# PURPOSE: THIS MODULE CONTAINS FUNCTIONS USED TO CALCULATE AND PLOT THE POWER SPECTRAL DENSITY OF CURRENT FLUCTUATIONS.
# THE CALCULATION INVOLVES LOADING A STEADY BOLTZMANN TRANSPORT SOLUTION, CALCULATING THE EFFECTIVE DISTRIBUTION FCN
# INDEXED BY FREQUENCY AND FIELD STRENGTH, AND TAKING THE APPROPRIATE BZ SUM OVER THE EFFECTIVE DISTRIBUTION TO GET THE
# POWER SPECTRAL DENSITY. CALCULATES THE TRANSVERSE AND LONGITUDINAL SPECTRAL DENSITY. HARD CODED FOR GAAS.

# ORDER: THIS MODULE REQUIRES THAT STEADY SOLUTIONS HAVE ALREADY BEEN WRITTEN TO FILE BY OCCUPATION_SOLVER.PY.

# OUTPUT: THIS MODULE SAVES THE EFFECTIVE DISTRIBUTION INDEXED BY FREQUENCY AND FIELD TO THE SUBPROBLEM OUTPUT. IT CAN
# PRODUCE PLOTS OF THE SMALL SIGNAL CONDUCTIVITY FOR EXPLORATORY DATA ANALYSIS BUT IT DOES NOT SAVE THOSE PLOTS.


class gmres_counter(object):
    """A class object that can be called during GMRES to print stepwise iterative residual."""
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


def steady_low_field_corr(b, matrix_sc, kptdf, field, freq):
    """Generalized minimal residual solver for calculating the solution to the matix equation Ax = b, where b is to be
        specified and A is the transient relaxation operator (time + scm). Hard-coded for the low field approximation.
        Parameters:
            b (nparray) : The forcing for the matrix equation.
            matrix_sc (memmap): Scattering matrix in simple linearization by default..
            matrix_fd (memmap): Finite difference matrix, generated by apply_centraldiff_matrix.
            kptdf (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
            field (dbl): Value of the electric field in V/m.
            freq (dbl): Frequency in GHz to be used for the transient solution
        Returns:
            x_next (nparray): Numpy array containing the (hopefully) converged iterative solution as chi.
            x_0 (nparray): Numpy array containing the RTA solution as chi.
            counter.niter (dbl): Number of iterations to reach desired relative convergence
        """
    counter = gmres_counter()
    print('Starting low-field inverse relaxation solver for {:.3E} and {:.3E} GHz'.format(field, freq))
    if pp.scmBool:
        scmfac = pp.scmVal
        print('Applying correction factor to the scattering matrix.')
    else:
        scmfac = 1
        print('Not applying correction factor to the scattering matrix.')

    loopstart = time.time()
    chi2psi = np.squeeze(kptdf['k_FD'] * (1 - kptdf['k_FD']))
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    x_0 = b * invdiag
    tau_0 = (np.sum(np.diag(matrix_sc) * kptdf['k_FD']) / np.sum(kptdf['k_FD'])) ** (-1)

    x_smrta = b * np.ones(len(kptdf)) * tau_0
    if freq > 0:
        freq_matrix = np.diag(np.ones(len(kptdf)) * 1j * 10 ** 9 * 2 * np.pi * freq)  # Positive quantity
        print('Starting GMRES solver.')
        x_next, criteria = linalg.gmres(freq_matrix + matrix_sc * scmfac, b, x0=x_0, tol=pp.relConvergence,
                                        callback=counter, atol=pp.absConvergence)
    if freq == 0:
        print('Starting GMRES solver.')
        x_next, criteria = linalg.gmres(matrix_sc * scmfac, b, x0=x_0, tol=pp.relConvergence,
                                        callback=counter, atol=pp.absConvergence)
        freq_matrix = np.diag(np.zeros(len(kptdf)))
    print('GMRES convergence criteria: {:3E}'.format(criteria))
    # The following step is the calculation of the relative residual, which involves another MVP. This adds expense. If
    # we're confident in the convergence, we can omit this check to increase speed.
    if pp.verboseError:
        b_check = np.dot(freq_matrix + matrix_sc * scmfac, x_next)
        error = np.linalg.norm(b_check - b) / np.linalg.norm(b)
        print('Norm of b is {:3E}'.format(np.linalg.norm(b)))
        print('Absolute residual error is {:3E}'.format(np.linalg.norm(b_check - b)))
        print('Relative residual error is {:3E}'.format(error))
    else:
        error = 0
        print('Error not stored.')
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if not pp.simpleBool:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in canonical linearization')
        x_next = x_next * chi2psi
        x_0 = x_0 * chi2psi
        x_smrta = x_smrta * chi2psi
    return x_next, x_smrta, error, counter.niter


def write_low_field_correlation(fieldVector,df,freqVector):
    """Calls the GMRES solver hard coded for solving the effective BTE w/low-field approx.
    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        None. Just writes the chi solutions to file. chi_#_EField. #1 corresponds to low-field RTA, #2 corresponds to
        low-field iterative, #3 corresponds to full finite-difference iterative.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    for freq in freqVector:
        for ee in fieldVector:
            chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}.npy".format(ee))
            f0 = df['k_FD'].values
            f = chi + f0
            Vx = utilities.mean_velocity(chi, df)
            b_xx = (-1) * ((df['vx [m/s]'] - Vx) * f)
            corr_xx,_,_,_ = steady_low_field_corr(b_xx, scm, df, ee, freq)
            np.save(pp.outputLoc + 'SB_Density/' + 'xx_' + '2_' + "f_{:.1e}_E_{:.1e}".format(freq,ee),corr_xx)
            print('Transient solution written to file for ' + "{:.1e} V/m and {:.1e} GHz".format(ee,freq))
            print('\n \n')


def lowfield_density(chi, EField,df,freq, partialSum = False, cutoff = 0):
    f0 = df['k_FD'].values
    f = chi + f0
    Nuc = pp.kgrid ** 3
    conductivity_xx = 2 * c.e / (Nuc * c.Vuc * EField) * np.sum(chi * df['vx [m/s]'])
    n = utilities.calculate_noneq_density(chi,df)
    mobility = conductivity_xx/c.e/n
    print('Mobility is {:3f} cm^2/(V-s)'.format(mobility*100**2))
    conductivity_yy = 2 * c.e / (Nuc * c.Vuc * EField) * np.sum(chi * df['vx [m/s]']) # True at equilibirum in GaA

    corr_xx = np.load(pp.outputLoc + 'SB_Density/' +'xx_' + '2_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, EField))

    prefactor = (c.e / c.Vuc / Nuc) ** 2
    S_xx = 8*np.real(prefactor*np.sum(corr_xx*(df['vx [m/s]'])))
    if partialSum:
        print('Calculating spectral density using a cutoff.')
        energyInds = np.array(df['energy [eV]'].values < cutoff+ np.min(df['energy [eV]']))
        print(len(energyInds))
        S_xx = 8*np.real(prefactor*np.sum(corr_xx[energyInds]*(df.loc[energyInds,'vx [m/s]'])))

    return S_xx, conductivity_xx


def write_correlation(fieldVector,df,freqVector):
    """Calls the GMRES solver hard coded for solving the effective BTE w/FDM and writes the effective distribution to
    file. Calculates the longitudinal (xx) and transverse (yy) effective distribution functions.
    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        None. Just writes the chi solutions to file. chi_#_EField. #1 corresponds to low-field RTA, #2 corresponds to
        low-field iterative, #3 corresponds to full finite-difference iterative.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    for freq in freqVector:
        for ee in fieldVector:
            chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
            fdm = np.memmap(pp.inputLoc + 'finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
            f0 = df['k_FD'].values
            f = chi + f0
            Vx = utilities.mean_velocity(chi, df)
            b_xx = (-1) * ((df['vx [m/s]'] - Vx) * f)
            b_yy = (-1) * ((df['vy [m/s]']) * f)

            corr_xx, corr_xx_RTA, _, _ = occupation_solver.gaas_inverse_relaxation_operator(b_xx, scm, fdm, df, ee, freq)

            del fdm
            # This may be a problem since writing fdm twice????????????????????????
            fdm = np.memmap(pp.inputLoc + 'finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
            corr_yy, corr_yy_RTA, _, _ = occupation_solver.gaas_inverse_relaxation_operator(b_yy, scm, fdm, df, ee, freq)

            del fdm
            np.save(pp.outputLoc + 'SB_Density/' + 'xx_' + '3_' + "f_{:.1e}_E_{:.1e}".format(freq,ee),corr_xx)
            np.save(pp.outputLoc + 'SB_Density/' + 'xx_' + '1_' + "f_{:.1e}_E_{:.1e}".format(freq,ee),corr_xx_RTA)
            np.save(pp.outputLoc + 'SB_Density/' + 'yy_' + '3_' + "f_{:.1e}_E_{:.1e}".format(freq,ee),corr_yy)
            np.save(pp.outputLoc + 'SB_Density/' + 'yy_' + '1_' + "f_{:.1e}_E_{:.1e}".format(freq,ee),corr_yy_RTA)
            print('Transient solution written to file for ' + "{:.1e} V/m and {:.1e} GHz".format(ee,freq))
            print('\n \n')


def write_energy_correlation(fieldVector,df,freqVector):
    """Calls the GMRES solver hard coded for solving the effective BTE w/FDM and writes the effective distribution to
    file. Calculates the longitudinal (xx) and transverse (yy) effective distribution functions.
    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        None. Just writes the chi solutions to file. chi_#_EField. #1 corresponds to low-field RTA, #2 corresponds to
        low-field iterative, #3 corresponds to full finite-difference iterative.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    for freq in freqVector:
        for ee in fieldVector:
            chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
            fdm = np.memmap(pp.inputLoc + 'finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
            f0 = df['k_FD'].values
            f = chi + f0
            Ex = utilities.mean_energy(chi, df)
            b_xx = (-1) * ((df['energy [eV]'] - Ex)*c.e * f)
            # b_yy = (-1) * ((df['energy [eV]'])*c.e * f)

            corr_xx, corr_xx_RTA, _, _ = occupation_solver.gaas_inverse_relaxation_operator(b_xx, scm, fdm, df, ee, freq)

            del fdm
            # This may be a problem since writing fdm twice????????????????????????
            # fdm = np.memmap(pp.inputLoc + 'finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
            # corr_yy, corr_yy_RTA, _, _ = occupation_solver.gaas_inverse_relaxation_operator(b_yy, scm, fdm, df, ee, freq)

            # del fdm
            np.save(pp.outputLoc + 'E_Density/' + 'xx_' + '3_' + "f_{:.1e}_E_{:.1e}".format(freq,ee),corr_xx)
            np.save(pp.outputLoc + 'E_Density/' + 'xx_' + '1_' + "f_{:.1e}_E_{:.1e}".format(freq,ee),corr_xx_RTA)
            # np.save(pp.outputLoc + 'E_Density/' + 'yy_' + '3_' + "f_{:.1e}_E_{:.1e}".format(freq,ee),corr_yy)
            # np.save(pp.outputLoc + 'E_Density/' + 'yy_' + '1_' + "f_{:.1e}_E_{:.1e}".format(freq,ee),corr_yy_RTA)
            print('Transient solution written to file for ' + "{:.1e} V/m and {:.1e} GHz".format(ee,freq))
            print('\n \n')


def energy_density(chi, EField,df,freq, partialSum = False, cutoff = 0):
    """Calls the GMRES solver hard coded for solving the effective BTE w/FDM and writes the effective distribution to
    file. Calculates the longitudinal (xx) and transverse (yy) effective distribution functions.
    Parameters:
        chi (nparray): Containing the non-equilibrium solution to the BTE to be used for calculating the effective dist.
        EField (dbl): The value of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        freq (dbl): Frequency in GHz to be used for the transient solution
        partialSum (bool): If "True", apply an energy cutoff and only consider the contribution of states below.
        cutoff (dbl): Value of the energy cutoff in eV.
    Returns:
        None. Just writes the effective distribution solutions to file. Hardcoded for type #3 solutions. FDM + Perturbo
    TODO: Right now the "RTA" solutions are not impletmented well. They represent the problem with full FDM being solved
    using only the on-diagonal elements of the Perturbo scattering matrix. They are still using the chi solution to the
    transport BTE that uses the full Perturbo scattering matrix, so this is not implemented correctly yet. It isn't
    being plotted at the moment.
    """
    Nuc = pp.kgrid ** 3
    conductivity_xx = 2 * c.e / (Nuc * c.Vuc * EField) * np.sum(chi * df['vx [m/s]'])
    print('Conductivity is {:3f} S/m at E = {:.1f} V/cm'.format(conductivity_xx,EField/100))
    n = utilities.calculate_density(df)
    print('Carrier density is {:3e} cm^-3'.format(n/100**3))
    mobility = conductivity_xx/c.e/n
    print('Mobility is {:3f} cm^2/(V-s)'.format(mobility*100**2))

    corr_xx = np.load(pp.outputLoc + 'E_Density/' +'xx_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, EField))

    prefactor = (1 / Nuc / c.kb_joule) ** 2
    S_xx = 8*np.real(prefactor*np.sum(corr_xx*(df['energy [eV]']-np.min(df['energy [eV]']))*c.e))
    if partialSum:
        print('Calculating spectral density using a cutoff.')
        energyInds = np.array(df['energy [eV]'].values < cutoff+ np.min(df['energy [eV]']))

    return S_xx


def density(chi, EField,df,freq, partialSum = False, cutoff = 0):
    """Calls the GMRES solver hard coded for solving the effective BTE w/FDM and writes the effective distribution to
    file. Calculates the longitudinal (xx) and transverse (yy) effective distribution functions.
    Parameters:
        chi (nparray): Containing the non-equilibrium solution to the BTE to be used for calculating the effective dist.
        EField (dbl): The value of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        freq (dbl): Frequency in GHz to be used for the transient solution
        partialSum (bool): If "True", apply an energy cutoff and only consider the contribution of states below.
        cutoff (dbl): Value of the energy cutoff in eV.
    Returns:
        None. Just writes the effective distribution solutions to file. Hardcoded for type #3 solutions. FDM + Perturbo
    TODO: Right now the "RTA" solutions are not impletmented well. They represent the problem with full FDM being solved
    using only the on-diagonal elements of the Perturbo scattering matrix. They are still using the chi solution to the
    transport BTE that uses the full Perturbo scattering matrix, so this is not implemented correctly yet. It isn't
    being plotted at the moment.
    """
    Nuc = pp.kgrid ** 3
    conductivity_xx = 2 * c.e / (Nuc * c.Vuc * EField) * np.sum(chi * df['vx [m/s]'])
    print('Conductivity is {:3f} S/m at E = {:.1f} V/cm'.format(conductivity_xx,EField/100))
    n = utilities.calculate_density(df)
    print('Carrier density is {:3e} cm^-3'.format(n/100**3))
    mobility = conductivity_xx/c.e/n
    print('Mobility is {:3f} cm^2/(V-s)'.format(mobility*100**2))

    corr_yy = np.load(pp.outputLoc + 'SB_Density/'+'yy_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, EField))
    corr_xx = np.load(pp.outputLoc + 'SB_Density/' +'xx_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, EField))
    corr_xx_RTA = np.load(pp.outputLoc + 'SB_Density/' +'xx_'+ '1_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, EField))
    corr_yy_RTA = np.load(pp.outputLoc + 'SB_Density/' +'yy_'+ '1_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, EField))

    prefactor = (c.e / c.Vuc / Nuc) ** 2
    S_xx = 8*np.real(prefactor*np.sum(corr_xx*(df['vx [m/s]'])))
    S_xx_RTA = 8*np.real(prefactor*np.sum(corr_xx_RTA*(df['vx [m/s]'])))
    S_yy = 8*np.real(prefactor*np.sum(corr_yy*(df['vy [m/s]'])))
    S_yy_RTA = 8*np.real(prefactor*np.sum(corr_yy_RTA*(df['vy [m/s]'])))
    if partialSum:
        print('Calculating spectral density using a cutoff.')
        energyInds = np.array(df['energy [eV]'].values < cutoff+ np.min(df['energy [eV]']))
        S_xx = 8*np.real(prefactor*np.sum(corr_xx[energyInds]*(df.loc[energyInds,'vx [m/s]'])))
        S_yy = 8*np.real(prefactor*np.sum(corr_yy[energyInds]*(df.loc[energyInds,'vy [m/s]'])))

    return S_xx, S_xx_RTA, S_yy, S_yy_RTA, conductivity_xx


def plot_density(fieldVector,freqVector,df):
    """Makes plots of the spectral density as a function of frequency and field. Right now hard coded to return the plots
    of only the type #3 solutions using Perturbo SCM + FDM."""
    for freq in freqVector:
        S_xx_vector = []
        S_xx_RTA_vector = []
        S_yy_vector = []
        conductivity_xx_vector = []
        for ee in fieldVector:
            chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
            S_xx, S_xx_RTA, S_yy, S_yy_RTA, conductivity_xx = density(chi,ee,df,freq)
            S_xx_vector.append(S_xx)
            S_xx_RTA_vector.append(S_xx_RTA)
            S_yy_vector.append(S_yy)
            conductivity_xx_vector.append(conductivity_xx)
            # conductivity_yy_vector.append(conductivity_yy)
            kvcm = np.array(fieldVector) * 1e-5

            Nuc = pp.kgrid ** 3
        fig, ax = plt.subplots()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = '\n'.join((r'$f = %.1f GHz \, \, (100) $' % (freq,), pp.fdmName))
        ax.plot(kvcm, S_xx_vector, label=r'$S^{xx}$')
        ax.plot(kvcm, S_yy_vector, label=r'$S^{yy}$')
        ax.axhline(np.array(conductivity_xx_vector[0]) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',
                    label=r'$\frac{4 kT \sigma_{xx}^{eq}}{V_0 N_0}$')
        plt.legend()
        plt.xlabel('Field [kV/cm]')
        plt.ylabel('Spectral Density [A^2/m^4/Hz]')
        plt.title(pp.title_str)
        ax.text(0.55, 0.9, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

    S_xx_vector = []
    S_xx_RTA_vector = []
    S_yy_vector = []
    for freq in freqVector:
        plotfield = fieldVector[-1]
        chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(plotfield))
        S_xx, S_xx_RTA, S_yy, S_yy_RTA, conductivity_xx = density(chi, plotfield, df, freq)
        S_xx_vector.append(S_xx)
        S_xx_RTA_vector.append(S_xx_RTA)
        S_yy_vector.append(S_yy)
        Nuc = pp.kgrid ** 3
        print('freq')
    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((pp.fdmName,r'$E = %.1f kV/cm  \, \, (100)$' % (plotfield/1e5,)))
    ax.plot(freqVector, S_xx_vector, label=r'$S^{xx}$')
    ax.plot(freqVector, S_yy_vector, label=r'$S^{yy}$')
    ax.axhline(np.array(conductivity_xx_vector[0]) * 4 * c.kb_joule * pp.T / c.Vuc / Nuc, linestyle='--',
                    label=r'$\frac{4 kT \sigma_{xx}^{eq}}{V_0 N_0}$')
    plt.legend()
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Spectral Density [A^2/m^4/Hz]')
    plt.title(pp.title_str)
    ax.text(0.05, 0.15, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    plt.xscale('log')


def plot_energy_density(fieldVector,freqVector,df):
    """Makes plots of the spectral density as a function of frequency and field. Right now hard coded to return the plots
    of only the type #3 solutions using Perturbo SCM + FDM."""
    fig, ax = plt.subplots()
    colorList = ['black', 'dodgerblue', 'tomato']
    i = 0
    for ee in fieldVector:
        S_xx_vector = []
        for freq in freqVector:
            chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
            S_xx = energy_density(chi, ee, df, freq)
            S_xx_vector.append(S_xx)
        S_xx_vector = np.array(S_xx_vector)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.plot(freqVector, S_xx_vector, label=r'{:.1f} '.format(ee/100) + r'$\rm V \, cm^{-1}$',color=colorList[i])
        i = i + 1

    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Temperature fluctuation PSD ' + r'$ \rm (K^2 \, Hz^{-1})$')
    plt.xscale('log')
    plt.savefig(pp.figureLoc +'temperature_PSD.png', bbox_inches='tight',dpi=600)



if __name__ == '__main__':
    # Create electron and phonon dataframes

    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)

    fields = pp.small_signal_fields
    freqs = pp.freqVector

    # fields = pp.moment_fields
    # freqs = np.array([0.1])


    # write_correlation(fields,electron_df,freqs)
    # write_correlation(pp.moment_fields, electron_df, np.array([0.1]))
    # plot_density(fields, freqs, electron_df)

    # write_energy_correlation(fields,electron_df,freqs)
    # plot_energy_density(fields, freqs, electron_df)

    plt.show()
