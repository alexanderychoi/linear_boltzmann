#!/usr/bin/python3
import utilities
import noise_solver
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import problemparameters as pp

def velocity_distribution_kde(chi, df,title=[]):
    """Takes chi solutions which are already calculated and plots the KDE of the distribution in velocity space
    Parameters:
        chi (nparray): Numpy array containing a solution of the steady Boltzmann equation in chi form.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        title (str): String containing the desired name of the plot
    Returns:
        Nothing. Just the plots.
    """
    vel = df['vx [m/s]']
    npts = 600  # number of points in the KDE
    vdist = np.zeros(npts)
    vdist_tot = np.zeros(npts)
    vdist_f0 = np.zeros(npts)
    # Need to define the energy range that I'm doing integration over
    # en_axis = np.linspace(enk.min(), enk.min() + 0.4, npts)
    v_ax = np.linspace(vel.min(), vel.max(), npts)
    dx = (v_ax.max() - v_ax.min()) / npts
    f0 = np.squeeze(df['k_FD'].values)
    spread = 22 * dx

    def gaussian(x, mu, sigma=spread):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
    for k in range(len(chi)):
        istart = int(np.maximum(np.floor((vel[k] - v_ax[0]) / dx) - (4 * spread / dx), 0))
        iend = int(np.minimum(np.floor((vel[k] - v_ax[0]) / dx) + (4 * spread / dx), npts - 1))
        vdist_tot[istart:iend] += (chi[k] + f0[k]) * gaussian(v_ax[istart:iend], vel[k])
        vdist_f0[istart:iend] += f0[k] * gaussian(v_ax[istart:iend], vel[k])
        vdist[istart:iend] += chi[k] * gaussian(v_ax[istart:iend], vel[k])

    plt.figure()
    ax = plt.axes([0.18, 0.15, 0.76, 0.76])
    ax.plot(v_ax, [0]*len(v_ax), 'k')
    ax.plot(v_ax, vdist_f0, '--', linewidth=2, label='Equilbrium')
    ax.plot(v_ax, vdist_tot, linewidth=2, label='Hot electron distribution')
    # plt.fill(v_ax, vdist, label='non-eq distr', color='red')
    ax.fill(v_ax, vdist, '--', linewidth=2, label='Non-equilibrium deviation', color='C1')
    ax.set_xlabel(r'Velocity [ms$^{-1}$]')
    ax.set_ylabel(r'Occupation [arb.]')
    plt.legend()
    if title:
        plt.title(title)


def plot_vel_KDEs(outLoc,field,df,plotRTA=True,plotLowField=True,plotFDM=True):
    """Wrapper script for velocity_distribution_kde. Can do for the various solution schemes saved to file.
        Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions and ready steady state chis.
        field (dbl): the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        Nothing. Just the plots.
    """
    if plotRTA:
        f_i = np.load(outLoc + 'f_1.npy')
        chi_1_i = utilities.f2chi(f_i,df,field)
        velocity_distribution_kde(chi_1_i, df, title='RTA Chi {:.1e} V/m'.format(field))
    if plotLowField:
        f_i = np.load(outLoc + 'f_2.npy')
        chi_2_i = utilities.f2chi(f_i,df,field)
        velocity_distribution_kde(chi_2_i, df, title='Low Field Iterative Chi {:.1e} V/m'.format(field))
    if plotFDM:
        chi_3_i = np.load(outLoc + 'chi_3_{:.1e}.npy'.format(field))
        velocity_distribution_kde(chi_3_i, df, title='FDM Iterative Chi {:.1e} V/m'.format(field))


def plot_noise(outLoc,fieldVector,df,plotRTA=True,plotFDM=True):
    """Wrapper script for noise_power. Can do for the various solution schemes saved to file.
        Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions and ready steady state chis.
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        Nothing. Just the plots.
    """
    noise_1 = []
    noise_3 =[]
    Tn_1 = []
    Tn_3 = []

    for ee in fieldVector:
        if plotRTA:
            g_1_i = np.load(outLoc + 'g_1_{:.1e}.npy'.format(ee))
            noise_1.append(noise_solver.lowfreq_diffusion(g_1_i, df))
            f_i = np.load(outLoc + 'f_1.npy')
            mu_1 = utilities.calc_mobility(f_i,df)
            Tn_1.append(noise_solver.noiseT(in_Loc,noise_1[-1], mu_1, df))

        if plotFDM:
            g_3_i = np.load(outLoc + 'g_3_{:.1e}.npy'.format(ee))
            noise_3.append(noise_solver.lowfreq_diffusion(g_3_i, df))
            chi_3_i = np.load(outLoc + 'chi_3_{:.1e}.npy'.format(ee))
            mu_3 = (utilities.calc_diff_mobility(chi_3_i,df,ee))
            Tn_3.append(noise_solver.noiseT(in_Loc,noise_3[-1], mu_3, df))
    kvcm = np.array(fieldVector) * 1E-5
    plt.figure()
    if plotRTA:
        g_1_johnson = np.load(outLoc + 'g_1_johnson.npy')
        plt.plot(kvcm, noise_1, linewidth=2, label='RTA')
        plt.axhline(noise_solver.lowfreq_diffusion(g_1_johnson,df), color = 'black',linestyle='--',label='RTA Johnson')

    if plotFDM:
        plt.plot(kvcm, noise_3, linewidth=2, label='FDM')
        plt.axhline(noise_3[0], color = 'black',linestyle='--',label='FDM Johnson')

    plt.xlabel('Field [kV/cm]')
    plt.ylabel(r'Non-equilibrium diffusion coefficient [m^2/s]')
    plt.legend()

    plt.figure()
    if plotRTA:
        plt.plot(kvcm, Tn_1, linewidth=2, label='RTA')

    if plotFDM:
        plt.plot(kvcm, Tn_3, linewidth=2, label='FDM')

    plt.xlabel('Field [kV/cm]')
    plt.ylabel(r'Noise Temperature [K]')
    plt.legend()


def driftvel_mobility_vs_field(outLoc,df,fieldVector,plotRTA=True,plotLowField=True,plotFDM=True):
    """Takes chi solutions which are already calculated and plots drift velocity vs field
    Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions and ready steady state chis.
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        Nothing. Just the plots.
    """
    if plotRTA:
        vd_1,meanE_1,n_1,mu_1,ng_1,nl_1 = ([] for i in range(6))
    if plotLowField:
        vd_2,meanE_2,n_2,mu_2,ng_2,nl_2 = ([] for i in range(6))
    if plotFDM:
        vd_3,meanE_3,n_3,mu_3,ng_3,nl_3 = ([] for i in range(6))

    for ee in fieldVector:
        if plotRTA:
            f_i = np.load(outLoc + 'f_1.npy')
            chi_1_i = utilities.f2chi(f_i, df, ee)
            vd_1.append(utilities.drift_velocity(chi_1_i,df))
            meanE_1.append(utilities.mean_energy(chi_1_i,df))
            n_1.append(utilities.calculate_noneq_density(chi_1_i,df))
            mu_1.append(utilities.calc_mobility(f_i,df)*10**4)
            ng_i,nl_i=utilities.calc_L_Gamma_pop(chi_1_i,df)
            ng_1.append(ng_i)
            nl_1.append(nl_i)
        if plotLowField:
            f_i = np.load(outLoc + 'f_2.npy')
            chi_2_i = utilities.f2chi(f_i, df, ee)
            vd_2.append(utilities.drift_velocity(chi_2_i,df))
            meanE_2.append(utilities.mean_energy(chi_2_i,df))
            n_2.append(utilities.calculate_noneq_density(chi_2_i,df))
            mu_2.append(utilities.calc_mobility(f_i,df)*10**4)
            ng_i,nl_i=utilities.calc_L_Gamma_pop(chi_2_i,df)
            ng_2.append(ng_i)
            nl_2.append(nl_i)
        if plotFDM:
            chi_3_i = np.load(outLoc + 'chi_3_{:.1e}.npy'.format(ee))
            vd_3.append(utilities.drift_velocity(chi_3_i,df))
            meanE_3.append(utilities.mean_energy(chi_3_i,df))
            n_3.append(utilities.calculate_noneq_density(chi_3_i,df))
            mu_3.append(utilities.calc_diff_mobility(chi_3_i,df,ee)*10**4)
            ng_i,nl_i=utilities.calc_L_Gamma_pop(chi_3_i,df)
            ng_3.append(ng_i)
            nl_3.append(nl_i)
    kvcm = np.array(fieldVector) * 1E-5
    plt.figure()
    if plotRTA:
        plt.plot(kvcm, vd_1, 'o-', linewidth=2, label='RTA')
    if plotLowField:
        plt.plot(kvcm, vd_2, linewidth=2, label='Low Field Iterative')
    if plotFDM:
        plt.plot(kvcm, vd_3, linewidth=2, label='FDM Iterative')
    plt.xlabel('Field [kV/cm]')
    plt.ylabel(r'Drift velocity [m/s]')
    plt.legend()

    plt.figure()
    if plotRTA:
        plt.plot(kvcm, n_1, 'o-', linewidth=2, label='RTA')
    if plotLowField:
        plt.plot(kvcm, n_2, linewidth=2, label='Low Field Iterative')
    if plotFDM:
        plt.plot(kvcm, n_3, linewidth=2, label='FDM Iterative')
    plt.xlabel('Field [kV/cm]')
    plt.ylabel(r'Carrier population [m^-3]')
    plt.legend()

    plt.figure()
    if plotRTA:
        plt.plot(kvcm, meanE_1, 'o-', linewidth=2, label='RTA')
    if plotLowField:
        plt.plot(kvcm, meanE_2, linewidth=2, label='Low Field Iterative')
    if plotFDM:
        plt.plot(kvcm, meanE_3, linewidth=2, label='FDM Iterative')
    plt.xlabel('Field [kV/cm]')
    plt.ylabel(r'Mean energy [eV]')
    plt.legend()

    plt.figure()
    if plotRTA:
        plt.plot(kvcm, mu_1, 'o-', linewidth=2, label='RTA')
    if plotLowField:
        plt.plot(kvcm, mu_2, linewidth=2, label='Low Field Iterative')
    if plotFDM:
        plt.plot(kvcm, mu_3, linewidth=2, label='FDM Iterative')
    plt.xlabel('Field [kV/cm]')
    plt.ylabel(r'Mobility [$cm^2 V^{-1} s^{-1}$]')
    plt.legend()

    plt.figure()
    if plotRTA:
        plt.plot(kvcm, ng_1, 'o-', linewidth=2, label='RTA Gamma')
        plt.plot(kvcm, nl_1, 'o-', linewidth=2, label='RTA L')
    if plotLowField:
        plt.plot(kvcm, ng_2, linewidth=2, label='Low Field Iterative Gamma')
        plt.plot(kvcm, nl_2, linewidth=2, label='Low Field Iterative L')
    if plotLowField:
        plt.plot(kvcm, ng_3, linewidth=2, label='FDM Iterative Gamma')
        plt.plot(kvcm, nl_3, linewidth=2, label='FDM Iterative L')
    plt.xlabel('Field [kV/cm]')
    plt.ylabel(r'Carrier Population [m^-3]$]')
    plt.legend()


def plot_scattering_rates(inLoc,df,applyscmFac=False):
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
    scm = np.memmap(inLoc + 'scattering_matrix_5.87_simple.mmap', dtype='float64', mode='r', shape=(42433, 42433))
    rates = (-1) * np.diag(scm) * scmfac * 1E-12
    plt.figure()
    plt.plot(df['energy'], rates, '.', MarkerSize=3)
    plt.xlabel('Energy [eV]')
    plt.ylabel(r'Scattering rate [ps$^{-1}$]')


if __name__ == '__main__':
    # Points to inputs and outputs
    out_Loc = pp.outputLoc
    in_Loc = pp.inputLoc

    # Read problem parameters and specify electron DataFrame
    utilities.read_problem_params(in_Loc)
    electron_df = pd.read_pickle(in_Loc+'electron_df.pkl')
    electron_df = utilities.fermi_distribution(electron_df)

    # Specify fields to plot over
    fields = np.array([1e1,2e1,3e1,4e1,5e1,6e1,7e1,8e1,9e1,1e2,2e2,3e2,4e2,5e2,6e2,7e2,8e2,9e2,2e3,4e3,6e3,8e3,1e4,2e4,4e4,6e4,8e4,1.1e5,1.2e5,1.3e5,1.4e5,1.5e5,1.6e5,1.7e5,1.8e5,1.9e5,2e5])
    KDEField = 1.8e5
    plotTransport = True
    plotKDE = True
    plotScattering = True
    plotNoise = True
    applySCMFac = pp.scmBool

    if plotTransport:
        driftvel_mobility_vs_field(out_Loc, electron_df, fields)
    if plotKDE:
        plot_vel_KDEs(out_Loc, KDEField, electron_df, plotRTA=True, plotLowField=True, plotFDM=True)
    if plotScattering:
        plot_scattering_rates(in_Loc, electron_df, applySCMFac)
    if plotNoise:
        plot_noise(out_Loc, fields, electron_df, plotRTA=True, plotFDM=True)
    plt.show()

