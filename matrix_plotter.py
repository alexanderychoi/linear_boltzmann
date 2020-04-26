import numpy as np
import pandas as pd
import time
import constants as c
import utilities
import problemparameters as pp
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy.linalg
from scipy.sparse import linalg
import noise_solver

def velocity_distribution_kde(chi, df, title=[]):
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
    # spread = 22 * dx # For 200^3
    spread = 25 * dx  # For 160^3

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
    ax.plot(v_ax, [0] * len(v_ax), 'k')
    ax.plot(v_ax, vdist_f0, '--', linewidth=2, label='Equilbrium')
    ax.plot(v_ax, vdist_tot, linewidth=2, label='Hot electron distribution')
    # plt.fill(v_ax, vdist, label='non-eq distr', color='red')
    ax.fill(v_ax, vdist, '--', linewidth=2, label='Non-equilibrium deviation', color='Red')
    ax.set_xlabel(r'Velocity [ms$^{-1}$]')
    ax.set_ylabel(r'Occupation [arb.]')
    plt.legend()
    if title:
        plt.title(title, fontsize=8)


def plot_vel_KDEs(outLoc, field, df, plotRTA=True, plotLowField=True, plotFDM=True):
    """Wrapper script for velocity_distribution_kde. Can do for the various solution schemes saved to file.
        Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions and ready steady state chis.
        field (dbl): the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        Nothing. Just the plots.
    """
    if plotRTA:
        f_i = np.load(outLoc + 'f_1_gmres.npy')
        chi_1_i = utilities.f2chi(f_i, df, field)
        velocity_distribution_kde(chi_1_i, df, title='RTA Chi {:.1e} V/m '.format(field) + pp.title_str)
        ng_1, nl_1, g_inds_1, l_inds_1 = utilities.calc_L_Gamma_pop(chi_1_i, df)
    if plotLowField:
        f_i = np.load(outLoc + 'f_2_gmres.npy')
        chi_2_i = utilities.f2chi(f_i, df, field)
        velocity_distribution_kde(chi_2_i, df,
                                  title='Low Field Iterative Chi {:.1e} V/m '.format(field) + pp.title_str)
        ng_2, nl_2, g_inds_2, l_inds_2 = utilities.calc_L_Gamma_pop(chi_1_i, df)
    if plotFDM:
        chi_3_i = np.load(outLoc + 'chi_3_gmres_{:.1e}.npy'.format(field))
        velocity_distribution_kde(chi_3_i, df, title='FDM Iterative Chi {:.1e} V/m '.format(field) + pp.title_str)
        ng_3, nl_3, g_inds_3, l_inds_3 = utilities.calc_L_Gamma_pop(chi_1_i, df)

    if plotRTA:
        velocity_distribution_kde(chi_1_i[g_inds_1], df.loc[g_inds_1 + 1].reset_index(drop=True),
                                  title='Gamma RTA Chi {:.1e} V/m '.format(field) + pp.title_str)
        plt.xlabel('x-Velocity [m/s]')
        plt.ylabel('Occupation [arb]')
    if plotFDM:
        velocity_distribution_kde(chi_2_i[g_inds_2], df.loc[g_inds_2 + 1].reset_index(drop=True),
                                  title='Gamma Low Field Iterative Chi {:.1e} V/m '.format(field) + pp.title_str)
        plt.xlabel('x-Velocity [m/s]')
        plt.ylabel('Occupation [arb]')
    if plotFDM:
        velocity_distribution_kde(chi_3_i[g_inds_3], df.loc[g_inds_3 + 1].reset_index(drop=True),
                                  title='Gamma FDM Iterative Chi {:.1e} V/m '.format(field) + pp.title_str)
        plt.xlabel('x-Velocity [m/s]')
        plt.ylabel('Occupation [arb]')

    plt.figure()
    plt.plot(df['vx [m/s]'].values, chi_3_i + df['k_FD'].values)


def driftvel_mobility_vs_field(outLoc, df, fieldVector, plotRTA=True, plotLowField=True, plotFDM=True):
    """Takes chi solutions which are already calculated and plots drift velocity vs field
    Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions and ready steady state chis.
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        Nothing. Just the plots.
    """
    if plotRTA:
        vd_1, meanE_1, n_1, mu_1, ng_1, nl_1 = ([] for i in range(6))
    if plotLowField:
        vd_2, meanE_2, n_2, mu_2, ng_2, nl_2 = ([] for i in range(6))
    if plotFDM:
        vd_3, meanE_3, n_3, mu_3, ng_3, nl_3 = ([] for i in range(6))

    for ee in fieldVector:
        if plotRTA:
            f_i = np.load(outLoc + 'f_1_gmres.npy')
            chi_1_i = utilities.f2chi(f_i, df, ee)
            vd_1.append(utilities.drift_velocity(chi_1_i, df))
            meanE_1.append(utilities.mean_energy(chi_1_i, df))
            n_1.append(utilities.calculate_noneq_density(chi_1_i, df))
            mu_1.append(utilities.calc_mobility(f_i, df) * 10 ** 4)
            ng_i, nl_i, _, _ = utilities.calc_L_Gamma_pop(chi_1_i, df)
            ng_1.append(ng_i)
            nl_1.append(nl_i)
        if plotLowField:
            f_i = np.load(outLoc + 'f_2_gmres.npy')
            chi_2_i = utilities.f2chi(f_i, df, ee)
            vd_2.append(utilities.drift_velocity(chi_2_i, df))
            meanE_2.append(utilities.mean_energy(chi_2_i, df))
            n_2.append(utilities.calculate_noneq_density(chi_2_i, df))
            mu_2.append(utilities.calc_mobility(f_i, df) * 10 ** 4)
            ng_i, nl_i, _, _ = utilities.calc_L_Gamma_pop(chi_2_i, df)
            ng_2.append(ng_i)
            nl_2.append(nl_i)
        if plotFDM:
            chi_3_i = np.load(outLoc + 'chi_3_gmres_{:.1e}.npy'.format(ee))
            vd_3.append(utilities.drift_velocity(chi_3_i, df))
            meanE_3.append(utilities.mean_energy(chi_3_i, df))
            n_3.append(utilities.calculate_noneq_density(chi_3_i, df))
            mu_3.append(utilities.calc_diff_mobility(chi_3_i, df, ee) * 10 ** 4)
            ng_i, nl_i, _, _ = utilities.calc_L_Gamma_pop(chi_3_i, df)
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
    plt.title(pp.title_str)
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
    plt.ylim([0,1.1*np.max(n_1)])
    plt.title(pp.title_str)
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
    plt.title(pp.title_str)
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
    plt.title(pp.title_str)
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
    plt.title(pp.title_str)
    plt.legend()


def plot_scattering_rates(inLoc, df, applyscmFac=False, simplelin=True):
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
        scmfac = pp.scmVal
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1
    nkpts = len(df)
    scm = np.memmap(inLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    if simplelin:
        rates = (-1) * np.diag(scm) * scmfac * 1E-12
        # inds = np.where(np.asarray(rates==0))
        # print(inds)
    else:
        chi2psi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
        rates = (-1) * np.diag(scm) * scmfac * 1E-12 / chi2psi
    # par_RT = np.load(inLoc+'scattering_rates.npy')
    plt.figure()
    plt.plot(df['energy'], rates, '.', MarkerSize=3, label='Simple Diagonal')
    # plt.plot(df['energy'],par_RT*scmfac, '.', MarkerSize=1,label='Par. RTs')
    plt.xlabel('Energy [eV]')
    plt.ylabel(r'Scattering rate [ps$^{-1}$]')
    plt.title(pp.title_str)


def plot_scattering_rates_pop(inLoc,chi, df, applyscmFac=False, simplelin=True):
    """Takes chi solutions which are already calculated and plots drift velocity vs field
    Parameters:
        inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
        linearization by default.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        applyscmFac (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.

    Returns:
        Nothing. Just the plots.
    """
    df['chi+f0'] = df['k_FD'].values + chi
    if applyscmFac:
        scmfac = pp.scmVal
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1
    nkpts = len(df)
    scm = np.memmap(inLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    if simplelin:
        rates = (-1) * np.diag(scm) * scmfac * 1E-12
        # inds = np.where(np.asarray(rates==0))
        # print(inds)
    else:
        chi2psi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
        rates = (-1) * np.diag(scm) * scmfac * 1E-12 / chi2psi
    # par_RT = np.load(inLoc+'scattering_rates.npy')
    plt.figure()
    plt.plot(df['energy'], rates, '.', MarkerSize=3, label='Simple Diagonal', )
    # plt.plot(df['energy'],par_RT*scmfac, '.', MarkerSize=1,label='Par. RTs')
    plt.xlabel('Energy [eV]')
    plt.ylabel(r'Scattering rate [ps$^{-1}$]')
    plt.title(pp.title_str)


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
            f_i = np.load(outLoc + 'f_1_gmres.npy')
            mu_1 = utilities.calc_mobility(f_i,df)
            Tn_1.append(noise_solver.noiseT(in_Loc,noise_1[-1], mu_1, df,applySCMFac))

        if plotFDM:
            g_3_i = np.load(outLoc + 'g_3_gmres_{:.1e}.npy'.format(ee))
            noise_3.append(noise_solver.lowfreq_diffusion(g_3_i, df))
            chi_3_i = np.load(outLoc + 'chi_3_gmres_{:.1e}.npy'.format(ee))
            mu_3 = (utilities.calc_diff_mobility(chi_3_i,df,ee))
            Tn_3.append(noise_solver.noiseT(in_Loc,noise_3[-1], mu_3, df,applySCMFac))
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
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    if plotRTA:
        plt.plot(kvcm, Tn_1, linewidth=2, label='RTA')

    if plotFDM:
        plt.plot(kvcm, Tn_3, linewidth=2, label='FDM')

    plt.xlabel('Field [kV/cm]')
    plt.ylabel(r'Noise Temperature [K]')
    plt.title(pp.title_str)
    plt.legend()


def occupation_v_energy_sep(chi, enk, kptsdf):
    kptsdf['kpt_mag'] = np.sqrt(kptsdf['kx [1/A]'].values**2 + kptsdf['ky [1/A]'].values**2 +
                                 kptsdf['kz [1/A]'].values**2)
    kptsdf['ingamma'] = kptsdf['kpt_mag'] < 0.3  # Boolean. In gamma if kpoint magnitude less than some amount
    vmags = kptsdf['v_mag [m/s]']

    npts = 1000  # number of points in the KDE
    g_chiax = np.zeros(npts)  # capital sigma as defined in Jin Jian's paper Eqn 3
    g_ftot = np.zeros(npts)
    g_f0ax = np.zeros(npts)

    l_chiax = np.zeros(npts)  # capital sigma as defined in Jin Jian's paper Eqn 3
    l_ftot = np.zeros(npts)
    l_f0ax = np.zeros(npts)

    # Need to define the energy range that I'm doing integration over
    # en_axis = np.linspace(enk.min(), enk.min() + 0.4, npts)
    g_en_axis = np.linspace(enk.min(), enk.max(), npts)
    l_en_axis = np.linspace(enk.min(), enk.max(), npts)

    g_inds = kptsdf.loc[kptsdf['ingamma'] == 1].index - 1
    l_inds = kptsdf.loc[kptsdf['ingamma'] == 0].index - 1

    g_chi = chi[g_inds]
    l_chi = chi[l_inds]
    g_enk = enk[g_inds]
    l_enk = enk[l_inds]

    dx = (g_en_axis.max() - g_en_axis.min()) / npts
    g_f0 = np.squeeze(kptsdf.loc[kptsdf['ingamma'] == 1,'k_FD'].values)
    l_f0 = np.squeeze(kptsdf.loc[kptsdf['ingamma'] == 0,'k_FD'].values)
    # spread = 50 * dx
    spread = 90 * dx
    # def gaussian(x, mu, sigma=spread):
    #     return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)

    def gaussian(x, mu, vmag, stdev=spread):
        sigma = stdev - (vmag/1E6) * 0.9 * stdev
        vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
        return vals

    for k in range(len(g_inds)):
        istart = int(np.maximum(np.floor((g_enk[k] - g_en_axis[0]) / dx) - (4*spread/dx), 0))
        iend = int(np.minimum(np.floor((g_enk[k] - g_en_axis[0]) / dx) + (4*spread/dx), npts - 1))
        g_ftot[istart:iend] += (g_chi[k] + g_f0[k]) * gaussian(g_en_axis[istart:iend], g_enk[k],vmags[k])
        g_f0ax[istart:iend] += g_f0[k] * gaussian(g_en_axis[istart:iend], g_enk[k], vmags[k])
        g_chiax[istart:iend] += g_chi[k] * gaussian(g_en_axis[istart:iend], g_enk[k], vmags[k])

    for k in range(len(l_inds)):
        istart = int(np.maximum(np.floor((l_enk[k] - l_en_axis[0]) / dx) - (4 * spread / dx), 0))
        iend = int(np.minimum(np.floor((l_enk[k] - l_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
        l_ftot[istart:iend] += (l_chi[k] + l_f0[k]) * gaussian(l_en_axis[istart:iend], l_enk[k], vmags[k])
        l_f0ax[istart:iend] += l_f0[k] * gaussian(l_en_axis[istart:iend], l_enk[k], vmags[k])
        l_chiax[istart:iend] += l_chi[k] * gaussian(l_en_axis[istart:iend], l_enk[k], vmags[k])

    return g_en_axis, g_ftot, g_chiax, g_f0ax, l_en_axis, l_ftot, l_chiax, l_f0ax

def plot_energy_sep(outLoc,df,fields):
    plt.figure(figsize=(6, 6))
    ax = plt.axes([0.19, 0.2, 0.75, 0.73])
    for ee in fields:
        chi_fullfield = chi_3_i = np.load(outLoc + 'chi_3_gmres_{:.1e}.npy'.format(ee))
        g_en_axis, g_ftot, g_chiax, g_f0ax, l_en_axis, l_ftot, l_chiax, l_f0ax = occupation_v_energy_sep(
            chi_fullfield, df['energy'].values, df)
        plt.plot(g_en_axis, g_chiax, label=r'$\Gamma$ Valley')
        plt.plot(l_en_axis, l_chiax, '--', label='L Valley')
        plt.xlabel('Energy above CBM (eV)')
        # plt.xlim([0, 0.475])
        plt.yticks([])
        plt.ylabel(r'Deviational occupation ($\chi_{\mathbf{k}}$)')
        plt.legend()


if __name__ == '__main__':
    out_Loc = pp.outputLoc
    in_Loc = pp.inputLoc

    # Read problem parameters and specify electron DataFrame
    utilities.load_electron_df(in_Loc)
    utilities.read_problem_params(in_Loc)
    electron_df = pd.read_pickle(in_Loc + 'electron_df.pkl')
    electron_df = utilities.fermi_distribution(electron_df)

    # Steady state solutions
    fields = np.array([1e2, 1e3, 1e4, 2.5e4, 5e4, 7.5e4, 1e5, 2e5, 3e5])
    # fields = np.array([3.5e5,4e5,4.5e5,5e5,5.5e5,6e5])
    applySCMFac = pp.scmBool
    simpleLin = pp.simpleBool

    KDEField = fields[-1]
    plotTransport = False
    plotKDE = False
    plotScattering = False
    plotNoise = False
    plots_vs_energy_separate_gamma_and_l = True

    if plotTransport:
        driftvel_mobility_vs_field(out_Loc, electron_df, fields)
    if plotKDE:
        plot_vel_KDEs(out_Loc, KDEField, electron_df, plotRTA=True, plotLowField=True, plotFDM=True)
    if plotScattering:
        plot_scattering_rates(in_Loc, electron_df, applySCMFac, simpleLin)
    if plotNoise:
        plot_noise(out_Loc, fields, electron_df, plotRTA=True, plotFDM=True)
    if plots_vs_energy_separate_gamma_and_l:
        plot_energy_sep(out_Loc, electron_df, fields)

    plt.show()

