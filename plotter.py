#!/usr/bin/python3
import utilities
# import noise_solver
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import problem_parameters as pp
import constants as c

import matplotlib
font = {'size'   : 12}
matplotlib.rc('font', **font)

import plotly.offline as py
import plotly.graph_objs as go
import plotly


def bz_3dscatter(df):
    if np.any(df['energy [eV]']):
        colors = df['energy [eV]']
    else:
        colors = 'k'
    trace1 = go.Scatter3d(
        x=df['kx [1/A]'].values / (2 * np.pi / c.a),
        y=df['ky [1/A]'].values / (2 * np.pi / c.a),
        z=df['kz [1/A]'].values / (2 * np.pi / c.a),
        mode='markers',
        marker=dict(size=2, color=colors, colorscale='Rainbow', showscale=True, opacity=1)
    )

    b1edge = 0.5 * c.b1 / (2 * np.pi / c.a)
    vector1 = go.Scatter3d(x=[0, b1edge[0]], y=[0, b1edge[1]], z=[0, b1edge[2]],
                           marker=dict(size=1,color="rgb(84,48,5)"),
                           line=dict(color="rgb(84,48,5)", width=5))
    b2edge = 0.5 * c.b2 / (2 * np.pi / c.a)
    vector2 = go.Scatter3d(x=[0, b2edge[0]], y=[0, b2edge[1]], z=[0, b2edge[2]],
                           marker=dict(size=1,color="rgb(84,48,5)"),
                           line=dict(color="rgb(84,48,5)", width=5))
    b3edge = 0.5 * c.b3 / (2 * np.pi / c.a)
    vector3 = go.Scatter3d(x=[0, b3edge[0]], y=[0, b3edge[1]], z=[0, b3edge[2]],
                           marker=dict(size=1,color="rgb(84,48,5)"),
                           line=dict(color="rgb(84,48,5)", width=5))
    xedge = -0.5 * (c.b1 + c.b3) / (2 * np.pi / c.a)
    vector4 = go.Scatter3d(x=[0, xedge[0]], y=[0, xedge[1]], z=[0, xedge[2]],
                           marker=dict(size=1, color="rgb(84,48,5)"),
                           line=dict(color="rgb(84,48,5)", width=5))
    yedge = 0.5 * (c.b2 + c.b3) / (2 * np.pi / c.a)
    vector5 = go.Scatter3d(x=[0, yedge[0]], y=[0, yedge[1]], z=[0, yedge[2]],
                           marker=dict(size=1, color="rgb(84,48,5)"),
                           line=dict(color="rgb(84,48,5)", width=5))
    zedge = 0.5 * (c.b1 + c.b2) / (2 * np.pi / c.a)
    vector6 = go.Scatter3d(x=[0, zedge[0]], y=[0, zedge[1]], z=[0, zedge[2]],
                           marker=dict(size=1, color="rgb(84,48,5)"),
                           line=dict(color="rgb(84,48,5)", width=5))
    ledge = 0.5 * (c.b1 + c.b2 + c.b3) / (2 * np.pi / c.a)
    vector7 = go.Scatter3d(x=[0, ledge[0]], y=[0, ledge[1]], z=[0, ledge[2]],
                           marker=dict(size=1, color="rgb(84,48,5)"),
                           line=dict(color="rgb(84,48,5)", width=5))

    data = [trace1, vector1, vector2, vector3, vector4, vector5, vector6, vector7]
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title='kx', titlefont=dict(family='Oswald, monospace', size=18)),
            yaxis=dict(
                title='ky', titlefont=dict(family='Oswald, monospace', size=18)),
            zaxis=dict(
                title='kz', titlefont=dict(family='Oswald, monospace', size=18)), ))
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='bz_scatter.html')
    return fig


def occupation_v_energy(field, outloc, el_df):
    chi = np.load(outloc + 'chi_3_gmres_{:.1e}.npy'.format(field))
    npts = 1000  # number of points in the KDE
    chiax = np.zeros(npts)  # capital sigma as defined in Jin Jian's paper Eqn 3
    ftot = np.zeros(npts)
    f0ax = np.zeros(npts)
    # Need to define the energy range that I'm doing integration over
    # en_axis = np.linspace(enk.min(), enk.min() + 0.4, npts)
    enk = el_df['energy [eV]'] - el_df['energy [eV]'].min()
    en_axis = np.linspace(enk.min(), enk.max(), npts)
    dx = (en_axis.max() - en_axis.min()) / npts
    f0 = np.squeeze(el_df['k_FD'])
    vmags = el_df['v_mag [m/s]']
    spread = 46 * dx

    # def gaussian(x, mu, sigma=spread):
    #     return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)

    def gaussian(x, mu, vmag, stdev=spread):
        sigma = stdev - (vmag/1E6) * 0.9 * stdev
        vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
        return vals

    for k in range(len(enk)):
        istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4*spread/dx), 0))
        iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4*spread/dx), npts - 1))
        ftot[istart:iend] += (chi[k] + f0[k]) * gaussian(en_axis[istart:iend], enk[k], vmags[k])
        f0ax[istart:iend] += f0[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k])
        chiax[istart:iend] += chi[k] * gaussian(en_axis[istart:iend], enk[k], vmags[k])

    ftot = f0ax + chiax

    plt.figure(figsize=(5, 4.8))
    ax = plt.axes([0.22, 0.15, 0.73, 0.73])
    plt.plot(en_axis, f0ax, '--', linewidth=1.5, label='Equilibrium (FD)', color='C1')
    plt.plot(en_axis, ftot, '-', linewidth=1.5, label='full iterative {:.1E} V/m'.format(field), color='C1')
    # plt.plot(enax, ftot_iter_enax, label='low field iterative {:.1E} V/m'.format(field))
    # plt.plot(enax, ftot_rta_enax, label='low field rta {:.1E} V/m'.format(field))
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel(r'Total occupation ($f^0_{\mathbf{k}} + \chi_{\mathbf{k}}$)')
    # plt.ylim([0, 0.35])
    # plt.xlim([0, 0.475])


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
    spread = 30 * dx

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
    ax.fill(v_ax, vdist, '--', linewidth=2, label='Non-equilibrium deviation', color='orange')
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
        chi_3_i = np.load(outLoc + 'chi_3_gmres_{:.1e}.npy'.format(field))
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
        scmfac = 1
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1
    nkpts = len(df)
    rates = np.load(inLoc + 'scattering_rates.npy')
    rates = scmfac * rates
    plt.figure()
    plt.plot(df['energy [eV]'], rates, '.', MarkerSize=3)
    plt.xlabel('Energy [eV]')
    plt.ylabel(r'Scattering rate [ps$^{-1}$]')


if __name__ == '__main__':
    # Points to inputs and outputs
    out_Loc = pp.outputLoc
    in_Loc = pp.inputLoc

    # Read problem parameters and specify electron DataFrame
    electron_df, ph_df = utilities.load_el_ph_data(in_Loc)
    utilities.fermi_distribution(electron_df)

    # Specify fields to plot over
    fields = np.array([1E4, 2.5E4, 5E4, 7.5E4, 1E5])
    KDEField = 3E5
    plotTransport = False
    plotKDE = False
    plotScattering = True
    plotNoise = False
    plot_vs_energy = False
    applySCMFac = True

    # utilities.translate_into_fbz(electron_df)
    # bz_3dscatter(electron_df)

    if plotTransport:
        driftvel_mobility_vs_field(out_Loc, electron_df, fields)
    if plotKDE:
        plot_vel_KDEs(out_Loc, KDEField, electron_df, plotRTA=False, plotLowField=False, plotFDM=True)
    if plot_vs_energy:
        occupation_v_energy(KDEField, out_Loc, electron_df)
    if plotScattering:
        plot_scattering_rates(in_Loc, electron_df, applySCMFac)
    if plotNoise:
        plot_noise(out_Loc, fields, electron_df, plotRTA=True, plotFDM=True)
    plt.show()

