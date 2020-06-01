import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
import noise_solver

import matplotlib as mpl
font = {'size': 11}
mpl.rc('font', **font)


def plot_scattering_rates(df):
    """Plots the scattering rates by pulling them from the on-diagonal of the simple scattering matrix
    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
    Returns:
        Nothing. Just the plots.
    """
    if pp.scmBool:
        scmfac = pp.scmVal
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1
    nkpts = len(df)
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    g_inds, l_inds, x_inds = utilities.split_valleys(df,False)
    if pp.simpleBool:
        rates = (-1) * np.diag(scm) * scmfac * 1E-12
    else:
        chi2psi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
        rates = (-1) * np.diag(scm) * scmfac * 1E-12 / chi2psi
    plt.figure()
    plt.plot(df.loc[g_inds,'energy [eV]'], rates[g_inds], '.', MarkerSize=3, label=r'$\Gamma$')
    plt.plot(df.loc[l_inds,'energy [eV]'], rates[l_inds], '.', MarkerSize=3, label='L')
    if pp.getX:
        plt.plot(df.loc[x_inds, 'energy [eV]'], rates[x_inds], '.', MarkerSize=3, label=r'X')
    plt.xlabel('Energy [eV]')
    plt.ylabel(r'Scattering rate [ps$^{-1}$]')
    plt.title(pp.title_str)
    plt.legend()


def plot_transport_moments(df,fieldVector,freq):
    """Takes chi solutions which are already calculated and plots transport moments: average energy, drift velocity,
    carrier population, carrier mobility
    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        Nothing. Just the plots.
    """
    vd_1, meanE_1, n_1, mu_1, ng_1, nl_1 = ([] for i in range(6))
    vd_2, meanE_2, n_2, mu_2, ng_2, nl_2 = ([] for i in range(6))
    vd_3, meanE_3, n_3, mu_3, ng_3, nl_3, nx_3, vd_3g, vd_3l, vd_3x, vm_3g, vm_3l, vm_3x, meanE_3g, meanE_3l, meanE_3x \
        , ng_left, ng_right = ([] for i in range(18))
    vd_3t, meanE_3t, n_3t, mu_3t, ng_3t, nl_3t = ([] for i in range(6))
    diff_new = []
    f_1 = np.load(pp.outputLoc + 'Steady/' + 'f_1.npy')
    f_2 = np.load(pp.outputLoc + 'Steady/' + 'f_2.npy')
    g_inds, l_inds, x_inds = utilities.split_valleys(df, False)
    icinds_l = np.load(pp.outputLoc + 'left_icinds.npy')
    icinds_r = np.load(pp.outputLoc + 'right_icinds.npy')
    for ee in fieldVector:
        chi_1_i = utilities.f2chi(f_1, df, ee)
        chi_2_i = utilities.f2chi(f_2, df, ee)
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        chi_3t_i = np.load(pp.outputLoc + 'Transient/' + 'chi_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq,ee))

        vd_1.append(utilities.drift_velocity(chi_1_i,df))
        vd_2.append(utilities.drift_velocity(chi_2_i,df))
        vd_3.append(utilities.drift_velocity(chi_3_i,df))
        vd_3t.append(utilities.drift_velocity(chi_3t_i,df))

        meanE_1.append(utilities.mean_energy(chi_1_i,df))
        meanE_2.append(utilities.mean_energy(chi_2_i,df))
        meanE_3.append(utilities.mean_energy(chi_3_i,df))
        meanE_3t.append(utilities.mean_energy(chi_3t_i,df))

        meanE_3g.append(utilities.mean_energy(chi_3_i[g_inds],df.loc[g_inds]))
        meanE_3l.append(utilities.mean_energy(chi_3_i[l_inds],df.loc[l_inds]))


        mu_1.append(utilities.calc_diff_mobility(chi_1_i,df,ee) * 10 ** 4)
        mu_2.append(utilities.calc_diff_mobility(chi_2_i,df,ee) * 10 ** 4)
        mu_3.append(utilities.calc_diff_mobility(chi_3_i,df,ee) * 10 ** 4)
        mu_3t.append(utilities.calc_diff_mobility(chi_3t_i,df,ee) * 10 ** 4)

        n_1.append(utilities.calculate_noneq_density(chi_1_i,df))
        n_2.append(utilities.calculate_noneq_density(chi_2_i,df))
        n_3.append(utilities.calculate_noneq_density(chi_3_i,df))
        n_3t.append(utilities.calculate_noneq_density(chi_3t_i,df))
        ng_left.append(utilities.calc_popinds(chi_3_i,df,icinds_l))
        ng_right.append(utilities.calc_popinds(chi_3_i,df,icinds_r))

        ng,nl,nx,n = utilities.calc_popsplit(chi_1_i,df)
        ng_1.append(ng)
        nl_1.append(nl)


        ng,nl,nx,n = utilities.calc_popsplit(chi_2_i,df)
        ng_2.append(ng)
        nl_2.append(nl)

        ng,nl,nx,n = utilities.calc_popsplit(chi_3_i,df)
        ng_3.append(ng)
        nl_3.append(nl)

        if pp.getX:
            meanE_3x.append((utilities.mean_energy(chi_3_i[x_inds],df.loc[x_inds])))
            nx_3.append(nx)
            vd_3x.append(utilities.mean_velocity(chi_3_i[x_inds],df.loc[x_inds]))
            vm_3x.append(utilities.mean_xvelocity_mag(chi_3_i[x_inds],df.loc[x_inds]))

        ng,nl,nx,n = utilities.calc_popsplit(chi_3t_i,df)
        ng_3t.append(ng)
        nl_3t.append(nl)

        vd_3g.append(utilities.mean_velocity(chi_3_i[g_inds],df.loc[g_inds]))
        vd_3l.append(utilities.mean_velocity(chi_3_i[l_inds],df.loc[l_inds]))
        vm_3g.append(utilities.mean_xvelocity_mag(chi_3_i[g_inds],df.loc[g_inds]))
        vm_3l.append(utilities.mean_xvelocity_mag(chi_3_i[l_inds],df.loc[l_inds]))
        N = np.sum(chi_3_i+df['k_FD'])
        Ng = np.sum(chi_3_i[g_inds]+df.loc[g_inds,'k_FD'])
        Nl = np.sum(chi_3_i[l_inds]+df.loc[l_inds,'k_FD'])

        diff_new.append((vd_3g[-1]-vd_3l[-1])**2/(2*N**2)*(Ng*Nl))

    kvcm = np.array(fieldVector)*1e-5

    plt.figure()
    plt.plot(kvcm,diff_new,'o-', linewidth=2, label='RTA')
    plt.xlabel(r'Field [kV/cm]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,vd_1,'o-', linewidth=2, label='RTA')
    plt.plot(kvcm,vd_2,'o-', linewidth=2, label='L-F')
    plt.plot(kvcm,vd_3,'o-', linewidth=2, label='FDM')
    # plt.plot(kvcm,np.array(mu_3)/1e4*np.array(fieldVector), label=r'$\mu E$')
    plt.plot(kvcm,vd_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Drift velocity [m/s]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,n_1,'o-', linewidth=2, label='RTA')
    plt.plot(kvcm,n_2,'o-', linewidth=2, label='L-F')
    plt.plot(kvcm,n_3,'o-', linewidth=2, label='FDM')
    # plt.plot(kvcm,np.array(mu_3)/1e4*np.array(fieldVector), label=r'$\mu E$')
    plt.plot(kvcm,n_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))

    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Carrier Population [m$^-3$]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,mu_1,'o-', linewidth=2, label='RTA')
    plt.plot(kvcm,mu_2,'o-', linewidth=2, label='L-F')
    plt.plot(kvcm,mu_3,'o-', linewidth=2, label='FDM')
    plt.plot(kvcm,mu_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Differential mobility [$cm^2 V^{-1} s^{-1}$]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,meanE_1,'o-', linewidth=2, label='RTA')
    plt.plot(kvcm,meanE_2,'o-', linewidth=2, label='L-F')
    plt.plot(kvcm,meanE_3,'o-', linewidth=2, label='FDM')
    plt.plot(kvcm,meanE_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Mean energy [eV]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(meanE_1,mu_1,'o-', linewidth=2, label='RTA')
    plt.plot(meanE_2,mu_2,'o-', linewidth=2, label='L-F')
    plt.plot(meanE_3,mu_3,'o-', linewidth=2, label='FDM')
    plt.plot(meanE_3t,mu_3t,'o-', linewidth=2, label='FDM {:.1e} GHz'.format(freq))
    plt.xlabel(r'Mean energy [eV]')
    plt.ylabel(r'Differential mobility [$cm^2 V^{-1} s^{-1}$]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,meanE_3g,'o-', linewidth=2, label='FDM Gamma')
    plt.plot(kvcm,meanE_3l,'o-', linewidth=2, label='FDM L')
    if pp.getX:
        plt.plot(kvcm, meanE_3x, 'o-', linewidth=2, label='FDM X')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Mean energy [eV]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,ng_right,'o-', linewidth=2, label=r'FDM Left')
    plt.plot(kvcm,ng_left,'o-', linewidth=2, label=r'FDM Right')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Carrier Population [m$^-3$]')
    plt.title(pp.title_str)
    plt.legend()

    x = df.loc[icinds_r,'kx [1/A]'].values / (2 * np.pi / c.a)
    y = df.loc[icinds_r,'ky [1/A]'].values / (2 * np.pi / c.a)
    z = df.loc[icinds_r,'kz [1/A]'].values / (2 * np.pi / c.a)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    plt.figure()
    plt.plot(kvcm,ng_1,'o-', linewidth=2, label=r'RTA Gamma')
    plt.plot(kvcm,ng_2,'o-', linewidth=2, label=r'L-F Gamma')
    plt.plot(kvcm,ng_3,'o-', linewidth=2, label=r'FDM Gamma')
    plt.plot(kvcm,ng_3t,'o-', linewidth=2, label=r'FDM {:.1e} GHz Gamma'.format(freq))

    plt.plot(kvcm,nl_1,'o-', linewidth=2, label=r'RTA L')
    plt.plot(kvcm,nl_2,'o-', linewidth=2, label=r'L-F L')
    plt.plot(kvcm,nl_3,'o-', linewidth=2, label=r'FDM L')
    plt.plot(kvcm,nl_3t,'o-', linewidth=2, label=r'FDM {:.1e} GHz L'.format(freq))
    if pp.getX:
        plt.plot(kvcm, nx_3, 'o-', linewidth=2, label='FDM X')
        # plt.plot(kvcm,np.array(nx_3)+np.array(nl_3)+np.array(ng_3),'--',linewidth=2,label='Sum')

    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Carrier Population [m$^-3$]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,vd_3g,'o-', linewidth=2, label=r'FDM Gamma')
    plt.plot(kvcm,vd_3l,'o-', linewidth=2, label=r'FDM L')
    if pp.getX:
        plt.plot(kvcm, vd_3x, 'o-', linewidth=2, label='FDM X')
        plt.plot(kvcm, np.array(vd_3l) * np.array(nl_3) / np.real(n) + np.array(vd_3g) * np.array(ng_3) / np.real(n) +
                 np.array(vd_3x) * np.array(nx_3) / np.real(n),
                 '-o', linewidth=2, label=r'Bulk Drift')
    else:
        plt.plot(kvcm,np.array(vd_3l)*np.array(nl_3)/np.real(n)+np.array(vd_3g)*np.array(ng_3)/np.real(n),'-o', linewidth=2, label=r'Bulk Drift')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Drift velocity [m/s]')
    plt.title(pp.title_str)
    plt.legend()

    plt.figure()
    plt.plot(kvcm,vm_3g,'o-', linewidth=2, label=r'FDM Gamma')
    plt.plot(kvcm,vm_3l,'o-', linewidth=2, label=r'FDM L')
    if pp.getX:
        plt.plot(kvcm, vm_3x, 'o-', linewidth=2, label=r'FDM X')
    plt.plot(kvcm,np.array(vd_3l)*np.array(nl_3)/np.real(n)+np.array(vd_3g)*np.array(ng_3)/np.real(n),'-o', linewidth=2, label=r'Bulk Drift')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Average x-velocity magnitude [m/s]')
    plt.title(pp.title_str)
    plt.legend()


def plot_diffusion(df,fieldVector,freq):
    """Takes chi solutions which are already calculated and plots the thermal and intervalley diffusion.
    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        Nothing. Just the plots.
    """
    diff_iv = []
    diff_iv_gl = []
    diff_iv_gx = []
    diff_iv_lx = []

    diff_th_g = []
    diff_th_l = []
    diff_th_x = []
    diff_th = []
    n = []
    ng = []
    nl = []
    nx = []
    for ee in fieldVector:
        chi_3t_i = np.load(pp.outputLoc + 'Transient/' + 'chi_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq,ee))
        if pp.getX:
            ng_nl_i = np.load(pp.outputLoc + 'Intervalley/' + 'ng_nl_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq,ee))
            ng_nx_i = np.load(pp.outputLoc + 'Intervalley/' + 'ng_nx_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq,ee))
            nl_nx_i = np.load(pp.outputLoc + 'Intervalley/' + 'nl_nx_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq,ee))
            vd_vd_i = np.load(pp.outputLoc + 'Thermal/' + 'vd_vd_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq,ee))
            dth,dg,dl, dx = noise_solver.thermal_diffusion(vd_vd_i,chi_3t_i,df)
            D_tr, D_tr_gl, D_tr_gx, D_tr_lx = noise_solver.intervalley_diffusion_three_valley(ng_nl_i, ng_nx_i, nl_nx_i, chi_3t_i, df)
            diff_iv.append(D_tr)
            diff_iv_gl.append(D_tr_gl)
            diff_iv_gx.append(D_tr_gx)
            diff_iv_lx.append(D_tr_lx)

        else:
            ng_ng_i = np.load(pp.outputLoc + 'Intervalley/' + 'ng_ng_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq,ee))
            vd_vd_i = np.load(pp.outputLoc + 'Thermal/' + 'vd_vd_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq,ee))
            diff_iv.append(noise_solver.intervalley_diffusion_two_valley(ng_ng_i,chi_3t_i,df))
            dth,dg,dl, dx = noise_solver.thermal_diffusion(vd_vd_i,chi_3t_i,df)
        diff_th_g.append(dg)
        diff_th_l.append(dl)
        if pp.getX:
            diff_th_x.append(dx)
        diff_th.append(dth)

        ng_i,nl_i,nx_i,n_i = utilities.calc_popsplit(chi_3t_i,df)
        ng.append(ng_i)
        nl.append(nl_i)
        nx.append(nx_i)
        n.append(n_i)

    kvcm = np.array(fieldVector)*1e-5
    chi_3t_i = np.load(pp.outputLoc + 'Transient/' + 'chi_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq, fieldVector[0]))
    mu = utilities.calc_diff_mobility(chi_3t_i, df, fieldVector[0])
    plt.figure()
    plt.plot(kvcm,diff_th_g,'o-', linewidth=2, label=r'$\Gamma$')
    plt.plot(kvcm,diff_th_l,'o-', linewidth=2, label='L')
    plt.plot(kvcm,diff_th,'o-', linewidth=2, label='Total')
    if pp.getX:
        plt.plot(kvcm, diff_th_x, 'o-', linewidth=2, label='X')
    plt.hlines(mu*c.kb_joule*pp.T/c.e, kvcm[0],kvcm[-1],linestyles='dashed',linewidth=2,label='Einstein')
    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Thermal Diffusion coefficient [m^2/s]')
    plt.title(pp.title_str)
    plt.legend()

    if pp.getX:
        plt.figure()
        plt.plot(kvcm,diff_iv,'o-', linewidth=2, label=r'IV Total')
        plt.plot(kvcm,diff_iv_gl,'o-', linewidth=2, label=r'Gamma-L')
        plt.plot(kvcm,diff_iv_gx,'o-', linewidth=2, label=r'Gamma-X')
        plt.plot(kvcm,diff_iv_lx,'o-', linewidth=2, label=r'L-X')

        plt.xlabel(r'Field [kV/cm]')
        plt.ylabel(r'Intervalley Diffusion coefficient [m^2/s]')
        plt.title(pp.title_str)
        plt.legend()
    else:
        plt.figure()
        plt.plot(kvcm, diff_iv, 'o-', linewidth=2, label=r'IV Total')
        plt.xlabel(r'Field [kV/cm]')
        plt.ylabel(r'Intervalley Diffusion coefficient [m^2/s]')
        plt.title(pp.title_str)

    plt.figure()
    plt.plot(kvcm,np.array(ng)/np.array(n),'o-', linewidth=2, label=r'Gamma')
    plt.plot(kvcm,np.array(nl)/np.array(n),'o-', linewidth=2, label=r'L')
    if pp.getX:
        plt.plot(kvcm,np.array(nx)/np.array(n),'o-', linewidth=2, label=r'X')

    plt.xlabel(r'Field [kV/cm]')
    plt.ylabel(r'Population fraction')
    plt.title(pp.title_str)
    plt.legend()


if __name__ == '__main__':
    fields = pp.fieldVector
    freq = pp.freqGHz
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    # plot_scattering_rates(electron_df)
    plot_transport_moments(electron_df, fields, freq)
    # plot_diffusion(electron_df, fields, freq)

    plt.show()