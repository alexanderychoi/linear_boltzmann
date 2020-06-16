import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy.linalg
import plotly.offline as py
import plotly.graph_objs as go
import preprocessing
from mpl_toolkits.mplot3d import Axes3D


def plot_steady_transient_difference(fieldVector, freq):
    """Wrapper script for velocity_distribution_kde. Can do for the various solution schemes saved to file.
        Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        freq (dbl): Frequency in GHz to be used for the transient solution

    Returns:
        Nothing. Just the plots.
    """
    ac_dc_error = []
    ac_dc_cosine = []
    for ee in fieldVector:
        transient_chi = np.load(pp.outputLoc + 'Transient/' + 'chi_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq,ee))
        steady_chi = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        ac_dc_error.append(np.linalg.norm(transient_chi - steady_chi) / np.linalg.norm(steady_chi))
        ac_dc_cosine.append(distance.cosine(steady_chi,transient_chi)*-1)
        print('Relative residual is {:E}'.format(ac_dc_error[-1]))
        print('Cosine similarity is {:E}'.format(ac_dc_cosine[-1]))

    plt.figure()
    plt.plot(fieldVector*1E-5, ac_dc_error)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel('Re[{:.1e} GHz]'.format(freq) + '/DC Residual')
    plt.title(pp.title_str)

    plt.figure()
    plt.plot(fieldVector*1E-5, ac_dc_cosine)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel('Re[{:.1e} GHz]'.format(freq) + '/DC Cosine Similarity')
    plt.title(pp.title_str)


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


def plot_vel_KDEs(field, df, freq):
    """Wrapper script for velocity_distribution_kde. Can do for the various solution schemes saved to file.
        Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions and ready steady state chis.
        field (dbl): the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        Nothing. Just the plots.
    """
    # chi_3t_i = np.load(pp.outputLoc + 'Transient/' + 'chi_' + '3_' + "f_{:.1e}_E_{:.1e}.npy".format(freq,field))
    # chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(field))
    # chi_2_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}.npy".format(field))
    # chi_1_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '1_' + "E_{:.1e}.npy".format(field))
    # velocity_distribution_kde(chi_3_i, df, title='DC Chi FDM {:.1e} V/m '.format(field) + pp.title_str)
    # velocity_distribution_kde(chi_2_i, df, title='DC Chi L-F {:.1e} V/m '.format(field) + pp.title_str)
    # velocity_distribution_kde(chi_1_i, df, title='DC Chi RTA {:.1e} V/m '.format(field) + pp.title_str)
    # velocity_distribution_kde(np.real(chi_3t_i), df, title=r'{:.1e} GHz Chi {:.1e} V/m '.format(freq,field) + pp.title_str)

    chi_3 = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(field))
    g_inds,l_inds,x_inds = utilities.split_valleys(df,False)
    chi_3_g = chi_3[g_inds]
    chi_3_l = chi_3[l_inds]
    if pp.getX:
        chi_3_x = chi_3[x_inds]
    # velocity_distribution_kde(chi_3, df, title='DC Chi FDM {:.1e} V/m '.format(field) + pp.title_str)

    velocity_distribution_kde(chi_3_g, df.loc[g_inds].reset_index(), title='Gamma Chi FDM {:.1e} V/m '.format(field) + pp.title_str)
    velocity_distribution_kde(chi_3_l, df.loc[l_inds].reset_index(), title='L Chi FDM {:.1e} V/m '.format(field) + pp.title_str)
    if pp.getX:
        velocity_distribution_kde(chi_3_x, df.loc[x_inds].reset_index(), title='X Chi FDM {:.1e} V/m '.format(field) + pp.title_str)


def occupation_v_energy_sep(chi, enk, kptsdf):
    g_inds, l_inds, x_inds = utilities.split_valleys(kptsdf,False)

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

    g_chi = chi[g_inds]
    l_chi = chi[l_inds]
    g_enk = enk[g_inds]
    l_enk = enk[l_inds]

    dx = (g_en_axis.max() - g_en_axis.min()) / npts
    g_f0 = np.squeeze(kptsdf.loc[g_inds,'k_FD'].values)
    l_f0 = np.squeeze(kptsdf.loc[l_inds,'k_FD'].values)
    spread = 90 * dx
    spread = 120 * dx
    spread = 200 * dx

    def gaussian(x, mu, vmag, stdev=spread):
        sigma = stdev - (vmag/1E6) * 0.9 * stdev
        vals = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
        return vals

    for k in range(len(g_chi)):
        istart = int(np.maximum(np.floor((g_enk[k] - g_en_axis[0]) / dx) - (4*spread/dx), 0))
        iend = int(np.minimum(np.floor((g_enk[k] - g_en_axis[0]) / dx) + (4*spread/dx), npts - 1))
        g_ftot[istart:iend] += (g_chi[k] + g_f0[k]) * gaussian(g_en_axis[istart:iend], g_enk[k],vmags[k])
        g_f0ax[istart:iend] += g_f0[k] * gaussian(g_en_axis[istart:iend], g_enk[k], vmags[k])
        g_chiax[istart:iend] += g_chi[k] * gaussian(g_en_axis[istart:iend], g_enk[k], vmags[k])

    for k in range(len(l_chi)):
        istart = int(np.maximum(np.floor((l_enk[k] - l_en_axis[0]) / dx) - (4 * spread / dx), 0))
        iend = int(np.minimum(np.floor((l_enk[k] - l_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
        l_ftot[istart:iend] += (l_chi[k] + l_f0[k]) * gaussian(l_en_axis[istart:iend], l_enk[k], vmags[k])
        l_f0ax[istart:iend] += l_f0[k] * gaussian(l_en_axis[istart:iend], l_enk[k], vmags[k])
        l_chiax[istart:iend] += l_chi[k] * gaussian(l_en_axis[istart:iend], l_enk[k], vmags[k])

    if pp.getX:
        x_chiax = np.zeros(npts)  # capital sigma as defined in Jin Jian's paper Eqn 3
        x_ftot = np.zeros(npts)
        x_f0ax = np.zeros(npts)
        x_en_axis = np.linspace(enk.min(), enk.max(), npts)
        x_chi = chi[x_inds]
        x_enk = enk[x_inds]
        x_f0 = np.squeeze(kptsdf.loc[x_inds,'k_FD'].values)
        for k in range(len(x_chi)):
            istart = int(np.maximum(np.floor((x_enk[k] - x_en_axis[0]) / dx) - (4 * spread / dx), 0))
            iend = int(np.minimum(np.floor((x_enk[k] - x_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
            x_ftot[istart:iend] += (x_chi[k] + x_f0[k]) * gaussian(x_en_axis[istart:iend], x_enk[k], vmags[k])
            x_f0ax[istart:iend] += x_f0[k] * gaussian(x_en_axis[istart:iend], x_enk[k], vmags[k])
            x_chiax[istart:iend] += x_chi[k] * gaussian(x_en_axis[istart:iend], x_enk[k], vmags[k])
    else:
        x_en_axis = []
        x_ftot = []
        x_chi = []
        x_chiax = []
        x_f0ax =[]
    return g_en_axis, g_ftot, g_chiax, g_f0ax, l_en_axis, l_ftot, l_chiax, l_f0ax, x_en_axis, x_ftot, x_chiax, x_f0ax


def plot_energy_sep(df,fields):
    utilities.split_valleys(df, True)
    plt.figure(figsize=(6, 6))
    for ee in fields:
        chi_3_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}.npy".format(ee))
        g_en_axis, g_ftot, g_chiax, g_f0ax, l_en_axis, l_ftot, l_chiax, l_f0ax, x_en_axis, x_ftot, x_chiax, x_f0ax = \
        occupation_v_energy_sep(chi_3_i, df['energy [eV]'].values, df)
        plt.plot(g_en_axis-np.min(df['energy [eV]']), g_chiax, label=r'$\Gamma$'+' Valley')
        plt.plot(l_en_axis-np.min(df['energy [eV]']), l_chiax, '--', label='L Valley')
        if pp.getX:
            plt.plot(x_en_axis - np.min(df['energy [eV]']), x_chiax, '--', label='X Valley')
    plt.xlabel('Energy above CBM (eV)')
    plt.ylabel(r'FDM deviational occupation ($\delta f_{\mathbf{k}}$) [arb]')
    plt.title(pp.title_str)

    # plt.figure()
    # for ee in fields:
    #     chi_2_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}.npy".format(ee))
    #     g_en_axis, g_ftot, g_chiax, g_f0ax, l_en_axis, l_ftot, l_chiax, l_f0ax, x_en_axis, x_ftot, x_chiax, x_f0ax = \
    #     occupation_v_energy_sep(chi_2_i, df['energy [eV]'].values, df)
    #     plt.plot(g_en_axis-np.min(df['energy [eV]']), g_chiax, label=r'$\Gamma$'+' Valley')
    #     plt.plot(l_en_axis-np.min(df['energy [eV]']), l_chiax, '--', label='L Valley')
    #     if pp.getX:
    #         plt.plot(x_en_axis - np.min(df['energy [eV]']), x_chiax, '--', label='X Valley')
    # plt.xlabel('Energy above CBM (eV)')
    # plt.ylim([-0.05,0.05])
    # plt.ylabel(r'L-F deviational occupation ($\delta f_{\mathbf{k}}$) [arb]')
    # plt.title(pp.title_str)
    #
    # plt.figure()
    # for ee in fields:
    #     chi_1_i = np.load(pp.outputLoc + 'Steady/' + 'chi_' + '1_' + "E_{:.1e}.npy".format(ee))
    #     g_en_axis, g_ftot, g_chiax, g_f0ax, l_en_axis, l_ftot, l_chiax, l_f0ax, x_en_axis, x_ftot, x_chiax, x_f0ax = \
    #     occupation_v_energy_sep(chi_1_i, df['energy [eV]'].values, df)
    #     plt.plot(g_en_axis-np.min(df['energy [eV]']), g_chiax, label=r'$\Gamma$'+' Valley')
    #     plt.plot(l_en_axis-np.min(df['energy [eV]']), l_chiax, '--', label='L Valley')
    #     if pp.getX:
    #         plt.plot(x_en_axis - np.min(df['energy [eV]']), x_chiax, '--', label='X Valley')
    # plt.xlabel('Energy above CBM (eV)')
    # plt.ylim([-0.05,0.05])
    # plt.ylabel(r'RTA deviational occupation ($\delta f_{\mathbf{k}}$) [arb]')
    # plt.title(pp.title_str)


def bz_3dscatter(points, useplotly=True, icind=False):
    if icind:
        icinds_l = np.load(pp.outputLoc+'left_icinds.npy')
        icinds_r = np.load(pp.outputLoc+'right_icinds.npy')
        icr_df = points.loc[icinds_r]
        icl_df = points.loc[icinds_l]

    if useplotly:
        if np.any(points['energy [eV]']):
            colors = points['energy [eV]']
        else:
            colors = 'k'
        trace1 = go.Scatter3d(
            x=points['kx [1/A]'].values / (2 * np.pi / c.a),
            y=points['ky [1/A]'].values / (2 * np.pi / c.a),
            z=points['kz [1/A]'].values / (2 * np.pi / c.a),

            mode='markers',
            marker=dict(size=2, color=colors, colorscale='Rainbow', showscale=True, opacity=1)
        )
        if icind:
            trace2 = go.Scatter3d(
                x=icr_df['kx [1/A]'].values / (2 * np.pi / c.a),
                y=icr_df['ky [1/A]'].values / (2 * np.pi / c.a),
                z=icr_df['kz [1/A]'].values / (2 * np.pi / c.a),

                mode='markers',
                marker=dict(size=2, color='black', opacity=1)
            )

            trace3 = go.Scatter3d(
                x=icl_df['kx [1/A]'].values / (2 * np.pi / c.a),
                y=icl_df['ky [1/A]'].values / (2 * np.pi / c.a),
                z=icl_df['kz [1/A]'].values / (2 * np.pi / c.a),

                mode='markers',
                marker=dict(size=2, color='#7f7f7f', opacity=1)
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
        if icind:
            data = [trace1, trace2,trace3, vector1, vector2, vector3, vector4, vector5, vector6, vector7]

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
    else:
        x = points['kx [1/A]'].values / (2 * np.pi / c.a)
        y = points['ky [1/A]'].values / (2 * np.pi / c.a)
        z = points['kz [1/A]'].values / (2 * np.pi / c.a)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        return ax


if __name__ == '__main__':
    fields = pp.fieldVector
    freq = pp.freqGHz
    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    # electron_df = utilities.fermi_distribution(electron_df)
    bz_3dscatter(electron_df, True, True)

    bz_3dscatter()
    # plot_steady_transient_difference(fields,freq)
    # plot_vel_KDEs(fields[-1],electron_df,pp.freqGHz)
    # plot_energy_sep(electron_df, fields)
    plt.show()