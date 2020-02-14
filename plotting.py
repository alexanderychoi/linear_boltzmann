#!/usr/bin/env python

import preprocessing_largegrid
import noise_power

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import os

import plotly.offline as py
import plotly.graph_objs as go
import plotly


def highlighted_points(fig, points, con):
    fig.add_trace(go.Scatter3d(
                    x=points['kx [1/A]'].values / (2 * np.pi / con.a),
                    y=points['ky [1/A]'].values / (2 * np.pi / con.a),
                    z=points['kz [1/A]'].values / (2 * np.pi / con.a),
                    mode='markers',
                    marker=dict(size=3, color='black', opacity=1)
                )
    )
    py.plot(fig, filename='bz_scatter_highlights.html')


def bz_3dscatter(con, points, energies=pd.DataFrame([]), useplotly=True):
    if useplotly:
        if np.any(energies['energy [Ryd]']):
            colors = energies['energy [Ryd]']
        else:
            colors = 'k'
        trace1 = go.Scatter3d(
            x=points['kx [1/A]'].values / (2 * np.pi / con.a),
            y=points['ky [1/A]'].values / (2 * np.pi / con.a),
            z=points['kz [1/A]'].values / (2 * np.pi / con.a),
            # x=points[:, 0] / (2 * np.pi / con.a),
            # y=points[:, 1] / (2 * np.pi / con.a),
            # z=points[:, 2] / (2 * np.pi / con.a),
            mode='markers',
            marker=dict(size=2, color=colors, colorscale='Rainbow', showscale=True, opacity=1)
        )

        b1edge = 0.5 * con.b1 / (2 * np.pi / con.a)
        vector1 = go.Scatter3d(x=[0, b1edge[0]], y=[0, b1edge[1]], z=[0, b1edge[2]],
                               marker=dict(size=1,color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))
        b2edge = 0.5 * con.b2 / (2 * np.pi / con.a)
        vector2 = go.Scatter3d(x=[0, b2edge[0]], y=[0, b2edge[1]], z=[0, b2edge[2]],
                               marker=dict(size=1,color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))
        b3edge = 0.5 * con.b3 / (2 * np.pi / con.a)
        vector3 = go.Scatter3d(x=[0, b3edge[0]], y=[0, b3edge[1]], z=[0, b3edge[2]],
                               marker=dict(size=1,color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))
        xedge = -0.5 * (con.b1 + con.b3) / (2 * np.pi / con.a)
        vector4 = go.Scatter3d(x=[0, xedge[0]], y=[0, xedge[1]], z=[0, xedge[2]],
                               marker=dict(size=1, color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))
        yedge = 0.5 * (con.b2 + con.b3) / (2 * np.pi / con.a)
        vector5 = go.Scatter3d(x=[0, yedge[0]], y=[0, yedge[1]], z=[0, yedge[2]],
                               marker=dict(size=1, color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))
        zedge = 0.5 * (con.b1 + con.b2) / (2 * np.pi / con.a)
        vector6 = go.Scatter3d(x=[0, zedge[0]], y=[0, zedge[1]], z=[0, zedge[2]],
                               marker=dict(size=1, color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))
        ledge = 0.5 * (con.b1 + con.b2 + con.b3) / (2 * np.pi / con.a)
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
    else:
        x = points['kx [1/A]'].values / (2 * np.pi / con.a)
        y = points['ky [1/A]'].values / (2 * np.pi / con.a)
        z = points['kz [1/A]'].values / (2 * np.pi / con.a)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        return ax


def plot_dispersion(kpts, enk):
    """Plots phonon dispersion.

    Path is hardcoded for FCC unit cell. Currently just plotting Gamma-L and Gamma-X

    Parameters:
    ------------
    kpts : dataframe containing
        k_inds : vector_like, shape (n,1)
        Index of k point

        'kx [1/A]' : vector_like, shape (n,1)
        x-coordinate in Cartesian momentum space

        'ky [1/A]' : vector_like, shape (n,1)
        y-coordinate in Cartesian momentum space

        'kz [1/A]' : vector_like, shape (n,1)
        z-coordinate in Cartesian momentum space

    enk : dataframe containing
        k_inds : vector_like, shape (n,1)
        Index of k point

        band_inds : vector_like, shape (n,1)
        Band index

        energy [Ryd] : vector_like, shape (n,1)
        Energy associated with k point in Rydberg units

    Returns:
    ---------
    No variable returns. Just plots the dispersion
    """

    # Lattice constant and reciprocal lattice vectors
    # b1 = 2 pi/a (kx - ky + kz)
    # b2 = 2 pi/a (kx + ky - kz)
    # b3 = 2 pi/a (-kx + ky + kz)
    a = 5.556  # [A]
    b1 = (2 * np.pi / a) * np.array([1, -1, 1])
    b2 = (2 * np.pi / a) * np.array([1, 1, -1])
    b3 = (2 * np.pi / a) * np.array([-1, 1, 1])

    # L point in BZ is given by 0.5*b1 + 0.5*b2 + 0.5*b3
    # X point in BZ is given by 0.5*b2 + 0.5*b3
    lpoint = 0.5 * (b1 + b2 + b3)
    xpoint = 0.5 * (b2 + b3)

    # We can find kpoints along a path just by considering a dot product with lpoint and xpoint vectors.
    # Any kpoints with angle smaller than some tolerance are considered on the path and we can plot their frequencies
    deg2rad = 2 * np.pi / 360
    ang_tol = 1 * deg2rad  # 1 degree in radians

    print(list(kpts))

    enkonly = np.array(enk['energy [Ryd]'])[:, np.newaxis]
    enkinds = np.array(enk['q_inds'])
    kptsonly = np.array(kpts[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']]) / (2 * np.pi / a)
    kptsinds = np.array(kpts['q_inds'])
    kptsmag = np.linalg.norm(kptsonly, axis=1)[:, np.newaxis]

    dot_l = np.zeros(len(kpts))
    dot_x = np.zeros(len(kpts))

    # Separate assignment for gamma point to avoid divide by zero error
    nongamma = kptsmag != 0
    dot_l[np.squeeze(nongamma)] = np.divide(np.dot(kptsonly, lpoint[:, np.newaxis])[nongamma],
                                            kptsmag[nongamma]) / np.linalg.norm(lpoint)
    dot_x[np.squeeze(nongamma)] = np.divide(np.dot(kptsonly, xpoint[:, np.newaxis])[nongamma],
                                            kptsmag[nongamma]) / np.linalg.norm(xpoint)
    dot_l[np.squeeze(kptsmag == 0)] = 0
    dot_x[np.squeeze(kptsmag == 0)] = 0

    lpath = np.logical_or(np.arccos(dot_l) < ang_tol, np.squeeze(kptsmag == 0))
    xpath = np.logical_or(np.arccos(dot_x) < ang_tol, np.squeeze(kptsmag == 0))

    linds = kptsinds[lpath]
    xinds = kptsinds[xpath]
    lkmag = kptsmag[lpath]
    xkmag = kptsmag[xpath]

    plt.figure()

    for i, ki in enumerate(linds):
        energies = enkonly[enkinds == ki, 0]
        thiskmag = lkmag[i]
        if len(energies) > 1:
            veck = np.ones((len(energies), 1)) * thiskmag
            plt.plot(veck, energies, '.', color='C0')
        else:
            plt.plot(thiskmag, energies, '.', color='C0')

    for i, ki in enumerate(xinds):
        energies = enkonly[enkinds == ki, 0]
        thiskmag = lkmag[i]
        if len(energies) > 1:
            veck = np.ones((len(energies), 1)) * thiskmag
            plt.plot(-1 * veck, energies, '.', color='C1')
        else:
            plt.plot(-1 * thiskmag, energies, '.', color='C1')

    plt.xlabel('k magnitude')
    plt.ylabel('Energy in Ry')


def plot_bandstructure(kpts, enk):
    """Plots electron bandstructure.

    Path is hardcoded for FCC unit cell. Currently just plotting Gamma-L and Gamma-X

    Parameters:
    ------------
    kpts : dataframe containing
        k_inds : vector_like, shape (n,1)
        Index of k point

        'kx [1/A]' : vector_like, shape (n,1)
        x-coordinate in Cartesian momentum space

        'ky [1/A]' : vector_like, shape (n,1)
        y-coordinate in Cartesian momentum space

        'kz [1/A]' : vector_like, shape (n,1)
        z-coordinate in Cartesian momentum space

    enk : dataframe containing
        k_inds : vector_like, shape (n,1)
        Index of k point

        band_inds : vector_like, shape (n,1)
        Band index

        energy [Ryd] : vector_like, shape (n,1)
        Energy associated with k point in Rydberg units

    Returns:
    ---------
    No variable returns. Just plots the dispersion"""

    # Lattice constant and reciprocal lattice vectors
    # b1 = 2 pi/a (kx - ky + kz)
    # b2 = 2 pi/a (kx + ky - kz)
    # b3 = 2 pi/a (-kx + ky + kz)
    a = 5.556  # [A]
    b1 = (2 * np.pi / a) * np.array([1, -1, 1])
    b2 = (2 * np.pi / a) * np.array([1, 1, -1])
    b3 = (2 * np.pi / a) * np.array([-1, 1, 1])

    # L point in BZ is given by 0.5*b1 + 0.5*b2 + 0.5*b3
    # X point in BZ is given by 0.5*b2 + 0.5*b3
    lpoint = 0.5 * (b1 + b2 + b3)
    xpoint = 0.5 * (b2 + b3)

    # We can find kpoints along a path just by considering a dot product with lpoint and xpoint vectors.
    # Any kpoints with angle smaller than some tolerance are considered on the path and we can plot their energies
    deg2rad = 2 * np.pi / 360
    ang_tol = 1 * deg2rad  # 1 degree in radians

    enkonly = np.array(enk['energy [Ryd]'])[:, np.newaxis]
    enkinds = np.array(enk['k_inds'])
    kptsonly = np.array(kpts[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']]) / (2 * np.pi / a)
    kptsinds = np.array(kpts['k_inds'])
    kptsmag = np.linalg.norm(kptsonly, axis=1)[:, np.newaxis]

    dot_l = np.zeros(len(kpts))
    dot_x = np.zeros(len(kpts))

    # Separate assignment for gamma point to avoid divide by zero error
    nongamma = kptsmag != 0
    dot_l[np.squeeze(nongamma)] = np.divide(np.dot(kptsonly, lpoint[:, np.newaxis])[nongamma],
                                            kptsmag[nongamma]) / np.linalg.norm(lpoint)
    dot_x[np.squeeze(nongamma)] = np.divide(np.dot(kptsonly, xpoint[:, np.newaxis])[nongamma],
                                            kptsmag[nongamma]) / np.linalg.norm(xpoint)
    dot_l[np.squeeze(kptsmag == 0)] = 0
    dot_x[np.squeeze(kptsmag == 0)] = 0

    lpath = np.logical_or(np.arccos(dot_l) < ang_tol, np.squeeze(kptsmag == 0))
    xpath = np.logical_or(np.arccos(dot_x) < ang_tol, np.squeeze(kptsmag == 0))

    linds = kptsinds[lpath]
    xinds = kptsinds[xpath]
    lkmag = kptsmag[lpath]
    xkmag = kptsmag[xpath]

    plt.figure()

    for i, ki in enumerate(linds):
        energies = enkonly[enkinds == ki, 0]
        thiskmag = lkmag[i]
        if len(energies) > 1:
            veck = np.ones((len(energies), 1)) * thiskmag
            # plt.plot(veck, theseenergies, '.', color='C0')
        else:
            plt.plot(thiskmag, energies, '.', color='C0')

    for i, ki in enumerate(xinds):
        energies = enkonly[enkinds == ki, 0]
        thiskmag = lkmag[i]
        if len(energies) > 1:
            veck = np.ones((len(energies), 1)) * thiskmag
            plt.plot(-1 * veck, energies, '.', color='C1')
        else:
            plt.plot(-1 * thiskmag, energies, '.', color='C1')

    plt.xlabel('k magnitude')
    plt.ylabel('Energy in Ry')


def plot_scattering_rates(data_dir, energies, kpts):
    rates = np.load(data_dir + 'scattering_rates_direct.npy')

    # scm = np.memmap(data_dir + 'scattering_matrix.mmap', dtype='float64', mode='r', shape=(42433, 42433))
    # f0 = np.squeeze(kpts['k_FD'])
    # rates = (-1) * np.diag(scm) / (f0 * (1-f0)) * (2*np.pi)**2 * 1E-12

    rates = rates * (2*np.pi)**2

    font = {'size': 14}
    matplotlib.rc('font', **font)

    plt.plot(energies, rates, '.', MarkerSize=3)
    plt.xlabel('Energy [eV]')
    plt.ylabel(r'Scattering rate [ps$^{-1}$]')
    plt.xlim([0, 0.4])
    # plt.savefig('plot_scattering_rates.png')


def plot_cg_iter_rta(f_cg,f_iter,f_rta):
    font = {'size': 14}
    matplotlib.rc('font', **font)
    plt.plot(f_cg, linewidth=1, label='CG')
    plt.plot(f_iter, linewidth=1, label='Iterative')
    plt.plot(f_rta, linewidth=1, label='RTA')
    plt.xlabel('kpoint index')
    plt.ylabel('deviational occupation')
    plt.legend()
    plt.show()


def highlight_L(kpt_df,L_inds):
    print(L_inds)
    plotvec = np.sqrt(kpt_df['kx [1/A]'].values**2 + kpt_df['ky [1/A]'].values**2 + kpt_df['kz [1/A]'].values**2)
    font = {'size': 14}
    matplotlib.rc('font', **font)
    plt.plot(plotvec, linewidth=1, label='All')
    plt.plot(plotvec[L_inds], linewidth=1, label='L Energy')
    plt.xlabel('kpoint index')
    plt.ylabel('kpt mag [1/A]')
    plt.legend()
    plt.show()


def plot_1dim_steady_soln(solns, labels, fullkpts_df, plotf0=True):
    """solns is a list of np arrays all of nk length and the function plots them together"""
    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]']]
    f0 = fullkpts_df['k_FD']
    kpt_data = kptdata.sort_values(by=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'], ascending=True)
    ascending_inds = kpt_data.index
    plt.figure()
    for i, soln_i in enumerate(solns):
        plt.plot(kptdata['kx [1/A]'][ascending_inds], soln_i[ascending_inds],
                 '.-', markersize=5, linewidth=1, label=labels[i])
    if plotf0:
        plt.plot(kptdata['kx [1/A]'][ascending_inds], f0[ascending_inds], '.-', markersize=5,
                 linewidth=1, label='Equilibrium')
    plt.xlabel(r'$k_x$ (1/$\AA$)')
    plt.ylabel('Occupation functions')
    plt.legend()


def f2chi(f, kptsdf, c, arbfield=1.0):
    """Convert F_k from low field approximation iterative scheme into chi which is easy to plot"""
    # Since the solution we obtain from cg and from iterative scheme is F_k where chi_k = eE/kT * f0(1-f0) * F_k
    # then we need to bring these factors back in to get the right units
    f0 = np.squeeze(kptsdf['k_FD'].values)
    prefactor = arbfield * c.e / c.kb_joule / c.T * f0 * (1 - f0)
    chi = np.squeeze(f) * np.squeeze(prefactor)
    return chi


def psi2chi(psi, kptsdf):
    """Convert Psi_k from full drift iterative scheme into chi"""
    f0 = np.squeeze(kptsdf['k_FD'].values)
    prefactor = f0 * (1 - f0)
    chi = np.squeeze(psi) * np.squeeze(prefactor)
    return chi


def occupation_v_energy(chi, enk, kptsdf, c):
    npts = 4000  # number of points in the KDE
    chiax = np.zeros(npts)  # capital sigma as defined in Jin Jian's paper Eqn 3
    ftot = np.zeros(npts)
    f0ax = np.zeros(npts)
    # Need to define the energy range that I'm doing integration over
    # en_axis = np.linspace(enk.min(), enk.min() + 0.4, npts)
    en_axis = np.linspace(enk.min(), enk.max(), npts)
    dx = (en_axis.max() - en_axis.min()) / npts
    f0 = np.squeeze(kptsdf['k_FD'].values)
    spread = 50 * dx

    def gaussian(x, mu, sigma=spread):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)

    for k in range(len(enk)):
        istart = int(np.maximum(np.floor((enk[k] - en_axis[0]) / dx) - (4*spread/dx), 0))
        iend = int(np.minimum(np.floor((enk[k] - en_axis[0]) / dx) + (4*spread/dx), npts - 1))
        ftot[istart:iend] += (chi[k] + f0[k]) * gaussian(en_axis[istart:iend], enk[k])
        f0ax[istart:iend] += f0[k] * gaussian(en_axis[istart:iend], enk[k])
        chiax[istart:iend] += chi[k] * gaussian(en_axis[istart:iend], enk[k])

    ftot = f0ax + chiax

    return en_axis, ftot, chiax, f0ax


def occupation_v_energy_sep(chi, enk, kptsdf, c):
    kptsdf['kpt_mag'] = np.sqrt(kptsdf['kx [1/A]'].values**2 + kptsdf['ky [1/A]'].values**2 +
                                 kptsdf['kz [1/A]'].values**2)
    kptsdf['ingamma'] = kptsdf['kpt_mag'] < 0.3  # Boolean. In gamma if kpoint magnitude less than some amount

    npts = 4000  # number of points in the KDE
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
    spread = 75 * dx

    def gaussian(x, mu, sigma=spread):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)

    for k in range(len(g_inds)):
        istart = int(np.maximum(np.floor((g_enk[k] - g_en_axis[0]) / dx) - (4*spread/dx), 0))
        iend = int(np.minimum(np.floor((g_enk[k] - g_en_axis[0]) / dx) + (4*spread/dx), npts - 1))
        g_ftot[istart:iend] += (g_chi[k] + g_f0[k]) * gaussian(g_en_axis[istart:iend], g_enk[k])
        g_f0ax[istart:iend] += g_f0[k] * gaussian(g_en_axis[istart:iend], g_enk[k])
        g_chiax[istart:iend] += g_chi[k] * gaussian(g_en_axis[istart:iend], g_enk[k])

    for k in range(len(l_inds)):
        istart = int(np.maximum(np.floor((l_enk[k] - l_en_axis[0]) / dx) - (4 * spread / dx), 0))
        iend = int(np.minimum(np.floor((l_enk[k] - l_en_axis[0]) / dx) + (4 * spread / dx), npts - 1))
        l_ftot[istart:iend] += (l_chi[k] + l_f0[k]) * gaussian(l_en_axis[istart:iend], l_enk[k])
        l_f0ax[istart:iend] += l_f0[k] * gaussian(l_en_axis[istart:iend], l_enk[k])
        l_chiax[istart:iend] += l_chi[k] * gaussian(l_en_axis[istart:iend], l_enk[k])

    return g_en_axis, g_ftot, g_chiax, g_f0ax, l_en_axis, l_ftot, l_chiax, l_f0ax


def main():
    data_loc = '/home/peishi/nvme/k200-0.4eV/'
    chunk_loc = '/home/peishi/nvme/k200-0.4eV/chunked/'

    _, kpts_df, enk_df, qpts_df, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)

    con = preprocessing_largegrid.PhysicalConstants()
    reciplattvecs = np.concatenate((con.b1[np.newaxis, :], con.b2[np.newaxis, :], con.b3[np.newaxis, :]), axis=0)

    cart_kpts_df = preprocessing_largegrid.load_vel_data(data_loc, con)

    fbzcartkpts = preprocessing_largegrid.translate_into_fbz(cart_kpts_df.values[:, 2:5], reciplattvecs)
    fbzcartkpts = pd.DataFrame(data=fbzcartkpts, columns=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'])
    fbzcartkpts = pd.concat([cart_kpts_df[['k_inds', 'vx [m/s]', 'energy']], fbzcartkpts], axis=1)

    enk = cart_kpts_df.sort_values(by=['k_inds'])
    enk = np.array(enk['energy'])
    enk = enk - enk.min()

    cartkpts = noise_power.fermi_distribution(cart_kpts_df, fermilevel=con.mu, temp=con.T, testboltzmann=True)
    fbzcartkpts = noise_power.fermi_distribution(fbzcartkpts, fermilevel=con.mu, temp=con.T, testboltzmann=True)
    print('The fermi level is {:.2f} eV'.format(con.mu))

    # bz_3dscatter(con, cart_kpts_df, enk_df)
    # fo = bz_3dscatter(con, fbzcartkpts, enk_df)
    # plot_scattering_rates(data_loc, enk, cartkpts)

    font = {'size': 12}
    matplotlib.rc('font', **font)

    field = 1E7

    # psi_fullfield = np.load(data_loc + '/psi/psi_iter_{:.1E}_field.npy'.format(field))
    f_iter = np.load(data_loc + 'f_iterative.npy')
    f_rta = np.load(data_loc + 'f_rta.npy')
    # f_cg = np.load(data_loc + 'f_conjgrad.npy')

    # chi_fullfield = psi2chi(psi_fullfield, cartkpts)
    chi_iter = f2chi(f_iter, cartkpts, con, arbfield=field)
    chi_rta = f2chi(f_rta, cartkpts, con, arbfield=field)
    # np.save(data_loc + '/chi/chi_rta_{:.1E}'.format(field), chi_rta)

    chi_rta_1e6 = np.load(data_loc + '/chi/chi_rta_1.0E+06.npy')
    chi_rta_5e6 = np.load(data_loc + '/chi/chi_rta_5.0E+06.npy')
    chi_rta_1e7 = np.load(data_loc + '/chi/chi_rta_1.0E+07.npy')

    plot_1dim_steady_soln([chi_rta_1e6, chi_rta_5e6, chi_rta_1e7], ['1E6', '5E6', '1E7'], fbzcartkpts, plotf0=False)

    # enax, ftot_rta_enax, chi_rta_ax, f0ax = occupation_v_energy(chi_rta, enk, cartkpts, con)
    # enax, ftot_iter_enax, chi_iter_ax, f0ax = occupation_v_energy(chi_iter, enk, cartkpts, con)
    # enax, ftot_fullfield_enax, chi_fullfieldax, f0ax = occupation_v_energy(chi_fullfield, enk, cartkpts, con)

    # g_en_axis, g_ftot, g_chiax, g_f0ax, l_en_axis, l_ftot, l_chiax, l_f0ax = occupation_v_energy_sep(
    #     chi_fullfield, enk, cartkpts, con)

    # plt.figure()
    # plt.plot(enax, f0ax, label='Equilibrium (FD)')
    # plt.plot(enax, ftot_fullfield_enax, label='full iterative {:.1E} V/m'.format(field))
    # plt.plot(enax, ftot_iter_enax, label='low field iterative {:.1E} V/m'.format(field))
    # plt.plot(enax, ftot_rta_enax, label='low field rta {:.1E} V/m'.format(field))
    # plt.xlabel('Energy (ev)')
    # plt.ylabel('Total occupation (f0 + delta f)')
    # plt.legend()

    # plt.figure()
    # plt.plot(enax, chi_rta_ax, label='low field rta {:.1E} V/m'.format(field))
    # plt.plot(enax, chi_iter_ax, label='low field iterative {:.1E} V/m'.format(field))
    # plt.plot(enax, chi_fullfieldax, label='full iterative {:.1E} V/m'.format(field))
    # plt.xlim([0, 0.4])
    # plt.xlabel('Energy (eV)')
    # plt.ylabel(r'Deviational occupation ($\Delta$ f)')
    # plt.legend()

    # plt.figure()
    # plt.plot(g_en_axis, g_chiax, label='Gamma Valley')
    # plt.plot(l_en_axis, l_chiax, label='L Valley')
    # plt.xlabel('Energy (ev)')
    # plt.ylabel(r'Deviational occupation ($\Delta$ f)')
    # plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
