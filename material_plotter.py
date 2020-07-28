import numpy as np
import constants as c
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import pandas as pd
import noise_solver
import occupation_plotter
from matplotlib.font_manager import FontProperties
import preprocessing
from mpl_toolkits.mplot3d import Axes3D


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
            x=points['kx [1/A]'].values / (2 * np.pi / c.alat),
            y=points['ky [1/A]'].values / (2 * np.pi / c.alat),
            z=points['kz [1/A]'].values / (2 * np.pi / c.alat),

            mode='markers',
            marker=dict(size=2, color=colors, colorscale='Rainbow', showscale=True, opacity=1)
        )
        if icind:
            trace2 = go.Scatter3d(
                x=icr_df['kx [1/A]'].values / (2 * np.pi / c.alat),
                y=icr_df['ky [1/A]'].values / (2 * np.pi / c.alat),
                z=icr_df['kz [1/A]'].values / (2 * np.pi / c.alat),

                mode='markers',
                marker=dict(size=2, color='black', opacity=1)
            )

            trace3 = go.Scatter3d(
                x=icl_df['kx [1/A]'].values / (2 * np.pi / c.alat),
                y=icl_df['ky [1/A]'].values / (2 * np.pi / c.alat),
                z=icl_df['kz [1/A]'].values / (2 * np.pi / c.alat),

                mode='markers',
                marker=dict(size=2, color='#7f7f7f', opacity=1)
            )

        b1edge = 0.5 * c.b1 / (2 * np.pi / c.alat)
        vector1 = go.Scatter3d(x=[0, b1edge[0]], y=[0, b1edge[1]], z=[0, b1edge[2]],
                               marker=dict(size=1,color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))
        b2edge = 0.5 * c.b2 / (2 * np.pi / c.alat)
        vector2 = go.Scatter3d(x=[0, b2edge[0]], y=[0, b2edge[1]], z=[0, b2edge[2]],
                               marker=dict(size=1,color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))
        b3edge = 0.5 * c.b3 / (2 * np.pi / c.alat)
        vector3 = go.Scatter3d(x=[0, b3edge[0]], y=[0, b3edge[1]], z=[0, b3edge[2]],
                               marker=dict(size=1,color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))
        xedge = -0.5 * (c.b1 + c.b3) / (2 * np.pi / c.alat)
        vector4 = go.Scatter3d(x=[0, xedge[0]], y=[0, xedge[1]], z=[0, xedge[2]],
                               marker=dict(size=1, color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))
        yedge = 0.5 * (c.b2 + c.b3) / (2 * np.pi / c.alat)
        vector5 = go.Scatter3d(x=[0, yedge[0]], y=[0, yedge[1]], z=[0, yedge[2]],
                               marker=dict(size=1, color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))
        zedge = 0.5 * (c.b1 + c.b2) / (2 * np.pi / c.alat)
        vector6 = go.Scatter3d(x=[0, zedge[0]], y=[0, zedge[1]], z=[0, zedge[2]],
                               marker=dict(size=1, color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))
        ledge = 0.5 * (c.b1 + c.b2 + c.b3) / (2 * np.pi / c.alat)
        vector7 = go.Scatter3d(x=[0, ledge[0]], y=[0, ledge[1]], z=[0, ledge[2]],
                               marker=dict(size=1, color="rgb(84,48,5)"),
                               line=dict(color="rgb(84,48,5)", width=5))

        data = [trace1, vector1, vector2, vector3, vector4, vector5, vector6, vector7]
        if icind:
            data = [trace1, trace2,trace3, vector1, vector2, vector3, vector4, vector5, vector6, vector7]

        layout = go.Layout(
            scene=dict(
                xaxis=dict(
                    title='kx [normalized]'),
                yaxis=dict(
                    title='ky [normalized]'),
                zaxis=dict(
                    title=r'kz [normalized]'), ))
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename='bz_scatter.html')
        return fig
    else:
        x = points['kx [1/A]'].values / (2 * np.pi / c.alat)
        y = points['ky [1/A]'].values / (2 * np.pi / c.alat)
        z = points['kz [1/A]'].values / (2 * np.pi / c.alat)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        return ax


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
    # utilities.check_matrix_properties(scm)
    g_inds, l_inds, x_inds = utilities.split_valleys(df,False)
    if pp.simpleBool:
        rates = (-1) * np.diag(scm) * scmfac * 1E-12
    else:
        chi2psi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
        rates = (-1) * np.diag(scm) * scmfac * 1E-12 / chi2psi
    plt.figure()
    # plt.plot(df['energy [eV]'], rates, '.', MarkerSize=3)
    plt.plot(df.loc[g_inds,'energy [eV]'], rates[g_inds], '.', MarkerSize=3, label=r'$\Gamma$')
    plt.plot(df.loc[l_inds,'energy [eV]'], rates[l_inds], '.', MarkerSize=3, label='L')
    if pp.getX:
        plt.plot(df.loc[x_inds, 'energy [eV]'], rates[x_inds], '.', MarkerSize=3, label=r'X')
    plt.xlabel('Energy [eV]')
    plt.ylabel(r'Scattering rate [ps$^{-1}$]')
    plt.title(pp.title_str)
    plt.legend()


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

    enkonly = np.array(enk['energy [eV]'])[:, np.newaxis]
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
    plt.ylabel('Energy in eV')


def plot_energy_kx(df):
    """Plots the scattering rates by pulling them from the on-diagonal of the simple scattering matrix
    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
    Returns:
        Nothing. Just the plots.
    """
    plt.figure()
    plt.plot(df['kx [1/A]'], df['energy [eV]'], '.', MarkerSize=3)
    plt.ylabel('Energy [eV]')
    plt.xlabel(r'kx [1/A]')
    plt.title(pp.title_str)
    plt.legend()


if __name__ == '__main__':
    fields = pp.fieldVector
    freq = pp.freqGHz
    preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    plot_scattering_rates(electron_df)
    bz_3dscatter(electron_df, True, False)
    plot_bandstructure(electron_df, electron_df)
    plot_energy_kx(electron_df)
    # plot_diffusion(electron_df, fields, freq)
    # plot_L_valley_drift(electron_df,fields)
    # bz_3dscatter(electron_df, True, False)
    print(c.e*utilities.calculate_density(electron_df) * 18500/ 100 ** 2 * 2 * c.kb_joule* pp.T / c.Vuc)
    plt.show()