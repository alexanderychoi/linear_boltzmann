#!/usr/bin/env python

import preprocessing_largegrid

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import os

import plotly.offline as py
import plotly.graph_objs as go
import plotly


def bz_3dscatter(con, points, energies, useplotly=True):
    if useplotly:
        trace1 = go.Scatter3d(
            x=points['kx [1/A]'].values / (2 * np.pi / con.a),
            y=points['ky [1/A]'].values / (2 * np.pi / con.a),
            z=points['kz [1/A]'].values / (2 * np.pi / con.a),
            # x=points[:, 0] / (2 * np.pi / con.a),
            # y=points[:, 1] / (2 * np.pi / con.a),
            # z=points[:, 2] / (2 * np.pi / con.a),
            mode='markers',
            marker=dict(size=2, color=energies['energy [Ryd]'], colorscale='Rainbow', showscale=True, opacity=1)
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
    else:
        x = points['kx [1/A]'].values / (2 * np.pi / con.a)
        y = points['ky [1/A]'].values / (2 * np.pi / con.a)
        z = points['kz [1/A]'].values / (2 * np.pi / con.a)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        plt.show()


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
    # Any kpoints with angle smaller than some tolerance are considered on the path and we can plot their corresponding frequencies
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
    '''Plots electron bandstructure.

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
    '''

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
    # Any kpoints with angle smaller than some tolerance are considered on the path and we can plot their corresponding frequencies
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


def plot_scattering_rates(data_dir, energies):
    os.chdir(data_dir)
    rates = np.load('scattering_rates.npy')

    font = {'size': 14}
    matplotlib.rc('font', **font)

    plt.plot(energies, rates, '.', MarkerSize=3)
    plt.xlabel('Energy [eV]')
    plt.ylabel(r'Scattering rate [ps$^{-1}$]')
    plt.xlim([0, 0.4])
    # plt.savefig('plot_scattering_rates.png')


def main():
    # data_loc = '/home/peishi/nvme/k100-0.3eV/'
    # chunk_loc = '/home/peishi/nvme/k100-0.3eV/chunked/'
    data_loc = '/home/peishi/nvme/k200-0.4eV/'
    chunk_loc = '/home/peishi/nvme/k200-0.4eV/chunked/'
    _, kpts_df, enk_df, qpts_df, enq_df = preprocessing_largegrid.loadfromfile(data_loc, matrixel=False)

    con = preprocessing_largegrid.PhysicalConstants()
    reciplattvecs = np.concatenate((con.b1[np.newaxis, :], con.b2[np.newaxis, :], con.b3[np.newaxis, :]), axis=0)

    cart_kpts_df = preprocessing_largegrid.load_vel_data(data_loc, con)

    fbzcartkpts = preprocessing_largegrid.translate_into_fbz(cart_kpts_df.to_numpy()[:, 2:5], reciplattvecs)
    fbzcartkpts = pd.DataFrame(data=fbzcartkpts, columns=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'])

    enk = cart_kpts_df.sort_values(by=['k_inds'])
    enk = np.array(enk['energy'])
    enk = enk - enk.min()

    # plt.plot(enk, '.')
    # bz_3dscatter(con, cart_kpts_df, enk_df)
    # bz_3dscatter(con, fbzcartkpts, enk_df)
    plot_scattering_rates(data_loc, enk)

    plt.show()


if __name__ == '__main__':
    main()
