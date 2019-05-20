#!/usr/bin/env python

import data_processing

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go
import plotly
#plotly.tools.set_credentials_file(username='AYChoi', api_key='ZacDa7fKo8hfiELPfs57')
plotly.tools.set_credentials_file(username='AlexanderYChoi', api_key='VyLt05wzc89iXwSC82FO')


def bz_3dscatter():
    trace1 = go.Scatter3d(
        x=cart_kpts_df['kx [1/A]'].values / (2 * np.pi / (a)),
        y=cart_kpts_df['ky [1/A]'].values / (2 * np.pi / (a)),
        z=cart_kpts_df['kz [1/A]'].values / (2 * np.pi / (a)),
        mode='markers',
        marker=dict(
            size=2,
            color=enk_df['energy [Ryd]'],
            colorscale='Rainbow',
            showscale=True,
            opacity=1
        )
    )

    trace2 = go.Scatter

    data = [trace1]
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title='kx', titlefont=dict(family='Oswald, monospace', size=18)),
            yaxis=dict(
                title='ky', titlefont=dict(family='Oswald, monospace', size=18)),
            zaxis=dict(
                title='kz', titlefont=dict(family='Oswald, monospace', size=18)), ))
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='simple-3d-scatter')

    trace1 = go.Scatter3d(
        x=cart_kpts_df['kx [1/A]'].values / (2 * np.pi / (a)),
        y=cart_kpts_df['ky [1/A]'].values / (2 * np.pi / (a)),
        z=cart_kpts_df['kz [1/A]'].values / (2 * np.pi / (a)),
        mode='markers',
        marker=dict(
            size=2,
            color=cart_kpts_df['v_mag [m/s]'],
            colorscale='Rainbow',
            showscale=True,
            opacity=1
        )
    )

    trace2 = go.Scatter

    trace1 = go.Scatter3d(
        x=cart_kpts_df['kx [1/A]'].values / (2 * np.pi / a),
        y=cart_kpts_df['ky [1/A]'].values / (2 * np.pi / a),
        z=cart_kpts_df['kz [1/A]'].values / (2 * np.pi / a),
        mode='markers',
        marker=dict(
            size=2,
            color=cart_kpts_df['k_inds'],
            colorscale='Rainbow',
            showscale=True,
            opacity=1
        )
    )

    trace2 = go.Scatter

    data = [trace1]
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title='kx', titlefont=dict(family='Oswald, monospace', size=18)),
            yaxis=dict(
                title='ky', titlefont=dict(family='Oswald, monospace', size=18)),
            zaxis=dict(
                title='kz', titlefont=dict(family='Oswald, monospace', size=18)), ))
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='simple-3d-scatter')

    trace1 = go.Scatter3d(
        x=edit_cart_qpts_df['kx [1/A]'].values,
        y=edit_cart_qpts_df['ky [1/A]'].values,
        z=edit_cart_qpts_df['kz [1/A]'].values,
        mode='markers',
        marker=dict(
            size=2,
            opacity=1
        )
    )

    data = [trace1]
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title='kx', titlefont=dict(family='Oswald, monospace', size=18)),
            yaxis=dict(
                title='ky', titlefont=dict(family='Oswald, monospace', size=18)),
            zaxis=dict(
                title='kz', titlefont=dict(family='Oswald, monospace', size=18)), ))
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='simple-3d-scatter')

def plot_dispersion(kpts, enk):
    '''Plots phonon dispersion.

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
            plt.plot(veck, theseenergies, '.', color='C0')
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


def main():
    data_processing.loadfromfile()
    plot_dispersion(edit_cart_qpts_df, enq_df)
    plot_bandstructure(cart_kpts_df, enk_df)

    plt.show()

if __name__ == '__main__':
    main()