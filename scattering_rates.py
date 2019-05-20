#!/usr/bin/env python

import data_processing

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go
import plotly
#plotly.tools.set_credentials_file(username='AYChoi', api_key='ZacDa7fKo8hfiELPfs57')
plotly.tools.set_credentials_file(username='AlexanderYChoi', api_key='VyLt05wzc89iXwSC82FO')


def coupling_matrix_calc(g_df):
    """
    This function takes a list of k-point indices and returns the Fermi-distributions and energies associated with each k-point on that list. The Fermi distributions are calculated with respect to a particular chemical potential.
    Parameters:
    -----------

    abs_g_df : pandas dataframe containing:

        k_inds : vector_like, shape (n,1)
        Index of k point (pre-collision)

        q_inds : vector_like, shape (n,1)
        Index of q point

        k+q_inds : vector_like, shape (n,1)
        Index of k point (post-collision)

        m_band : vector_like, shape (n,1)
        Band index of post-collision state

        n_band : vector_like, shape (n,1)
        Band index of pre-collision state

        im_mode : vector_like, shape (n,1)
        Polarization of phonon mode

        g_element : vector_like, shape (n,1)
        E-ph matrix element

        k_fermi_dist : vector_like, shape (n,1)
        Fermi distribution of pre collision state

        k+q_fermi_dist : vector_like, shape (n,1)
        Fermi distribution of post collision state

        k_energy : vector_like, shape (n,1)
        Energy of the pre collision state

        k+q_energy : vector_like, shape (n,1)
        Energy of the post collision state


    T : scalar
    Lattice temperature in Kelvin

    Returns:
    --------

    """
    # Physical constants
    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
    kb = 1.38064852 * 10 ** (-23);  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    h = 1.0545718 * 10 ** (-34)

    g_df_ems = g_df.loc[(g_df['collision_state'] == -1)].copy(deep=True)
    g_df_abs = g_df.loc[(g_df['collision_state'] == 1)].copy(deep=True)

    g_df_ems['weight'] = np.multiply(
        np.multiply((g_df_ems['BE'].values + 1 - g_df_ems['k+q_FD'].values), g_df_ems['g_element'].values),
        g_df_ems['gaussian']) / 13.6056980659
    g_df_abs['weight'] = np.multiply(
        np.multiply((g_df_abs['BE'].values + g_df_abs['k+q_FD'].values), g_df_abs['g_element'].values),
        g_df_abs['gaussian']) / 13.6056980659

    abs_sr = g_df_abs.groupby(['k_inds', 'k+q_inds'])['weight'].agg('sum')
    summed_abs_df = abs_sr.to_frame().reset_index()

    ems_sr = g_df_ems.groupby(['k_inds', 'k+q_inds'])['weight'].agg('sum')
    summed_ems_df = ems_sr.to_frame().reset_index()

    return summed_abs_df, summed_ems_df


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def main():
    check_symmetric(np.abs(collision_array))


if __name__ == '__main__':
    main()
