import numpy as np
import itertools
import plotly
import os
import pandas as pd
from tqdm import tqdm, trange

# Load in the data from native .txt files

def load_vel_data(dirname,cons):
    """Dirname is the name of the directory where the .VEL file is stored.
    The k-points are given in Cartesian coordinates and are not yet shifted
    back into the first Brillouin Zone.
    The result of this function is a Pandas DataFrame containing the columns:
    [k_inds][bands][energy (eV)][kx (1/A)][ky (1/A)[kz (1/A)]"""

    kvel = pd.read_csv(dirname, sep='\t', header=None, skiprows=[0, 1, 2])
    kvel.columns = ['0']
    kvel_array = kvel['0'].values
    new_kvel_array = np.zeros((len(kvel_array), 10))
    for i1 in trange(len(kvel_array)):
        new_kvel_array[i1, :] = kvel_array[i1].split()
    kvel_df = pd.DataFrame(data=new_kvel_array,
                           columns=['k_inds', 'bands', 'energy', 'kx [2pi/alat]', 'ky [2pi/alat]', 'kz [2pi/alat]',
                                    'vx_dir', 'vy_dir', 'vz_dir', 'v_mag [m/s]'])
    kvel_df[['k_inds']] = kvel_df[['k_inds']].apply(pd.to_numeric, downcast='integer') # downcast indices to integer
    cart_kpts_df = kvel_df.copy(deep=True)
    cart_kpts_df['kx [2pi/alat]'] = cart_kpts_df['kx [2pi/alat]'].values * 2 * np.pi / cons.a
    cart_kpts_df['ky [2pi/alat]'] = cart_kpts_df['ky [2pi/alat]'].values * 2 * np.pi / cons.a
    cart_kpts_df['kz [2pi/alat]'] = cart_kpts_df['kz [2pi/alat]'].values * 2 * np.pi / cons.a
    cart_kpts_df.columns = ['k_inds', 'bands', 'energy', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]', 'vx_dir', 'vy_dir',
                            'vz_dir', 'v_mag [m/s]']
    cart_kpts_df = cart_kpts_df.drop(['bands'], axis=1)

    return cart_kpts_df

def load_enq_data(dirname):
    """Dirname is the name of the directory where the .ENQ file is stored.
    im_mode is the corresponding phonon polarization.
    The result of this function is a Pandas DataFrame containing the columns:
    [q_inds][im_mode][energy (Ryd)]"""

    enq = pd.read_csv(dirname, sep='\t', header=None)
    enq.columns = ['0']
    enq_array = enq['0'].values
    new_enq_array = np.zeros((len(enq_array), 3))
    for i1 in trange(len(enq_array)):
        new_enq_array[i1, :] = enq_array[i1].split()

    enq_df = pd.DataFrame(data=new_enq_array, columns=['q_inds', 'im_mode', 'energy [Ryd]'])
    enq_df[['q_inds', 'im_mode']] = enq_df[['q_inds', 'im_mode']].apply(pd.to_numeric, downcast='integer')

    return enq_df

def load_qpt_data(dirname):
    """Dirname is the name of the directory where the .QPT file is stored.
    The result of this function is a Pandas DataFrame containing the columns:
    [q_inds][b1][b2][b3]"""

    qpts = pd.read_csv(dirname, sep='\t', header=None)
    qpts.columns = ['0']
    qpts_array = qpts['0'].values
    new_qpt_array = np.zeros((len(qpts_array), 4))

    for i1 in trange(len(qpts_array)):
        new_qpt_array[i1, :] = qpts_array[i1].split()

    qpts_df = pd.DataFrame(data=new_qpt_array, columns=['q_inds', 'b1', 'b2', 'b3'])
    qpts_df[['q_inds']] = qpts_df[['q_inds']].apply(pd.to_numeric, downcast='integer')

    return qpts_df

def load_g_data(dirname):
    """Dirname is the name of the directory where the .eph_matrix file is stored.
    The result of this function is a Pandas DataFrame containing the columns:
    [k_inds][q_inds][k+q_inds][m_band][n_band][im_mode][g_element]"""
    data = pd.read_csv('gaas.eph_matrix', sep='\t', header=None, skiprows=(0, 1))
    data.columns = ['0']
    data_array = data['0'].values
    new_array = np.zeros((len(data_array), 7))
    for i1 in trange(len(data_array)):
        new_array[i1, :] = data_array[i1].split()

    g_df = pd.DataFrame(data=new_array,
                        columns=['k_inds', 'q_inds', 'k+q_inds', 'm_band', 'n_band', 'im_mode', 'g_element'])
    g_df[['k_inds', 'q_inds', 'k+q_inds', 'm_band', 'n_band', 'im_mode']] = g_df[
        ['k_inds', 'q_inds', 'k+q_inds', 'm_band', 'n_band', 'im_mode']].apply(pd.to_numeric, downcast='integer')

    g_df = g_df.drop(["m_band", "n_band"], axis=1)
    return g_df

# Now process the data to be in a more convenient form for calculations
def fermi_distribution(g_df,mu,T):
    """This function is designed to take a Pandas DataFrame containing e-ph data and return
    the Fermi-Dirac distribution associated with both the pre- and post- collision states.
    The distribution is calculated with respect to a given chemical potential, mu"""

    # Physical constants
    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
    kb = 1.38064852 * 10 ** (-23);  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    g_df['k_FD'] = (np.exp((g_df['k_en [eV]'].values * e - mu * e) / (kb * T)) + 1) ** (-1)
    g_df['k+q_FD'] = (np.exp((g_df['k+q_en [eV]'].values * e - mu * e) / (kb * T)) + 1) ** (-1)

    return g_df


def bose_distribution(g_df,T):
    """This function is designed to take a Pandas DataFrame containing e-ph data and return
    the Bose-Einstein distribution associated with the mediating phonon mode."""
    # Physical constants
    e = 1.602*10**(-19) # fundamental electronic charge [C]
    kb = 1.38064852*10**(-23); # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    g_df['BE'] = (np.exp((g_df['q_en [eV]'].values*e)/(kb*T)) - 1)**(-1)
    return g_df

def bosonic_processing(g_df,enq_df,T):
    """This function takes the e-ph DataFrame and assigns a phonon energy to each collision
    and calculates the Bose-Einstein distribution"""
    # Physical constants
    e = 1.602 * 10 ** (-19)  # fundamental electronic charge [C]
    kb = 1.38064852 * 10 ** (-23);  # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    modified_g_df = g_df.copy(deep=True)
    modified_g_df.set_index(['q_inds', 'im_mode'], inplace=True)
    modified_g_df = modified_g_df.sort_index()
    modified_enq_df = enq_df.copy(deep=True)
    modified_enq_df.set_index(['q_inds', 'im_mode'], inplace=True)
    modified_enq_df = modified_enq_df.sort_index()
    modified_enq_df = modified_enq_df.loc[modified_g_df.index.unique()]

    modified_enq_df = modified_enq_df.reset_index()
    modified_enq_df = modified_enq_df.sort_values(['q_inds', 'im_mode'], ascending=True)
    modified_enq_df = modified_enq_df[['q_inds', 'im_mode', 'energy [Ryd]']]
    modified_enq_df['q_id'] = modified_enq_df.groupby(['q_inds', 'im_mode']).ngroup()
    g_df['q_id'] = g_df.sort_values(['q_inds', 'im_mode'], ascending=True).groupby(['q_inds', 'im_mode']).ngroup()

    g_df['q_en [eV]'] = modified_enq_df['energy [Ryd]'].values[g_df['q_id'].values] * 13.6056980659

    g_df = bose_distribution(g_df, T)

    return g_df

def fermionic_processing(g_df,cart_kpts_df,mu,T):
    """This function takes the e-ph DataFrame and assigns the relevant pre and post collision energies
    as well as the Fermi-Dirac distribution associated with both states."""

    # Pre-collision
    modified_g_df_k = g_df.copy(deep=True)
    modified_g_df_k.set_index(['k_inds'], inplace=True)
    modified_g_df_k = modified_g_df_k.sort_index()

    modified_k_df = cart_kpts_df.copy(deep=True)
    modified_k_df.set_index(['k_inds'], inplace=True)
    modified_k_df = modified_k_df.sort_index()
    modified_k_df = modified_k_df.loc[modified_g_df_k.index.unique()]

    modified_k_df = modified_k_df.reset_index()
    modified_k_df = modified_k_df.sort_values(['k_inds'], ascending=True)
    modified_k_df = modified_k_df[['k_inds', 'energy', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]']]

    modified_k_df['k_id'] = modified_k_df.groupby(['k_inds']).ngroup()
    g_df['k_id'] = g_df.sort_values(['k_inds'], ascending=True).groupby(['k_inds']).ngroup()
    g_df['k_en [eV]'] = modified_k_df['energy'].values[g_df['k_id'].values]

    g_df['kx [1/A]'] = modified_k_df['kx [1/A]'].values[g_df['k_id'].values]
    g_df['ky [1/A]'] = modified_k_df['ky [1/A]'].values[g_df['k_id'].values]
    g_df['kz [1/A]'] = modified_k_df['kz [1/A]'].values[g_df['k_id'].values]

    # Post-collision
    modified_g_df_kq = g_df.copy(deep=True)
    modified_g_df_kq.set_index(['k_inds'], inplace=True)
    modified_g_df_kq = modified_g_df_kq.sort_index()

    modified_k_df = cart_kpts_df.copy(deep=True)
    modified_k_df.set_index(['k_inds'], inplace=True)
    modified_k_df = modified_k_df.sort_index()
    modified_k_df = modified_k_df.loc[modified_g_df_kq.index.unique()]

    modified_k_df = modified_k_df.reset_index()
    modified_k_df = modified_k_df.sort_values(['k_inds'], ascending=True)
    modified_k_df = modified_k_df[['k_inds', 'energy', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]']]

    modified_k_df['k+q_id'] = modified_k_df.groupby(['k_inds']).ngroup()
    g_df['k+q_id'] = g_df.sort_values(['k+q_inds'], ascending=True).groupby(['k+q_inds']).ngroup()
    g_df['k+q_en [eV]'] = modified_k_df['energy'].values[g_df['k+q_id'].values]

    g_df['kqx [1/A]'] = modified_k_df['kx [1/A]'].values[g_df['k+q_id'].values]
    g_df['kqy [1/A]'] = modified_k_df['ky [1/A]'].values[g_df['k+q_id'].values]
    g_df['kqz [1/A]'] = modified_k_df['kz [1/A]'].values[g_df['k+q_id'].values]

    g_df = fermi_distribution(g_df, mu, T)

    g_df = g_df.drop(['k_id', 'k+q_id'], axis=1)

    return g_df

def gaussian_weight(g_df,n):
    """This function assigns the value of the delta function of the energy conservation
    approimated by a Gaussian with broadening n"""

    energy_delta_ems = g_df['k_en [eV]'].values - g_df['k+q_en [eV]'].values - g_df['q_en [eV]'].values
    energy_delta_abs = g_df['k_en [eV]'].values - g_df['k+q_en [eV]'].values + g_df['q_en [eV]'].values

    g_df['abs_gaussian'] = 1 / np.sqrt(np.pi) * 1 / n * np.exp(-(energy_delta_abs / n) ** 2)
    g_df['ems_gaussian'] = 1 / np.sqrt(np.pi) * 1 / n * np.exp(-(energy_delta_ems / n) ** 2)

    return g_df

def populate_reciprocals(g_df,b):
    """The g^2 elements are invariant under substitution of the pre-and post- collision indices.
    Therefore, the original e-ph matrix DataFrame only contains half the set, since the other
    half is obtainable. This function populates the appropriate reciprocal elements."""

    modified_g_df = g_df.copy(deep=True)

    flipped_inds = g_df['k_inds'] > g_df['k+q_inds']
    modified_g_df.loc[flipped_inds, 'k_inds'] = g_df.loc[flipped_inds, 'k+q_inds']
    modified_g_df.loc[flipped_inds, 'k+q_inds'] = g_df.loc[flipped_inds, 'k_inds']

    modified_g_df.loc[flipped_inds, 'k_FD'] = g_df.loc[flipped_inds, 'k+q_FD']
    modified_g_df.loc[flipped_inds, 'k+q_FD'] = g_df.loc[flipped_inds, 'k_FD']

    modified_g_df.loc[flipped_inds, 'k_en [eV]'] = g_df.loc[flipped_inds, 'k+q_en [eV]']
    modified_g_df.loc[flipped_inds, 'k+q_en [eV]'] = g_df.loc[flipped_inds, 'k_en [eV]']

    modified_g_df.loc[flipped_inds, 'kqx [1/A]'] = g_df.loc[flipped_inds, 'kx [1/A]']
    modified_g_df.loc[flipped_inds, 'kqy [1/A]'] = g_df.loc[flipped_inds, 'ky [1/A]']
    modified_g_df.loc[flipped_inds, 'kqz [1/A]'] = g_df.loc[flipped_inds, 'kz [1/A]']
    modified_g_df.loc[flipped_inds, 'kx [1/A]'] = g_df.loc[flipped_inds, 'kqx [1/A]']
    modified_g_df.loc[flipped_inds, 'ky [1/A]'] = g_df.loc[flipped_inds, 'kqy [1/A]']
    modified_g_df.loc[flipped_inds, 'kz [1/A]'] = g_df.loc[flipped_inds, 'kqz [1/A]']

    modified_g_df['k_pair_id'] = modified_g_df.groupby(['k_inds', 'k+q_inds']).ngroup()

    reverse_df = modified_g_df.copy(deep=True)

    reverse_df['k_inds'] = modified_g_df['k+q_inds']
    reverse_df['k+q_inds'] = modified_g_df['k_inds']

    reverse_df['k_FD'] = modified_g_df['k+q_FD']
    reverse_df['k+q_FD'] = modified_g_df['k_FD']

    reverse_df['k_en [eV]'] = modified_g_df['k+q_en [eV]']
    reverse_df['k+q_en [eV]'] = modified_g_df['k_en [eV]']

    reverse_df['kqx [1/A]'] = modified_g_df['kx [1/A]']
    reverse_df['kqy [1/A]'] = modified_g_df['ky [1/A]']
    reverse_df['kqz [1/A]'] = modified_g_df['kz [1/A]']
    reverse_df['kx [1/A]'] = modified_g_df['kqx [1/A]']
    reverse_df['ky [1/A]'] = modified_g_df['kqy [1/A]']
    reverse_df['kz [1/A]'] = modified_g_df['kqz [1/A]']

    full_g_df = modified_g_df.append(reverse_df)

    full_g_df = gaussian_weight(full_g_df, b)

    return full_g_df


class physical_constants:
    """A class with constants to be passed into any method"""
    # Physical parameter definition
    a = 5.556                        # Lattice constant for GaAs [A]
    kb = 1.38064852*10**(-23)        # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    T = 300                          # Lattice temperature [K]
    e = 1.602*10**(-19)              # Fundamental electronic charge [C]
    mu = 5.780                       # Chemical potential [eV]
    b = 8/1000                       # Gaussian broadening [eV]


def loadfromfile():
    """Takes the raw text files and converts them into useful dataframes."""
    if os.path.isfile('matrix_el.h5'):
        g_df = pd.read_hdf('matrix_el.h5', key='df')
        print('loaded matrix elements from hdf5')
    else:
        data = pd.read_csv('gaas.eph_matrix', sep='\t', header=None, skiprows=(0,1))
        data.columns = ['0']
        data_array = data['0'].values
        new_array = np.zeros((len(data_array),7))
        for i1 in trange(len(data_array)):
            new_array[i1,:] = data_array[i1].split()

        g_df = pd.DataFrame(data=new_array, columns=['k_inds','q_inds','k+q_inds','m_band','n_band','im_mode','g_element'])
        g_df[['k_inds', 'q_inds', 'k+q_inds', 'm_band', 'n_band', 'im_mode']] = g_df[['k_inds', 'q_inds', 'k+q_inds', 'm_band','n_band', 'im_mode']].apply(pd.to_numeric, downcast='integer')
        g_df = g_df.drop(["m_band", "n_band"],axis=1)
        g_df.to_hdf('matrix_el.h5', key='df')


def cartesian_q_points(qpts_df, con):
    """Given a dataframe containing indexed q-points in terms of the crystal lattice vector, return the dataframe with cartesian q coordinates.
    Parameters:
    -----------
    con :  instance of the physical_constants class
    qpts_df : pandas dataframe containing:

        q_inds : vector_like, shape (n,1)
        Index of q point

        kx : vector_like, shape (n,1)
        x-coordinate in momentum space [1/A]

        ky : vector_like, shape (n,1)
        y-coordinate in momentum space [1/A]

        kz : vector_like, shape (n,1)
        z-coordinate in momentum space [1/A]

    For FCC lattice, use the momentum space primitive vectors as per:
    http://lampx.tugraz.at/~hadley/ss1/bzones/fcc.php

    b1 = 2 pi/a (kx - ky + kz)
    b2 = 2 pi/a (kx + ky - kz)
    b3 = 2 pi/a (-kx + ky + kz)

    Returns:
    --------
    cart_kpts_df : pandas dataframe containing:

        q_inds : vector_like, shape (n,1)
        Index of q point

        kx : vector_like, shape (n,1)
        x-coordinate in Cartesian momentum space [1/m]

        ky : vector_like, shape (n,1)
        y-coordinate in Cartesian momentum space [1/m]

        kz : vector_like, shape (n,1)
        z-coordinate in Cartesian momentum space [1/m]
    """
    cartesian_df = pd.DataFrame(columns=['q_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]'])

    con1 = pd.DataFrame(columns=['kx [1/A]', 'ky [1/A]', 'kz [1/A]'])
    con1['kx [1/A]'] = np.ones(len(qpts_df)) * -1
    con1['ky [1/A]'] = np.ones(len(qpts_df)) * -1
    con1['kz [1/A]'] = np.ones(len(qpts_df)) * 1

    con2 = con1.copy(deep=True)
    con2['kx [1/A]'] = con2['kx [1/A]'].values * -1
    con2['ky [1/A]'] = con2['ky [1/A]'].values * -1

    con3 = con1.copy(deep=True)
    con3['ky [1/A]'] = con2['ky [1/A]'].values
    con3['kz [1/A]'] = con3['kz [1/A]'].values * -1

    cartesian_df['kx [1/A]'] = np.multiply(qpts_df['b1'].values, (con1['kx [1/A]'].values)) + np.multiply(
        qpts_df['b2'].values, (con2['kx [1/A]'].values)) + np.multiply(qpts_df['b3'].values, (con3['kx [1/A]'].values))
    cartesian_df['ky [1/A]'] = np.multiply(qpts_df['b1'].values, (con1['ky [1/A]'].values)) + np.multiply(
        qpts_df['b2'].values, (con2['ky [1/A]'].values)) + np.multiply(qpts_df['b3'].values, (con3['ky [1/A]'].values))
    cartesian_df['kz [1/A]'] = np.multiply(qpts_df['b1'].values, (con1['kz [1/A]'].values)) + np.multiply(
        qpts_df['b2'].values, (con2['kz [1/A]'].values)) + np.multiply(qpts_df['b3'].values, (con3['kz [1/A]'].values))

    cartesian_df['q_inds'] = qpts_df['q_inds'].values

    cartesian_df_edit = cartesian_df.copy(deep=True)

    qx_plus = cartesian_df['kx [1/A]'] > 0.5
    qx_minus = cartesian_df['kx [1/A]'] < -0.5

    qy_plus = cartesian_df['ky [1/A]'] > 0.5
    qy_minus = cartesian_df['ky [1/A]'] < -0.5

    qz_plus = cartesian_df['kz [1/A]'] > 0.5
    qz_minus = cartesian_df['kz [1/A]'] < -0.5

    cartesian_df_edit.loc[qx_plus, 'kx [1/A]'] = cartesian_df.loc[qx_plus, 'kx [1/A]'] - 1
    cartesian_df_edit.loc[qx_minus, 'kx [1/A]'] = cartesian_df.loc[qx_minus, 'kx [1/A]'] + 1

    cartesian_df_edit.loc[qy_plus, 'ky [1/A]'] = cartesian_df.loc[qy_plus, 'ky [1/A]'] - 1
    cartesian_df_edit.loc[qy_minus, 'ky [1/A]'] = cartesian_df.loc[qy_minus, 'ky [1/A]'] + 1

    cartesian_df_edit.loc[qz_plus, 'kz [1/A]'] = cartesian_df.loc[qz_plus, 'kz [1/A]'] - 1
    cartesian_df_edit.loc[qz_minus, 'kz [1/A]'] = cartesian_df.loc[qz_minus, 'kz [1/A]'] + 1

    return cartesian_df, cartesian_df_edit


def shift_into_fbz(qpts_df,kvel_df, con):
    """Shifting k vectors back into BZ."""

    kvel_edit = kvel_df.copy(deep=True)

    # Shift the points back into the first BZ
    kx_plus = kvel_df['kx [2pi/alat]'] > 0.5
    kx_minus = kvel_df['kx [2pi/alat]'] < -0.5

    ky_plus = kvel_df['ky [2pi/alat]'] > 0.5
    ky_minus = kvel_df['ky [2pi/alat]'] < -0.5

    kz_plus = kvel_df['kz [2pi/alat]'] > 0.5
    kz_minus = kvel_df['kz [2pi/alat]'] < -0.5

    kvel_edit.loc[kx_plus, 'kx [2pi/alat]'] = kvel_df.loc[kx_plus, 'kx [2pi/alat]'] -1
    kvel_edit.loc[kx_minus,'kx [2pi/alat]'] = kvel_df.loc[kx_minus,'kx [2pi/alat]'] +1

    kvel_edit.loc[ky_plus,'ky [2pi/alat]'] = kvel_df.loc[ky_plus,'ky [2pi/alat]'] -1
    kvel_edit.loc[ky_minus,'ky [2pi/alat]'] = kvel_df.loc[ky_minus,'ky [2pi/alat]'] +1

    kvel_edit.loc[kz_plus,'kz [2pi/alat]'] = kvel_df.loc[kz_plus,'kz [2pi/alat]'] -1
    kvel_edit.loc[kz_minus,'kz [2pi/alat]'] = kvel_df.loc[kz_minus,'kz [2pi/alat]'] +1

    kvel_df = kvel_edit.copy(deep=True)
    kvel_df.head()

    cart_kpts_df = kvel_df.copy(deep=True)
    cart_kpts_df['kx [2pi/alat]'] = cart_kpts_df['kx [2pi/alat]'].values*2*np.pi/con.a
    cart_kpts_df['ky [2pi/alat]'] = cart_kpts_df['ky [2pi/alat]'].values*2*np.pi/con.a
    cart_kpts_df['kz [2pi/alat]'] = cart_kpts_df['kz [2pi/alat]'].values*2*np.pi/con.a

    cart_kpts_df.columns = ['k_inds', 'bands', 'energy', 'kx [1/A]', 'ky [1/A]','kz [1/A]', 'vx_dir', 'vy_dir', 'vz_dir', 'v_mag [m/s]']

    # Making Cartesian qpoints
    cart_qpts_df,edit_cart_qpts_df = cartesian_q_points(qpts_df)
    return cart_qpts_df,edit_cart_qpts_df



def main():

    con = physical_constants()
    enq_df = load_enq_data('gaas.enq')
    print('Phonon energies loaded')
    qpt_df = load_qpt_data('gaas.qpts')
    print('\n')
    print('Phonon qpts loaded')
    cart_kpts_df = load_vel_data('gaas.vel',con)
    print('Electron kpts loaded')
    print('Electron energies loaded')
    g_df = loadfromfile()
    print('E-ph data loaded')

    # At this point, the processed data are:
    # e-ph matrix elements = g_df['k_inds','q_inds','k+q_inds','m_band','n_band','im_mode']
    # k-points = kpts_df['k_inds','b1','b2','b3']
    # q-points = qpts_df['q_inds','b1','b2','b3']
    # k-energy = enk_df['k_inds','band_inds','energy [Ryd]']
    # q-energy = enq_df['q_inds','im_mode','energy [Ryd]']

    cartesian_df, cartesian_df_edit = cartesian_q_points(qpt_df, con)

if __name__ == '__main__':
    main()



