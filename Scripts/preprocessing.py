import numpy as np
import itertools
import plotly
import pandas as pd
from tqdm import tqdm, trange

# Load in the data from native .txt files

def load_vel_data(dirname):
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
    cart_kpts_df['kx [2pi/alat]'] = cart_kpts_df['kx [2pi/alat]'].values * 2 * np.pi / a
    cart_kpts_df['ky [2pi/alat]'] = cart_kpts_df['ky [2pi/alat]'].values * 2 * np.pi / a
    cart_kpts_df['kz [2pi/alat]'] = cart_kpts_df['kz [2pi/alat]'].values * 2 * np.pi / a
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






