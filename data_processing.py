#!/usr/bin/env python
"""Data processing module for the electron-phonon collision matrix

This is meant to be a frills-free calculation of the electron-phonon collision matrix utilizing the data from
Jin Jian Zhou for GaAs.
"""

import numpy as np
import plotting

# Image processing tools
import skimage
import skimage.filters

import os
import pandas as pd
import scipy.optimize
import scipy.stats as st
import numba
import itertools

from numpy.linalg import inv

from tqdm import tqdm, trange
from scipy import special, optimize
from scipy import integrate

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import plotly.offline as py9
import plotly.graph_objs as go
import plotly
plotly.tools.set_credentials_file(username='peishi', api_key='KQcCXB5wcgednHNjbNqd')
# plotly.tools.set_credentials_file(username='AlexanderYChoi', api_key='VyLt05wzc89iXwSC82FO')


class PhysicalConstants:
    """A class with constants to be passed into any method"""
    # Physical parameters
    a = 5.5563606                    # Lattice constant for GaAs [Angstrom]
    kb = 1.38064852*10**(-23)        # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    T = 300                          # Lattice temperature [K]
    e = 1.602*10**(-19)              # Fundamental electronic charge [C]
    mu = 5.780                       # Chemical potential [eV]
    b = 8/1000                       # Gaussian broadening [eV]

    # Lattice vectors
    a1 = np.array([-2.7781803, 0.0000000, 2.7781803])
    a2 = np.array([+0.0000000, 2.7781803, 2.7781803])
    a3 = np.array([-2.7781803, 2.7781803, 0.0000000])

    b1 = np.array([-1.1308095, -1.1308095, +1.1308095])
    b2 = np.array([+1.1308095, +1.1308095, +1.1308095])
    b3 = np.array([-1.1308095, +1.1308095, -1.1308095])
    # b1 = np.array([+1.1308095, -1.1308095, +1.1308095])
    # b2 = np.array([+1.1308095, +1.1308095, -1.1308095])
    # b3 = np.array([-1.1308095, +1.1308095, +1.1308095])


def loadfromfile(matrixel=True):
    """Takes the raw text files and converts them into useful dataframes."""
    if matrixel:  # maybe sometimes don't need matrix elements, just need the other data
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
    else:
        g_df = []

    # Importing electron k points
    kpts = pd.read_csv('gaas.kpts', sep='\t', header=None)
    kpts.columns = ['0']
    kpts_array = kpts['0'].values
    new_kpt_array = np.zeros((len(kpts_array), 4))
    for i1 in trange(len(kpts_array)):
        new_kpt_array[i1, :] = kpts_array[i1].split()

    kpts_df = pd.DataFrame(data=new_kpt_array, columns=['k_inds', 'b1', 'b2', 'b3'])
    kpts_df[['k_inds']] = kpts_df[['k_inds']].apply(pd.to_numeric, downcast='integer')

    # Import electron energy library
    enk = pd.read_csv('gaas.enk', sep='\t',header= None)
    enk.columns = ['0']
    enk_array = enk['0'].values
    new_enk_array = np.zeros((len(enk_array),3))
    for i1 in trange(len(enk_array)):
        new_enk_array[i1,:] = enk_array[i1].split()

    enk_df = pd.DataFrame(data=new_enk_array,columns = ['k_inds','band_inds','energy [Ryd]'])
    enk_df[['k_inds','band_inds']] = enk_df[['k_inds','band_inds']].apply(pd.to_numeric,downcast = 'integer')

    # Import phonon energy library
    enq = pd.read_csv('gaas.enq', sep='\t',header= None)
    enq.columns = ['0']
    enq_array = enq['0'].values
    new_enq_array = np.zeros((len(enq_array),3))
    for i1 in trange(len(enq_array)):
        new_enq_array[i1,:] = enq_array[i1].split()

    enq_df = pd.DataFrame(data=new_enq_array,columns = ['q_inds','im_mode','energy [Ryd]'])
    enq_df[['q_inds','im_mode']] = enq_df[['q_inds','im_mode']].apply(pd.to_numeric,downcast = 'integer')

    # Import phonon q-point index
    qpts = pd.read_csv('gaas.qpts', sep='\t',header= None)
    qpts.columns = ['0']
    qpts_array = qpts['0'].values
    new_qpt_array = np.zeros((len(qpts_array),4))
    for i1 in trange(len(qpts_array)):
        new_qpt_array[i1,:] = qpts_array[i1].split()

    qpts_df = pd.DataFrame(data=new_qpt_array, columns=['q_inds', 'b1', 'b2', 'b3'])
    qpts_df[['q_inds']] = qpts_df[['q_inds']].apply(pd.to_numeric, downcast='integer')

    # Import phonon energies
    enq = pd.read_csv('gaas.enq', sep='\t',header= None)
    enq.columns = ['0']
    enq_array = enq['0'].values
    new_enq_array = np.zeros((len(enq_array),3))
    for i1 in trange(len(enq_array)):
        new_enq_array[i1,:] = enq_array[i1].split()

    enq_df = pd.DataFrame(data=new_enq_array,columns = ['q_inds','im_mode','energy [Ryd]'])
    enq_df[['q_inds','im_mode']] = enq_df[['q_inds','im_mode']].apply(pd.to_numeric,downcast = 'integer')

    return g_df, kpts_df, enk_df, qpts_df, enq_df


def cartesian_q_points(qpts_df, con):
    """Given a dataframe containing indexed q-points in terms of the crystal lattice vector, return the dataframe with cartesian q coordinates.
    Parameters:
    -----------
    con :  instance of the PhysicalConstants class

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
    cartesian_df = pd.DataFrame(columns = ['q_inds','kx [1/A]','ky [1/A]','kz [1/A]'])
    
    con1 = pd.DataFrame(columns = ['kx [1/A]','ky [1/A]','kz [1/A]'])
    con1['kx [1/A]'] = np.ones(len(qpts_df))*-1
    con1['ky [1/A]'] = np.ones(len(qpts_df))*-1
    con1['kz [1/A]'] = np.ones(len(qpts_df))*1

    con2 = con1.copy(deep=True)
    con2['kx [1/A]'] = con2['kx [1/A]'].values*-1
    con2['ky [1/A]'] = con2['ky [1/A]'].values*-1

    con3 = con1.copy(deep=True)
    con3['ky [1/A]'] = con2['ky [1/A]'].values
    con3['kz [1/A]'] = con3['kz [1/A]'].values*-1

    cartesian_df['kx [1/A]'] = np.multiply(qpts_df['b1'].values,(con1['kx [1/A]'].values)) + np.multiply(qpts_df['b2'].values,(con2['kx [1/A]'].values)) + np.multiply(qpts_df['b3'].values,(con3['kx [1/A]'].values))
    cartesian_df['ky [1/A]'] = np.multiply(qpts_df['b1'].values,(con1['ky [1/A]'].values)) + np.multiply(qpts_df['b2'].values,(con2['ky [1/A]'].values)) + np.multiply(qpts_df['b3'].values,(con3['ky [1/A]'].values))
    cartesian_df['kz [1/A]'] = np.multiply(qpts_df['b1'].values,(con1['kz [1/A]'].values)) + np.multiply(qpts_df['b2'].values,(con2['kz [1/A]'].values)) + np.multiply(qpts_df['b3'].values,(con3['kz [1/A]'].values))

    cartesian_df['q_inds'] = qpts_df['q_inds'].values
    
    cartesian_df_edit = cartesian_df.copy(deep=True)

    qx_plus = cartesian_df['kx [1/A]'] > 0.5
    qx_minus = cartesian_df['kx [1/A]'] < -0.5

    qy_plus = cartesian_df['ky [1/A]'] > 0.5
    qy_minus = cartesian_df['ky [1/A]'] < -0.5

    qz_plus = cartesian_df['kz [1/A]'] > 0.5
    qz_minus = cartesian_df['kz [1/A]'] < -0.5

    cartesian_df_edit.loc[qx_plus,'kx [1/A]'] = cartesian_df.loc[qx_plus,'kx [1/A]'] - 1
    cartesian_df_edit.loc[qx_minus,'kx [1/A]'] = cartesian_df.loc[qx_minus,'kx [1/A]'] + 1

    cartesian_df_edit.loc[qy_plus,'ky [1/A]'] = cartesian_df.loc[qy_plus,'ky [1/A]'] - 1
    cartesian_df_edit.loc[qy_minus,'ky [1/A]'] = cartesian_df.loc[qy_minus,'ky [1/A]'] + 1

    cartesian_df_edit.loc[qz_plus,'kz [1/A]'] = cartesian_df.loc[qz_plus,'kz [1/A]'] - 1
    cartesian_df_edit.loc[qz_minus,'kz [1/A]'] = cartesian_df.loc[qz_minus,'kz [1/A]'] + 1
    
    return cartesian_df,cartesian_df_edit


def cartesian_kpts(con):
    """This directly takes Cartesian kpoints from the velocity file"""
    kvel = pd.read_csv('gaas.vel', sep='\t', header=None, skiprows=[0, 1, 2])
    kvel.columns = ['0']
    kvel_array = kvel['0'].values
    new_kvel_array = np.zeros((len(kvel_array), 10))
    for i1 in trange(len(kvel_array)):
        new_kvel_array[i1, :] = kvel_array[i1].split()

    kvel_df = pd.DataFrame(data=new_kvel_array, columns=['k_inds','bands','energy','kx [2pi/alat]','ky [2pi/alat]','kz [2pi/alat]','vx_dir','vy_dir','vz_dir','v_mag [m/s]'])
    kvel_df[['k_inds']] = kvel_df[['k_inds']].apply(pd.to_numeric, downcast='integer')

    kvel_edit = kvel_df.copy(deep=True)

    # # Shift the points back into the first BZ
    # kx_plus = kvel_df['kx [2pi/alat]'] > 0.5
    # kx_minus = kvel_df['kx [2pi/alat]'] < -0.5
    #
    # ky_plus = kvel_df['ky [2pi/alat]'] > 0.5
    # ky_minus = kvel_df['ky [2pi/alat]'] < -0.5
    #
    # kz_plus = kvel_df['kz [2pi/alat]'] > 0.5
    # kz_minus = kvel_df['kz [2pi/alat]'] < -0.5
    #
    # kvel_edit.loc[kx_plus,'kx [2pi/alat]'] = kvel_df.loc[kx_plus,'kx [2pi/alat]'] -1
    # kvel_edit.loc[kx_minus,'kx [2pi/alat]'] = kvel_df.loc[kx_minus,'kx [2pi/alat]'] +1
    #
    # kvel_edit.loc[ky_plus,'ky [2pi/alat]'] = kvel_df.loc[ky_plus,'ky [2pi/alat]'] -1
    # kvel_edit.loc[ky_minus,'ky [2pi/alat]'] = kvel_df.loc[ky_minus,'ky [2pi/alat]'] +1
    #
    # kvel_edit.loc[kz_plus,'kz [2pi/alat]'] = kvel_df.loc[kz_plus,'kz [2pi/alat]'] -1
    # kvel_edit.loc[kz_minus,'kz [2pi/alat]'] = kvel_df.loc[kz_minus,'kz [2pi/alat]'] +1

    kvel_df = kvel_edit.copy(deep=True)

    cart_kpts_df = kvel_df.copy(deep=True)
    cart_kpts_df['kx [2pi/alat]'] = cart_kpts_df['kx [2pi/alat]'].values*2*np.pi/con.a
    cart_kpts_df['ky [2pi/alat]'] = cart_kpts_df['ky [2pi/alat]'].values*2*np.pi/con.a
    cart_kpts_df['kz [2pi/alat]'] = cart_kpts_df['kz [2pi/alat]'].values*2*np.pi/con.a

    cart_kpts_df.columns = ['k_inds', 'bands', 'energy', 'kx [1/A]', 'ky [1/A]','kz [1/A]', 'vx_dir', 'vy_dir', 'vz_dir', 'v_mag [m/s]']

    return cart_kpts_df, kvel


def fermi_distribution(g_df,mu,T):
    """
    This function takes a list of k-point indices and returns the Fermi-distributions and energies associated with each k-point on that list. The Fermi distributions are calculated with respect to a particular chemical potential.      
    Parameters:
    -----------
    
    g_df : pandas dataframe containing:
    
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
        
        k_energy : vector_like, shape (n,1)
        Energy of the pre collision state
        
        k+q_energy : vector_like, shape (n,1)
        Energy of the post collision state
        
        
    mu : scalar
    Chemical potential of electronic states [eV]
    
    T : scalar
    Lattice temperature in Kelvin
    
    Returns:
    --------
    
    g_df : pandas dataframe containing:

        ...
        k_fermi_dist : vector_like, shape (n,1)
        Fermi distribution of pre collision state
        
        k+q_fermi_dist : vector_like, shape (n,1)
        Fermi distribution of post collision state
         
    """
    # Physical constants    
    e = 1.602*10**(-19) # fundamental electronic charge [C]
    kb = 1.38064852*10**(-23); # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    g_df['k_FD'] = (np.exp((g_df['k_en [eV]'].values*e - mu*e)/(kb*T)) + 1)**(-1)
    g_df['k+q_FD'] = (np.exp((g_df['k+q_en [eV]'].values*e - mu*e)/(kb*T)) + 1)**(-1)

    return g_df


def bose_distribution(g_df,T):
    """
    This function takes a list of q-point indices and returns the Bose-Einstein distributions associated with each q-point on that list.    
    Parameters:
    -----------
    
    g_df : pandas dataframe containing:
    
        ...
    
    T : scalar
    Lattice temperature in Kelvin
    
    Returns:
    --------
    
    g_df : pandas dataframe containing:

        ...
        
        BE : vector_like, shape (n,1)
        Bose-einstein distribution
         
    """
    # Physical constants    
    e = 1.602*10**(-19) # fundamental electronic charge [C]
    kb = 1.38064852*10**(-23); # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    g_df['BE'] = (np.exp((g_df['q_en [eV]'].values*e)/(kb*T)) - 1)**(-1)
    return g_df


def bosonic_processing(g_df,enq_df,T):
    """
    This function takes the g dataframe and assigns a phonon energy from the relevant phonon library to each collision and the appropriate Bose-Einstein distribution.
    -----------
    
    g_df : pandas dataframe containing:
    
    
    Returns:
    --------
    
    g_df : pandas dataframe containing:
    
    ...
        BE : vector_like, shape (n,1)
        Bose-Einstein distribution of the phonon mediating a collision
        
        q_en [eV] : vector_like, shape (n,1)
        The energy of the phonon mode mediating a collision
         
    """
    
    # Physical constants
    e = 1.602*10**(-19) # fundamental electronic charge [C]
    kb = 1.38064852*10**(-23); # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    
    modified_g_df = g_df.copy(deep=True)
    modified_g_df.set_index(['q_inds', 'im_mode'], inplace=True)
    modified_g_df = modified_g_df.sort_index()
    modified_enq_df = enq_df.copy(deep=True)
    modified_enq_df.set_index(['q_inds', 'im_mode'], inplace=True)
    modified_enq_df = modified_enq_df.sort_index()
    modified_enq_df = modified_enq_df.loc[modified_g_df.index.unique()]
    
    modified_enq_df = modified_enq_df.reset_index()
    modified_enq_df = modified_enq_df.sort_values(['q_inds','im_mode'],ascending=True)
    modified_enq_df = modified_enq_df[['q_inds','im_mode','energy [Ryd]']]
    modified_enq_df['q_id'] = modified_enq_df.groupby(['q_inds','im_mode']).ngroup()
    g_df['q_id'] = g_df.sort_values(['q_inds','im_mode'],ascending=True).groupby(['q_inds','im_mode']).ngroup()
    
    g_df['q_en [eV]'] = modified_enq_df['energy [Ryd]'].values[g_df['q_id'].values]*13.6056980659
    
    g_df = bose_distribution(g_df,T)
    
    return g_df


def fermionic_processing(g_df,cart_kpts_df,mu,T,b):
    """
    This function takes the g dataframe and assigns an electron energy from the relevant electron library to the pre and post collision states and the appropriate Fermi-Diract distributions.
    -----------
    
    g_df : pandas dataframe containing:
    
    ...
    
    Returns:
    --------
    
    g_df : pandas dataframe containing:
    
    ...
        k_fermi_dist : vector_like, shape (n,1)
        Fermi distribution of pre collision state
        
        k+q_fermi_dist : vector_like, shape (n,1)
        Fermi distribution of post collision state
         
    """

    # Pre-collision
    modified_g_df_k = g_df.copy(deep=True)
    modified_g_df_k.set_index(['k_inds'], inplace=True)
    modified_g_df_k = modified_g_df_k.sort_index()

    modified_k_df = cart_kpts_df.copy(deep=True)
    modified_k_df.set_index(['k_inds'], inplace=True)
    modified_k_df = modified_k_df.sort_index()
    modified_k_df = modified_k_df.loc[modified_g_df_k.index.unique()]
    
    modified_k_df = modified_k_df.reset_index()
    modified_k_df = modified_k_df.sort_values(['k_inds'],ascending=True)
    modified_k_df = modified_k_df[['k_inds','energy','kx [1/A]','ky [1/A]','kz [1/A]']]
    
    modified_k_df['k_id'] = modified_k_df.groupby(['k_inds']).ngroup()
    g_df['k_id'] = g_df.sort_values(['k_inds'],ascending=True).groupby(['k_inds']).ngroup()   
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
    modified_k_df = modified_k_df.sort_values(['k_inds'],ascending=True)
    modified_k_df = modified_k_df[['k_inds','energy','kx [1/A]','ky [1/A]','kz [1/A]']]
    
    modified_k_df['k+q_id'] = modified_k_df.groupby(['k_inds']).ngroup()
    g_df['k+q_id'] = g_df.sort_values(['k+q_inds'],ascending=True).groupby(['k+q_inds']).ngroup()   
    g_df['k+q_en [eV]'] = modified_k_df['energy'].values[g_df['k+q_id'].values]
    
    g_df['kqx [1/A]'] = modified_k_df['kx [1/A]'].values[g_df['k+q_id'].values]
    g_df['kqy [1/A]'] = modified_k_df['ky [1/A]'].values[g_df['k+q_id'].values]
    g_df['kqz [1/A]'] = modified_k_df['kz [1/A]'].values[g_df['k+q_id'].values]

    
    abs_inds = g_df['k_en [eV]'] < g_df['k+q_en [eV]'] #absorbed indices
    ems_inds = g_df['k_en [eV]'] > g_df['k+q_en [eV]'] #emission indices

    
    g_df.loc[abs_inds,'collision_state'] = 1
    g_df.loc[ems_inds,'collision_state'] = -1
    
    g_df = fermi_distribution(g_df,mu, T)
    
    g_df = g_df.drop(['k_id','k+q_id'],axis=1)
    
    g_df = gaussian_weight(g_df,b)
    
    return g_df


def gaussian_weight(g_df,n):
    """
    This function assigns the value of the delta function approximated by a Gaussian with broadening n.
    
    Parameters:
    -----------
    
    g_df : pandas dataframe containing:

        ...
            
    n : scalar
    Broadening of Gaussian in eV
    
    Returns:
    --------
    """
    abs_inds = g_df['collision_state'] == 1 #absorbed indices
    ems_inds = g_df['collision_state'] == -1 #emission indices
    
    energy_delta_ems = g_df.loc[ems_inds,'k_en [eV]'].values - g_df.loc[ems_inds,'k+q_en [eV]'].values - g_df.loc[ems_inds,'q_en [eV]'].values
    energy_delta_abs = g_df.loc[abs_inds,'k_en [eV]'].values - g_df.loc[abs_inds,'k+q_en [eV]'].values + g_df.loc[abs_inds,'q_en [eV]'].values
    
    g_df.loc[abs_inds,'gaussian'] = 1/np.sqrt(np.pi)*1/n*np.exp(-(energy_delta_abs/n)**2)
    g_df.loc[ems_inds,'gaussian'] = 1/np.sqrt(np.pi)*1/n*np.exp(-(energy_delta_ems/n)**2)
    
    return g_df


def populate_reciprocals(g_df):
    """
    The g^2 elements are invariant under substitution of k and k'. Jin-Jian provided the minimal set, that is for a given k-pair linked through a particular collision 
    and characterized by a say an emission, the reciprocal absorbtion is not included. Here we repopulate these states.
    -----------
    
    g_df : pandas dataframe containing:
    
    ...
    
    Returns:
    --------
    
    g_df : pandas dataframe containing:
    
    ...
         
    """

    modified_g_df = g_df.copy(deep=True)

    flipped_inds = g_df['k_inds']>g_df['k+q_inds']
    modified_g_df.loc[flipped_inds,'k_inds'] = g_df.loc[flipped_inds,'k+q_inds']
    modified_g_df.loc[flipped_inds,'k+q_inds'] = g_df.loc[flipped_inds,'k_inds']

    modified_g_df.loc[flipped_inds,'k_FD'] = g_df.loc[flipped_inds,'k+q_FD']
    modified_g_df.loc[flipped_inds,'k+q_FD'] = g_df.loc[flipped_inds,'k_FD']

    modified_g_df.loc[flipped_inds,'k_en [eV]'] = g_df.loc[flipped_inds,'k+q_en [eV]']
    modified_g_df.loc[flipped_inds,'k+q_en [eV]'] = g_df.loc[flipped_inds,'k_en [eV]']

    modified_g_df.loc[flipped_inds,'collision_state'] = g_df.loc[flipped_inds,'collision_state']*-1
    
    modified_g_df.loc[flipped_inds,'kqx [1/A]'] = g_df.loc[flipped_inds,'kx [1/A]']
    modified_g_df.loc[flipped_inds,'kqy [1/A]'] = g_df.loc[flipped_inds,'ky [1/A]']
    modified_g_df.loc[flipped_inds,'kqz [1/A]'] = g_df.loc[flipped_inds,'kz [1/A]']
    modified_g_df.loc[flipped_inds,'kx [1/A]'] = g_df.loc[flipped_inds,'kqx [1/A]']
    modified_g_df.loc[flipped_inds,'ky [1/A]'] = g_df.loc[flipped_inds,'kqy [1/A]']
    modified_g_df.loc[flipped_inds,'kz [1/A]'] = g_df.loc[flipped_inds,'kqz [1/A]']
    
    modified_g_df['k_pair_id'] = modified_g_df.groupby(['k_inds','k+q_inds']).ngroup()


    reverse_df = modified_g_df.copy(deep=True)

    reverse_df['k_inds'] = modified_g_df['k+q_inds']
    reverse_df['k+q_inds'] = modified_g_df['k_inds']

    reverse_df['k_FD'] = modified_g_df['k+q_FD']
    reverse_df['k+q_FD'] = modified_g_df['k_FD']

    reverse_df['k_en [eV]'] = modified_g_df['k+q_en [eV]']
    reverse_df['k+q_en [eV]'] = modified_g_df['k_en [eV]']

    reverse_df['collision_state'] = modified_g_df['collision_state']*-1
    
    reverse_df['kqx [1/A]'] = modified_g_df['kx [1/A]']
    reverse_df['kqy [1/A]'] = modified_g_df['ky [1/A]']
    reverse_df['kqz [1/A]'] = modified_g_df['kz [1/A]']
    reverse_df['kx [1/A]'] = modified_g_df['kqx [1/A]']
    reverse_df['ky [1/A]'] = modified_g_df['kqy [1/A]']
    reverse_df['kz [1/A]'] = modified_g_df['kqz [1/A]']

    full_g_df = modified_g_df.append(reverse_df)

    return full_g_df


# In[7]:


# g_df = bosonic_processing(g_df,enq_df,T)
# g_df = fermionic_processing(g_df,cart_kpts_df,mu,T,b)


# In[8]:


# np.sum(g_df['collision_state'] == 1),np.sum(g_df['collision_state'] == -1),


# In[10]:


# full_g_df = populate_reciprocals(g_df)


# In[11]:


# del g_df


# In[13]:


# np.sum(full_g_df['collision_state'] == 1),np.sum(full_g_df['collision_state'] == -1),


# In[14]:


# full_g_df = full_g_df[['k_inds', 'q_inds', 'k+q_inds', 'im_mode', 'g_element', 'q_id',
#        'q_en [eV]', 'BE', 'k_en [eV]', 'k+q_en [eV]',
#        'k_FD', 'k+q_FD','collision_state', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]', 'kqx [1/A]',
#        'kqy [1/A]', 'kqz [1/A]', 'k_pair_id','gaussian']]


# In[64]:


# collisionless_df = full_g_df.loc[full_g_df['collision_state'].isnull()]
# low_collisionless_df = collisionless_df.loc[collisionless_df['k_inds']<collisionless_df['k+q_inds']]
# high_collisionless_df = collisionless_df.loc[collisionless_df['k_inds']>collisionless_df['k+q_inds']]


# In[78]:


# low_collisionless_df['collision_state'] = 1
# high_collisionless_df['collision_state'] = -1
#
# collisionless_df.loc[collisionless_df['k_inds']<collisionless_df['k+q_inds']] = low_collisionless_df
# collisionless_df.loc[collisionless_df['k_inds']>collisionless_df['k+q_inds']] = high_collisionless_df
#
# full_g_df.loc[full_g_df['collision_state'].isnull()] = collisionless_df


# In[80]:


# del collisionless_df
# del low_collisionless_df
# del high_collisionless_df


# In[93]:


# full_g_df= gaussian_weight(full_g_df,b)


# In[109]:


# full_g_df.head()


# In[110]:


def scattering_rate(g_df):
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
    e = 1.602*10**(-19) # fundamental electronic charge [C]
    kb = 1.38064852*10**(-23); # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    h = 1.0545718*10**(-34)
        
    g_df_ems = g_df.loc[(g_df['collision_state'] == -1)].copy(deep=True)
    g_df_abs = g_df.loc[(g_df['collision_state'] == 1)].copy(deep=True)
    
    g_df_ems['weight'] = np.multiply(np.multiply(g_df_ems['BE'].values + 1 - g_df_ems['k+q_FD'].values,g_df_ems['g_element'].values),g_df_ems['gaussian'])/13.6056980659
    g_df_abs['weight'] = np.multiply(np.multiply((g_df_abs['BE'].values + g_df_abs['k+q_FD'].values),g_df_abs['g_element'].values),g_df_ems['gaussian'])/13.6056980659
    
    
    abs_sr = g_df_abs.groupby(['k_inds'])['weight'].agg('sum')*2*np.pi*2.418*10**(17)*10**(-12)/len(np.unique(g_df['q_id'].values))
    abs_scattering = abs_sr.to_frame().reset_index()
    
    ems_sr = g_df_ems.groupby(['k_inds'])['weight'].agg('sum')*2*np.pi*2.418*10**(17)*10**(-12)/len(np.unique(g_df['q_id'].values))
    ems_scattering = ems_sr.to_frame().reset_index()
    
    return ems_scattering,abs_scattering


# In[111]:


# ems_scattering,abs_scattering = scattering_rate(full_g_df)


# In[112]:


# abs_scattering_array = np.zeros(len(np.unique(enk_df['k_inds'])))
# ems_scattering_array = np.zeros(len(np.unique(enk_df['k_inds'])))
# abs_scattering_array[abs_scattering['k_inds'].values-1] = abs_scattering['weight'].values
# ems_scattering_array[ems_scattering['k_inds'].values-1] = ems_scattering['weight'].values


# In[120]:


# plt.rcParams.update({'font.size': 40})
# plt.rcParams.update({'lines.linewidth': 3.5})
#
# fig = plt.figure(figsize=(12,10))
# ax = plt.gca()


# plt.scatter((enk_df['energy [Ryd]'].values-enk_df['energy [Ryd]'].min())*13.6056980659,(abs_scattering_array+ems_scattering_array),c = 'Red')
# #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.ylabel('Scattering Rate [1/ps]')
# plt.xlabel('Energy [eV]')
# #plt.legend()
# plt.ylim((-0.1,20.1))
# plt.show()
# fig.savefig('test.png', bbox_inches='tight')


# In[123]:


# len(full_g_df)


# In[136]:


# np.sum(full_g_df['k_en [eV]'] > (full_g_df['k_en [eV]'].min()+ 0.25))/len(full_g_df)


# In[138]:


# del ems_scattering
# del abs_scattering


# In[209]:


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
    e = 1.602*10**(-19) # fundamental electronic charge [C]
    kb = 1.38064852*10**(-23); # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    h = 1.0545718*10**(-34)

    g_df_ems = g_df.loc[(g_df['collision_state'] == -1)].copy(deep=True)
    g_df_abs = g_df.loc[(g_df['collision_state'] == 1)].copy(deep=True)

    g_df_ems['weight'] = np.multiply(np.multiply((g_df_ems['BE'].values + 1 - g_df_ems['k+q_FD'].values),g_df_ems['g_element'].values),g_df_ems['gaussian'])/13.6056980659
    g_df_abs['weight'] = np.multiply(np.multiply((g_df_abs['BE'].values + g_df_abs['k+q_FD'].values),g_df_abs['g_element'].values),g_df_abs['gaussian'])/13.6056980659

    abs_sr = g_df_abs.groupby(['k_inds', 'k+q_inds'])['weight'].agg('sum')
    summed_abs_df = abs_sr.to_frame().reset_index()

    ems_sr = g_df_ems.groupby(['k_inds', 'k+q_inds'])['weight'].agg('sum')
    summed_ems_df = ems_sr.to_frame().reset_index()

    return summed_abs_df,summed_ems_df


# In[210]:


# summed_abs_df,summed_ems_df = coupling_matrix_calc(full_g_df)


# In[211]:


# abs_array = np.zeros((len(np.unique(enk_df['k_inds'])),len(np.unique(enk_df['k_inds']))))
# ems_array = np.zeros((len(np.unique(enk_df['k_inds'])),len(np.unique(enk_df['k_inds']))))
#
# abs_array[summed_abs_df['k_inds'].values-1,summed_abs_df['k+q_inds'].values-1] = summed_abs_df['weight'].values
# ems_array[summed_ems_df['k_inds'].values-1,summed_ems_df['k+q_inds'].values-1] = summed_ems_df['weight'].values


# In[308]:


# collision_array = (np.transpose(abs_array+ems_array)-(abs_array+ems_array))*2*np.pi*2.418*10**(17)*10**(-12)/len(np.unique(full_g_df['q_id'].values))


# In[324]:


# sorted_indices = cart_kpts_df.sort_values(['kx [1/A]','ky [1/A]','kz [1/A]'],ascending=True)['k_inds'].values-1
# i = np.argsort(sorted_indices)
# switch1 = collision_array[:,i]
# switch2 = switch1[i,:]
#
# plt.set_cmap('inferno')
# fig = plt.figure(figsize=(20, 12.2))
# plt.rcParams.update({'font.size': 20})
#
# ax = fig.add_subplot(111)
# ax.set_title('Collision Matrix (kpt ascending)')
# plt.imshow(np.abs(switch2),origin = 'lower')
# ax.set_aspect('equal')
# plt.xlabel('k index')
# plt.ylabel('k_p index')
#
# cax = fig.add_axes([0.12, 0.1, 0.75, 0.82])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# cbar = plt.colorbar(orientation='vertical')
# cbar.set_label('Coupling Rate [fs^-1]')
# fig.savefig('test.png', bbox_inches='tight')
# plt.show()


# In[319]:


# sorted_indices = cart_kpts_df.sort_values(['energy'],ascending=True)['k_inds'].values-1
# i = np.argsort(sorted_indices)
# switch1 = collision_array[:,i]
# switch2 = switch1[i,:]


# In[323]:

# plt.set_cmap('inferno')
# fig = plt.figure(figsize=(20, 12.2))
# plt.rcParams.update({'font.size': 20})
#
# ax = fig.add_subplot(111)
# ax.set_title('Collision Matrix (energy ascending)')
# plt.imshow(np.abs(switch2),origin = 'lower')
# ax.set_aspect('equal')
# plt.xlabel('k index')
# plt.ylabel('k_p index')
#
# cax = fig.add_axes([0.12, 0.1, 0.75, 0.82])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# cbar = plt.colorbar(orientation='vertical')
# cbar.set_label('Coupling Rate [fs^-1]')
# fig.savefig('test.png', bbox_inches='tight')
# plt.show()


# In[315]:


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


# In[322]:


# check_symmetric(np.abs(collision_array))


# In[317]:


# np.sum(collision_array,axis=0)

def vectorbasis2cartesian(coords, vecs):
    """Transform any coordinates written in lattice vector basis into Cartesian coordinates

    Given that the inputs are correct, then the transformation is a simple matrix multiply

    Parameters:
        vecs (numpy array): Array of vectors where the rows are the basis vectors given in Cartesian basis
        coords (numpy array): Array of coordinates where each row is an independent point, so that column 1 corresponds
            to the amount of the basis vector in row 1 of vecs, column 2 is amount of basis vector in row 2 of vecs...

    Returns:
        cartcoords (array): same size as coords but in Cartesian coordinates
    """

    cartcoords = np.matmul(coords, vecs)
    return cartcoords


def translate_into_fbz(coords, rlv):
    """Manually translate coordinates back into first Brillouin zone

    The way we do this is by finding all the planes that form the FBZ boundary and the vectors that are associated
    with these planes. Since the FBZ is centered on Gamma, the position vectors of the high symmetry points are also
    vectors normal to the plane. Once we have these vectors, we find the distance between a given point (u) and
    a plane (n) using the dot product of the difference vector (u-n). And if the distance is positive, then translate
    back into the FBZ.

    Args:
        rlv: numpy array of vectors where the rows are the reciprocal lattice vectors given in Cartesian basis
        coords: numpy array of coordinates where each row is a point. For N points, coords is N x 3

    Returns:
        fbzcoords:
    """
    # First, find all the vectors defining the boundary
    b1, b2, b3 = rlv[0, :], rlv[1, :], rlv[2, :]
    b1pos = 0.5 * b1[:, np.newaxis]
    b2pos = 0.5 * b2[:, np.newaxis]
    b3pos = 0.5 * b3[:, np.newaxis]
    lpos = 0.5 * (b1 + b2 + b3)[:, np.newaxis]
    b1neg = -1 * b1pos
    b2neg = -1 * b2pos
    b3neg = -1 * b3pos
    lneg = -1 * lpos
    xpos = -0.5 * (b1 + b3)[:, np.newaxis]
    ypos = 0.5 * (b2 + b3)[:, np.newaxis]
    zpos = 0.5 * (b1 + b2)[:, np.newaxis]
    xneg = -1 * xpos
    yneg = -1 * ypos
    zneg = -1 * zpos

    # Place them into octants to avoid problems when finding points
    # (naming is based on positive or negative for coordinate so octpmm means x+ y- z-. p=plus, m=minus)
    vecs_ppp = np.concatenate((b2pos, xpos, ypos, zpos), axis=1)[:, :, np.newaxis]
    vecs_ppm = np.concatenate((b1neg, xpos, ypos, zneg), axis=1)[:, :, np.newaxis]
    vecs_pmm = np.concatenate((lneg, xpos, yneg, zneg), axis=1)[:, :, np.newaxis]
    vecs_mmm = np.concatenate((b2neg, xneg, yneg, zneg), axis=1)[:, :, np.newaxis]
    vecs_mmp = np.concatenate((b1pos, xneg, yneg, zpos), axis=1)[:, :, np.newaxis]
    vecs_mpp = np.concatenate((lpos, xneg, ypos, zpos), axis=1)[:, :, np.newaxis]
    vecs_mpm = np.concatenate((b3pos, xneg, ypos, zneg), axis=1)[:, :, np.newaxis]
    vecs_pmp = np.concatenate((b3neg, xpos, yneg, zpos), axis=1)[:, :, np.newaxis]
    # Construct matrix which is 3 x 4 x 8 where we have 3 Cartesian coordinates, 4 vectors per octant, and 8 octants
    allvecs = np.concatenate((vecs_ppp, vecs_ppm, vecs_pmm, vecs_mmm, vecs_mmp, vecs_mpp, vecs_mpm, vecs_pmp), axis=2)

    # Since the number of points in each octant is not equal, can't create array of similar shape. Instead the 'octant'
    # array below is used as a boolean map where 1 (true) indicates positive, and 0 (false) indicates negative
    octants = np.array([[1, 1, 1],
                        [1, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [1, 0, 1]])

    fbzcoords = np.copy(coords)
    exitvector = np.zeros((8, 1))
    iteration = 0
    while not np.all(exitvector):  # don't exit until all octants have points inside
        exitvector = np.zeros((8, 1))
        for i in range(8):
            oct_vecs = allvecs[:, :, i]
            whichoct = octants[i, :]
            if whichoct[0]:
                xbool = fbzcoords[:, 0] > 0
            else:
                xbool = fbzcoords[:, 0] < 0
            if whichoct[1]:
                ybool = fbzcoords[:, 1] > 0
            else:
                ybool = fbzcoords[:, 1] < 0
            if whichoct[2]:
                zbool = fbzcoords[:, 2] > 0
            else:
                zbool = fbzcoords[:, 2] < 0
            octindex = np.logical_and(np.logical_and(xbool, ybool), zbool)
            octcoords = fbzcoords[octindex, :]
            allplanes = 0
            for j in range(oct_vecs.shape[1]):
                diffvec = octcoords[:, :] - np.tile(oct_vecs[:, j], (octcoords.shape[0], 1))
                dist2plane = np.dot(diffvec, oct_vecs[:, j]) / np.linalg.norm(oct_vecs[:, j])
                outside = dist2plane[:] > 0
                if np.any(outside):
                    octcoords[outside, :] = octcoords[outside, :] - \
                                            (2 * np.tile(oct_vecs[:, j], (np.count_nonzero(outside), 1)))
                    # Times 2 because the vectors that define FBZ are half of the full recip latt vectors
                    # print('number outside this plane is %d' % np.count_nonzero(outside))
                else:
                    allplanes += 1
            if allplanes == 4:
                exitvector[i] = 1
            fbzcoords[octindex, :] = octcoords

        iteration += 1
        print(iteration)

    print('Done bringing points into FBZ!')

    return fbzcoords


def main():

    con = PhysicalConstants()
    reciplattvecs = np.concatenate((con.b1[np.newaxis, :], con.b2[np.newaxis, :], con.b3[np.newaxis, :]), axis=0)

    g_df, kpts_df, enk_df, qpts_df, enq_df = loadfromfile(matrixel=False)

    kpts = np.array(kpts_df[['b1', 'b2', 'b3']])
    qpts = np.array(qpts_df[['b1', 'b2', 'b3']])

    cartkpts = vectorbasis2cartesian(kpts, reciplattvecs)
    cartqpts = vectorbasis2cartesian(qpts, reciplattvecs)

    # fbzcartkpts = translate_into_fbz(cartkpts, reciplattvecs)
    fbzcartqpts = translate_into_fbz(cartqpts, reciplattvecs)

    # plotting.bz_3dscatter(con, cartkpts, enk_df, useplotly=True)
    # plotting.bz_3dscatter(con, fbzcartkpts, enk_df, useplotly=True)
    # plotting.bz_3dscatter(con, cartqpts, enq_df, useplotly=True)
    plotting.bz_3dscatter(con, fbzcartqpts, enq_df, useplotly=True)

    # cartesian_df, cartesian_df_edit = cartesian_q_points(qpts_df, con)



if __name__ == '__main__':
    main()



