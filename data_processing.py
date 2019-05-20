#!/usr/bin/env python
# coding: utf-8

# ## Importing data (Alex updated: 4/30/19)

# In[1]:


import numpy as np

# Image processing tools
import skimage
import skimage.filters

import pandas as pd
import scipy.optimize
import scipy.stats as st
import numba
import itertools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from numpy.linalg import inv

from tqdm import tqdm, trange
from scipy import special, optimize
from scipy import integrate

import plotly.offline as py
import plotly.graph_objs as go
import plotly
#plotly.tools.set_credentials_file(username='AYChoi', api_key='ZacDa7fKo8hfiELPfs57')
plotly.tools.set_credentials_file(username='AlexanderYChoi', api_key='VyLt05wzc89iXwSC82FO')


# Load the e-ph matrix elements data. The two numbers reported at the end should be the same. If they are not, there are duplicate e-ph elements. It's VERY IMPORTANT to note that the g elements here are actually |g|^2 which is why they are real numbers.

# In[2]:


data = pd.read_csv('gaas.eph_matrix', sep='\t',header= None,skiprows=(0,1))
data.columns = ['0']
data_array = data['0'].values
new_array = np.zeros((len(data_array),7))
for i1 in trange(len(data_array)):
    new_array[i1,:] = data_array[i1].split()
    
g_df = pd.DataFrame(data=new_array,columns = ['k_inds','q_inds','k+q_inds','m_band','n_band','im_mode','g_element'])
g_df[['k_inds','q_inds','k+q_inds','m_band','n_band','im_mode']] = g_df[['k_inds','q_inds','k+q_inds','m_band','n_band','im_mode']].apply(pd.to_numeric,downcast = 'integer')
len(g_df[['k_inds','q_inds','k+q_inds','m_band','n_band','im_mode','g_element']]),len(g_df[['k_inds','q_inds','k+q_inds','m_band','n_band','im_mode','g_element']].drop_duplicates())


# Now load the k-point indices, q-point indices, k-point energies, phonon energies into dataframes.

# In[3]:


kpts = pd.read_csv('gaas.kpts', sep='\t',header= None)
kpts.columns = ['0']
kpts_array = kpts['0'].values
new_kpt_array = np.zeros((len(kpts_array),4))
for i1 in trange(len(kpts_array)):
    new_kpt_array[i1,:] = kpts_array[i1].split()
    
kpts_df = pd.DataFrame(data=new_kpt_array,columns = ['k_inds','b1','b2','b3'])
kpts_df[['k_inds']] = kpts_df[['k_inds']].apply(pd.to_numeric,downcast = 'integer')
kpts_df.head()


# In[4]:


enk = pd.read_csv('gaas.enk', sep='\t',header= None)
enk.columns = ['0']
enk_array = enk['0'].values
new_enk_array = np.zeros((len(enk_array),3))
for i1 in trange(len(enk_array)):
    new_enk_array[i1,:] = enk_array[i1].split()
    
enk_df = pd.DataFrame(data=new_enk_array,columns = ['k_inds','band_inds','energy [Ryd]'])
enk_df[['k_inds','band_inds']] = enk_df[['k_inds','band_inds']].apply(pd.to_numeric,downcast = 'integer')
enk_df.head()


# In[95]:


enq = pd.read_csv('gaas.enq', sep='\t',header= None)
enq.columns = ['0']
enq_array = enq['0'].values
new_enq_array = np.zeros((len(enq_array),3))
for i1 in trange(len(enq_array)):
    new_enq_array[i1,:] = enq_array[i1].split()
    
enq_df = pd.DataFrame(data=new_enq_array,columns = ['q_inds','im_mode','energy [Ryd]'])
enq_df[['q_inds','im_mode']] = enq_df[['q_inds','im_mode']].apply(pd.to_numeric,downcast = 'integer')
print(enq_df.shape)
enq_df.head()


# In[96]:


qpts = pd.read_csv('gaas.qpts', sep='\t',header= None)
qpts.columns = ['0']
qpts_array = qpts['0'].values
new_qpt_array = np.zeros((len(qpts_array),4))

for i1 in trange(len(qpts_array)):
    new_qpt_array[i1,:] = qpts_array[i1].split()
    
qpts_df = pd.DataFrame(data=new_qpt_array,columns = ['q_inds','b1','b2','b3'])
qpts_df[['q_inds']] = qpts_df[['q_inds']].apply(pd.to_numeric,downcast = 'integer')
print(qpts_df.shape)
qpts_df.head()

Processed data:

e-ph matrix elements = g_df[['k_inds','q_inds','k+q_inds','m_band','n_band','im_mode']]
k-points = kpts_df[['k_inds','b1','b2','b3']]
q-points = qpts_df[['q_inds','b1','b2','b3']]
k-energy = enk_df[['k_inds','band_inds','energy [Ryd]']]
q-energy = enq_df[['q_inds','im_mode','energy [Ryd]']]
# ## Data Processing (Alex Updated: 4/30)

# In[7]:


def cartesian_q_points(qpts_df):
    """
    Given a dataframe containing indexed q-points in terms of the crystal lattice vector, return the dataframe with cartesian q coordinates.     
    Parameters:
    -----------
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
    
    # Need a lattice constant for GaAs. This is obviously somewhat sensitive to temperature.
    a = 5.556 #[A]
    
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

    cartesian_df_edit.loc[qx_plus,'kx [1/A]'] = cartesian_df.loc[qx_plus,'kx [1/A]'] -1
    cartesian_df_edit.loc[qx_minus,'kx [1/A]'] = cartesian_df.loc[qx_minus,'kx [1/A]'] +1

    cartesian_df_edit.loc[qy_plus,'ky [1/A]'] = cartesian_df.loc[qy_plus,'ky [1/A]'] -1
    cartesian_df_edit.loc[qy_minus,'ky [1/A]'] = cartesian_df.loc[qy_minus,'ky [1/A]'] +1

    cartesian_df_edit.loc[qz_plus,'kz [1/A]'] = cartesian_df.loc[qz_plus,'kz [1/A]'] -1
    cartesian_df_edit.loc[qz_minus,'kz [1/A]'] = cartesian_df.loc[qz_minus,'kz [1/A]'] +1
    
    return cartesian_df,cartesian_df_edit


# In[8]:


a = 5.556 #[A]
kb = 1.38064852*10**(-23); # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
T = 300
e = 1.602*10**(-19)


# In[9]:


kvel = pd.read_csv('gaas.vel', sep='\t',header= None,skiprows=[0,1,2])
kvel.columns = ['0']
kvel_array = kvel['0'].values
new_kvel_array = np.zeros((len(kvel_array),10))
for i1 in trange(len(kvel_array)):
    new_kvel_array[i1,:] = kvel_array[i1].split()
    
kvel_df = pd.DataFrame(data=new_kvel_array,columns = ['k_inds','bands','energy','kx [2pi/alat]','ky [2pi/alat]','kz [2pi/alat]','vx_dir','vy_dir','vz_dir','v_mag [m/s]'])
kvel_df[['k_inds']] = kvel_df[['k_inds']].apply(pd.to_numeric,downcast = 'integer')

kvel_edit = kvel_df.copy(deep=True)

kx_plus = kvel_df['kx [2pi/alat]'] > 0.5
kx_minus = kvel_df['kx [2pi/alat]'] < -0.5

ky_plus = kvel_df['ky [2pi/alat]'] > 0.5
ky_minus = kvel_df['ky [2pi/alat]'] < -0.5

kz_plus = kvel_df['kz [2pi/alat]'] > 0.5
kz_minus = kvel_df['kz [2pi/alat]'] < -0.5

kvel_edit.loc[kx_plus,'kx [2pi/alat]'] = kvel_df.loc[kx_plus,'kx [2pi/alat]'] -1
kvel_edit.loc[kx_minus,'kx [2pi/alat]'] = kvel_df.loc[kx_minus,'kx [2pi/alat]'] +1

kvel_edit.loc[ky_plus,'ky [2pi/alat]'] = kvel_df.loc[ky_plus,'ky [2pi/alat]'] -1
kvel_edit.loc[ky_minus,'ky [2pi/alat]'] = kvel_df.loc[ky_minus,'ky [2pi/alat]'] +1

kvel_edit.loc[kz_plus,'kz [2pi/alat]'] = kvel_df.loc[kz_plus,'kz [2pi/alat]'] -1
kvel_edit.loc[kz_minus,'kz [2pi/alat]'] = kvel_df.loc[kz_minus,'kz [2pi/alat]'] +1

kvel_df = kvel_edit.copy(deep=True)
kvel_df.head()

cart_kpts_df = kvel_df.copy(deep=True)
cart_kpts_df['kx [2pi/alat]'] = cart_kpts_df['kx [2pi/alat]'].values*2*np.pi/a
cart_kpts_df['ky [2pi/alat]'] = cart_kpts_df['ky [2pi/alat]'].values*2*np.pi/a
cart_kpts_df['kz [2pi/alat]'] = cart_kpts_df['kz [2pi/alat]'].values*2*np.pi/a

cart_kpts_df.columns = ['k_inds', 'bands', 'energy', 'kx [1/A]', 'ky [1/A]','kz [1/A]', 'vx_dir', 'vy_dir', 'vz_dir', 'v_mag [m/s]']


# In[10]:


cart_qpts_df,edit_cart_qpts_df = cartesian_q_points(qpts_df)


# In[66]:


trace1 = go.Scatter3d(
    x=cart_kpts_df['kx [1/A]'].values/(2*np.pi/(a)),
    y=cart_kpts_df['ky [1/A]'].values/(2*np.pi/(a)),
    z=cart_kpts_df['kz [1/A]'].values/(2*np.pi/(a)),
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
                    scene = dict(
                    xaxis = dict(
                        title='kx',titlefont = dict(family='Oswald, monospace',size=18)),
                    yaxis = dict(
                        title='ky',titlefont = dict(family='Oswald, monospace',size=18)),
                    zaxis = dict(
                        title='kz',titlefont = dict(family='Oswald, monospace',size=18)),))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')


# In[12]:


trace1 = go.Scatter3d(
    x=cart_kpts_df['kx [1/A]'].values/(2*np.pi/(a)),
    y=cart_kpts_df['ky [1/A]'].values/(2*np.pi/(a)),
    z=cart_kpts_df['kz [1/A]'].values/(2*np.pi/(a)),
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

data = [trace1]
layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        title='kx',titlefont = dict(family='Oswald, monospace',size=18)),
                    yaxis = dict(
                        title='ky',titlefont = dict(family='Oswald, monospace',size=18)),
                    zaxis = dict(
                        title='kz',titlefont = dict(family='Oswald, monospace',size=18)),))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')


# In[13]:


trace1 = go.Scatter3d(
    x=cart_kpts_df['kx [1/A]'].values/(2*np.pi/(a)),
    y=cart_kpts_df['ky [1/A]'].values/(2*np.pi/(a)),
    z=cart_kpts_df['kz [1/A]'].values/(2*np.pi/(a)),
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
                    scene = dict(
                    xaxis = dict(
                        title='kx',titlefont = dict(family='Oswald, monospace',size=18)),
                    yaxis = dict(
                        title='ky',titlefont = dict(family='Oswald, monospace',size=18)),
                    zaxis = dict(
                        title='kz',titlefont = dict(family='Oswald, monospace',size=18)),))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')


# In[14]:


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
                    scene = dict(
                    xaxis = dict(
                        title='kx',titlefont = dict(family='Oswald, monospace',size=18)),
                    yaxis = dict(
                        title='ky',titlefont = dict(family='Oswald, monospace',size=18)),
                    zaxis = dict(
                        title='kz',titlefont = dict(family='Oswald, monospace',size=18)),))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')


# In[16]:


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
         
    """
    # Physical constants    
    e = 1.602*10**(-19) # fundamental electronic charge [C]
    kb = 1.38064852*10**(-23); # Boltzmann constant in SI [m^2 kg s^-2 K^-1]


    g_df['k_FD'] = (np.exp((g_df['k_en [eV]'].values*e - mu*e)/(kb*T)) + 1)**(-1)
    g_df['k+q_FD'] = (np.exp((g_df['k+q_en [eV]'].values*e - mu*e)/(kb*T)) + 1)**(-1)

    return g_df


# In[17]:


def bose_distribution(g_df,T):
    """
    This function takes a list of q-point indices and returns the Bose-Einstein distributions associated with each q-point on that list.    
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
        
        BE : vector_like, shape (n,1)
        Bose-einstein distribution
         
    """
    # Physical constants    
    e = 1.602*10**(-19) # fundamental electronic charge [C]
    kb = 1.38064852*10**(-23); # Boltzmann constant in SI [m^2 kg s^-2 K^-1]

    g_df['BE'] = (np.exp((g_df['q_en [eV]'].values*e)/(kb*T)) - 1)**(-1)
    return g_df


# In[18]:


def fermionic_processing(g_df,cart_kpts_df,enk_df,mu,T):
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
               
    cart_kpts_df : pandas dataframe containing:
    
        k_inds : vector_like, shape (n,1)
        Index of k point
        
        kx : vector_like, shape (n,1)
        x-coordinate in Cartesian momentum space [1/m]    
        
        ky : vector_like, shape (n,1)
        y-coordinate in Cartesian momentum space [1/m]  
        
        kz : vector_like, shape (n,1)
        z-coordinate in Cartesian momentum space [1/m]
        
    enk_df : pandas dataframe containing

        k_inds : vector_like, shape (n,1)
        Index of k point
        
        band_inds : vector_like, shape (n,1)
        Band index
        
        energy [Ryd] : vector_like, shape (n,1)
        Energy associated with k point in Rydberg units
        
        
    mu : scalar
    Chemical potential of electronic states [eV]
    
    T : scalar
    Lattice temperature in Kelvin
    
    Returns:
    --------
    
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
        
        k_fermi_dist : vector_like, shape (n,1)
        Fermi distribution of pre collision state
        
        k+q_fermi_dist : vector_like, shape (n,1)
        Fermi distribution of post collision state
        
        k_energy : vector_like, shape (n,1)
        Energy of the pre collision state
        
        k+q_energy : vector_like, shape (n,1)
        Energy of the post collision state
         
    """
    
    # Physical constants
    e = 1.602*10**(-19) # fundamental electronic charge [C]
    kb = 1.38064852*10**(-23); # Boltzmann constant in SI [m^2 kg s^-2 K^-1]
    
    index_vector = cart_kpts_df['k_inds'].values    
    g_df['k_en [eV]'] = np.zeros(len(g_df))
    g_df['k+q_en [eV]'] = np.zeros(len(g_df))
    g_df['collision_state'] = np.zeros(len(g_df))
    
    for i1 in trange(len(cart_kpts_df)):
        index = index_vector[i1]
        
        g_slice_k = g_df['k_inds'] == index        
        g_slice_kq = g_df['k+q_inds'] == index
        
        k_slice = cart_kpts_df['k_inds'] == index
        enk_slice = enk_df['k_inds'] == index
        
        g_df.loc[g_slice_k,'k_en [eV]'] = enk_df.loc[enk_slice,'energy [Ryd]'].values*13.6056980659 
        g_df.loc[g_slice_kq,'k+q_en [eV]'] = enk_df.loc[enk_slice,'energy [Ryd]'].values*13.6056980659
        
    abs_inds = g_df['k_en [eV]'] < g_df['k+q_en [eV]'] #absorbed indices
    ems_inds = g_df['k_en [eV]'] > g_df['k+q_en [eV]'] #emission indices
    
    g_df.loc[abs_inds,'collision_state'] = 1
    g_df.loc[ems_inds,'collision_state'] = -1
    
    g_df = fermi_distribution(g_df,mu, T)
    
    return g_df


# In[20]:


def bosonic_processing(g_df,enq_df,T):
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
               
    cart_qpts_df : pandas dataframe containing:
    
        q_inds : vector_like, shape (n,1)
        Index of q point
        
        kx : vector_like, shape (n,1)
        x-coordinate in Cartesian momentum space [1/m]    
        
        ky : vector_like, shape (n,1)
        y-coordinate in Cartesian momentum space [1/m]  
        
        kz : vector_like, shape (n,1)
        z-coordinate in Cartesian momentum space [1/m]
        
    enq_df : pandas dataframe containing

        k_inds : vector_like, shape (n,1)
        Index of k point
        
        im_mode : vector_like, shape (n,1)
        Phonon polarization index
        
        energy [Ryd] : vector_like, shape (n,1)
        Energy associated with k point in Rydberg units
        
            
    T : scalar
    Lattice temperature in Kelvin
    
    Returns:
    --------
    
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
        
        k_fermi_dist : vector_like, shape (n,1)
        Fermi distribution of pre collision state
        
        k+q_fermi_dist : vector_like, shape (n,1)
        Fermi distribution of post collision state
        
        k_energy : vector_like, shape (n,1)
        Energy of the pre collision state
        
        k+q_energy : vector_like, shape (n,1)
        Energy of the post collision state
         
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
    
    return g_df,modified_enq_df


# In[35]:


g_df.loc[g_df['q_id'] == 1000].head()


# In[68]:


g_df.loc[g_df['k_inds'] == 79]


# In[60]:


np.abs(-enk_df.loc[enk_df['k_inds'] == 10]['energy [Ryd]'].values+enk_df.loc[enk_df['k_inds'] == 79]['energy [Ryd]'].values)*13.6056980659


# In[61]:


np.abs(-enk_df.loc[enk_df['k_inds'] == 34]['energy [Ryd]'].values+enk_df.loc[enk_df['k_inds'] == 87]['energy [Ryd]'].values)*13.6056980659


# In[55]:


enq_df.loc[enq_df['q_inds'] == 177]


# In[19]:


g_df = fermionic_processing(g_df,cart_kpts_df,enk_df,5.780,300)


# In[21]:


g_df,modified_enq_df = bosonic_processing(g_df,enq_df,T)


# In[34]:


modified_enq_df.loc[modified_enq_df['q_id'] == 1000].head()


# In[62]:


cart_kpts_df.loc[cart_kpts_df['k_inds']==231]


# In[63]:


cart_kpts_df.loc[cart_kpts_df['k_inds']==1201]


# In[32]:


g_df.loc[g_df['collision_state'] == 0]


# In[66]:


np.min(g_df['g_element'].values)


# In[22]:


g_df.head()


# In[ ]:





# In[42]:


g_df['k_en [eV]'] - g_df['k+q_en [eV]']


# In[40]:


np.max(g_df['k_en [eV]'])-np.min(g_df['k_en [eV]'])


# In[41]:


(np.max(enk_df['energy [Ryd]'])-np.min(enk_df['energy [Ryd]']))*13.6056980659


# In[24]:


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
    
    
    g_df_ems = g_df.loc[(g_df['collision_state'] == -1)].copy(deep=True)
    g_df_abs = g_df.loc[(g_df['collision_state'] == 1)].copy(deep=True)
    
    g_df_ems['weight'] = np.multiply(g_df_ems['BE'].values + 1 - g_df_ems['k+q_FD'].values,g_df_ems['g_element'].values)
    g_df_abs['weight'] = np.multiply(g_df_abs['BE'].values + g_df_abs['k+q_FD'].values,g_df_abs['g_element'].values)

    
    
    
    abs_sr = g_df_abs.groupby(['k_inds'])['weight'].agg('sum')
    abs_scattering = abs_sr.to_frame().reset_index()
    
    ems_sr = g_df_ems.groupby(['k_inds'])['weight'].agg('sum')
    ems_scattering = ems_sr.to_frame().reset_index()
    
    return abs_scattering,ems_scattering


# In[25]:


abs_scattering,ems_scattering = scattering_rate(g_df)


# In[ ]:


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
    
    
    g_df_ems = g_df.loc[(g_df['collision_state'] == -1)].copy(deep=True)
    g_df_abs = g_df.loc[(g_df['collision_state'] == 1)].copy(deep=True)
    
    g_df_ems['weight'] = np.multiply(g_df_ems['BE'].values + 1 - g_df_ems['k+q_FD'].values,g_df_ems['g_element'].values)
    g_df_abs['weight'] = np.multiply(g_df_abs['BE'].values + g_df_abs['k+q_FD'].values,g_df_abs['g_element'].values)

    g_df_abs['id'] = g_df_abs.groupby(['k_inds','k+q_inds']).ngroup()
    g_df_ems['id'] = g_df_ems.groupby(['k_inds','k+q_inds']).ngroup()    
    
    
    
    abs_sr = g_df_abs.groupby(['k_inds', 'k+q_inds','id'])['weight'].agg('sum')
    summed_abs_df = abs_sr.to_frame().reset_index()
    
    ems_sr = g_df_ems.groupby(['k_inds', 'k+q_inds','id'])['weight'].agg('sum')
    summed_ems_df = ems_sr.to_frame().reset_index()
    
    return summed_abs_df,summed_ems_df


# In[ ]:


abs_scattering_array = np.zeros(len(np.unique(enk_df['k_inds'])))
ems_scattering_array = np.zeros(len(np.unique(enk_df['k_inds'])))
abs_scattering_array[abs_scattering['k_inds'].values-1] = abs_scattering['weight'].values
ems_scattering_array[ems_scattering['k_inds'].values-1] = ems_scattering['weight'].values


# In[ ]:


import matplotlib.cm as cm
plt.rcParams.update({'font.size': 40})
plt.rcParams.update({'lines.linewidth': 3.5})

fig = plt.figure(figsize=(12,10))
ax = plt.gca()

plt.scatter(enk_df['energy [Ryd]'].values,abs_scattering_array+ems_scattering_array,c = 'Red')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Scattering Rate [1/fs]')
plt.xlabel('Energy [Ryd]')
#plt.legend()
plt.show()
fig.savefig('test.png', bbox_inches='tight')


# In[ ]:


abs_scattering_array+ems_scattering_array


# In[ ]:


summed_abs_df,summed_ems_df = coupling_matrix_calc(g_df)


# In[ ]:


interesting_abs = summed_abs_df.groupby(['k_inds'])['weight'].agg('sum').to_frame().reset_index()


# In[ ]:


interesting_ems = summed_ems_df.groupby(['k_inds'])['weight'].agg('sum').to_frame().reset_index()


# In[ ]:


len(interesting_abs)


# In[ ]:


g_df.loc[(g_df['k_inds'] == 2213)* g_df['k+q_inds'] == 502]


# In[ ]:


g_df.loc[(g_df['q_inds'] == 4015)*(g_df['collision_state'] == -1)]


# In[ ]:


summed_abs_df.loc[summed_abs_df['id'] == 500]


# In[ ]:


summed_ems_df.loc[summed_ems_df['id'] == 539]


# In[ ]:


summed_ems_df.loc[summed_ems_df['k+q_inds'] == 1019]


# In[ ]:


abs_array = np.zeros((len(np.unique(enk_df['k_inds'])),len(np.unique(enk_df['k_inds']))))
ems_array = np.zeros((len(np.unique(enk_df['k_inds'])),len(np.unique(enk_df['k_inds']))))


# In[ ]:


abs_array[summed_abs_df['k_inds'].values-1,summed_abs_df['k+q_inds'].values-1] = summed_abs_df['weight'].values
ems_array[summed_ems_df['k_inds'].values-1,summed_ems_df['k+q_inds'].values-1] = summed_ems_df['weight'].values


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
plt.set_cmap('inferno')
fig = plt.figure(figsize=(20, 12.2))

ax = fig.add_subplot(111)
ax.set_title('Collision Matrix')
plt.imshow(abs_array/np.max(abs_array))
ax.set_aspect('equal')
plt.xlabel('k index')
plt.ylabel('k_p index')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
cbar = plt.colorbar(orientation='vertical')
cbar.set_label('Coupling Rate [arb]', rotation=270)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
plt.set_cmap('inferno')

fig = plt.figure(figsize=(20, 12.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(ems_array/np.max(ems_array))
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()


# In[ ]:


plt.imshow(ems_array);
plt.colorbar()
plt.show()


# In[ ]:


import matplotlib as mpl
from matplotlib import pyplot
import numpy as np

# make a color map of fixed colors
cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['blue','black','red'],
                                           256)

# tell imshow about color map so that only set colors are used
img = pyplot.imshow(ems_array*1000,interpolation='nearest',
                    cmap = cmap2,
                    origin='lower')

pyplot.show()


# In[ ]:


np.sum(abs_array+ems_array,axis=1)


# In[69]:


import matplotlib as mpl
from matplotlib import pyplot
import numpy as np

# make a color map of fixed colors
cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['blue','black','red'],
                                           256)

# tell imshow about color map so that only set colors are used
img = pyplot.imshow(ems_array*1000,interpolation='nearest',
                    cmap = cmap2,
                    origin='lower')

pyplot.show()


# In[71]:


"hello"


# ## Data validation (Peishi Updated: 4/30)

# In[110]:


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
    a = 5.556 #[A] 
    b1 = (2*np.pi/a) * np.array([1, -1, 1]) 
    b2 = (2*np.pi/a) * np.array([1, 1, -1]) 
    b3 = (2*np.pi/a) * np.array([-1, 1, 1]) 

    # L point in BZ is given by 0.5*b1 + 0.5*b2 + 0.5*b3 
    # X point in BZ is given by 0.5*b2 + 0.5*b3 
    lpoint = 0.5 * (b1 + b2 + b3) 
    xpoint = 0.5 * (b2 + b3) 

    # We can find kpoints along a path just by considering a dot product with lpoint and xpoint vectors. 
    # Any kpoints with angle smaller than some tolerance are considered on the path and we can plot their corresponding frequencies 
    deg2rad = 2*np.pi/360 
    ang_tol = 1 * deg2rad  # 1 degree in radians 

    enkonly = np.array(enk['energy [Ryd]'])[:, np.newaxis] 
    kptsonly = np.array(kpts[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']]) / (2*np.pi/a) 
    kptsmag = np.linalg.norm(kptsonly, axis=1)[:, np.newaxis] 

    dot_l = np.zeros(len(kpts))
    dot_x = np.zeros(len(kpts))

    # Separate assignment for gamma point to avoid divide by zero error
    nongamma = kptsmag!=0
    dot_l[np.squeeze(nongamma)] = np.divide(np.dot(kptsonly, lpoint[:, np.newaxis])[nongamma], kptsmag[nongamma]) / np.linalg.norm(lpoint) 
    dot_x[np.squeeze(nongamma)] = np.divide(np.dot(kptsonly, xpoint[:, np.newaxis])[nongamma], kptsmag[nongamma]) / np.linalg.norm(xpoint) 
    dot_l[np.squeeze(kptsmag==0)] = 0 
    dot_x[np.squeeze(kptsmag==0)] = 0 

    lpath = np.logical_or(np.arccos(dot_l) < ang_tol, np.squeeze(kptsmag == 0))
    xpath = np.logical_or(np.arccos(dot_x) < ang_tol, np.squeeze(kptsmag == 0))

    plt.figure() 
    plt.plot(kptsmag[lpath], enkonly[lpath], '.') 
    plt.plot(-1*kptsmag[xpath], enkonly[xpath], '.') 
    plt.xlabel('k magnitude') 
    plt.ylabel('Energy in Ry')
    plt.show()


# In[111]:


plot_bandstructure(cart_kpts_df, enk_df)


# In[112]:


# plot_bandstructure(edit_cart_qpts_df, enq_df)
kpts = cart_kpts_df
enk = enk_df

kpts = edit_cart_qpts_df
enk = enq_df

a = 5.556 #[A] 
b1 = (2*np.pi/a) * np.array([1, -1, 1]) 
b2 = (2*np.pi/a) * np.array([1, 1, -1]) 
b3 = (2*np.pi/a) * np.array([-1, 1, 1]) 

# L point in BZ is given by 0.5*b1 + 0.5*b2 + 0.5*b3 
# X point in BZ is given by 0.5*b2 + 0.5*b3 
lpoint = 0.5 * (b1 + b2 + b3) 
xpoint = 0.5 * (b2 + b3) 

# We can find kpoints along a path just by considering a dot product with lpoint and xpoint vectors. 
# Any kpoints with angle smaller than some tolerance are considered on the path and we can plot their corresponding frequencies 
deg2rad = 2*np.pi/360 
ang_tol = 1 * deg2rad  # 1 degree in radians 

kptsonly = np.array(kpts[['kx [1/A]', 'ky [1/A]', 'kz [1/A]']]) / (2*np.pi/a) 
kptsmag = np.linalg.norm(kptsonly, axis=1)[:, np.newaxis] 

print(kptsonly.shape)
print(kptsmag.shape)

dot_l = np.zeros(len(kpts))
dot_x = np.zeros(len(kpts))

# Separate assignment for gamma point to avoid divide by zero error
nongamma = kptsmag!=0
dot_l[np.squeeze(nongamma)] = np.divide(np.dot(kptsonly, lpoint[:, np.newaxis])[nongamma], kptsmag[nongamma]) / np.linalg.norm(lpoint) 
dot_x[np.squeeze(nongamma)] = np.divide(np.dot(kptsonly, xpoint[:, np.newaxis])[nongamma], kptsmag[nongamma]) / np.linalg.norm(xpoint) 
dot_l[np.squeeze(kptsmag==0)] = 0 
dot_x[np.squeeze(kptsmag==0)] = 0 

lpath = np.logical_or(np.arccos(dot_l) < ang_tol, np.squeeze(kptsmag == 0))
xpath = np.logical_or(np.arccos(dot_x) < ang_tol, np.squeeze(kptsmag == 0))

# Need to reshape the energy dataframe for easy plotting if there are multiple bands
enk_ra = np.array(enk.iloc[:,:])
enk_ra.sort(axis=0)
nk = int(np.max(enk_ra[:, 0]))  # nk = number of kpts = highest kpts index
if np.mod(len(enk_ra), nk) != 0:
    exit('Something is wack with the number of bands and kpoints in the array')
else:
    nb = int(len(enk_ra) / nk)
enkonly = enk_ra[:, 2]
enk_by_band = np.reshape(enkonly, (nk, nb), order='C')

print(enk_by_band.shape)

plt.figure()
for b in range(nb):
    plt.plot(kptsmag[lpath], enk_by_band[lpath, b], '.', color='C0') 
    plt.plot(-1*kptsmag[xpath], enk_by_band[xpath, b], '.', color='C1') 
plt.xlabel('k magnitude') 
plt.ylabel('Energy in Ry')
plt.show()

