import numpy as np
import constants as c
import utilities
import problemparameters as pp


# The following set calculates the low-frequency diffusion based on solutions to effective Boltzmann equation
def thermal_diffusion(vd_vd,chi,df):
    """Calculate the low-frequency non-eq diffusion coefficent as per my derivation.
    Parameters:
        vd_vd (nparray): Two-particle, one-time, correlation function for drift velocity fluctuations
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        D (double): The value of the non-eq diffusion coefficient in m^2/s.
    """
    g_inds,l_inds,x_inds = utilities.split_valleys(df,False)
    g_df = df.loc[g_inds]
    l_df = df.loc[l_inds]
    f = chi + df['k_FD']
    D_g = 1/np.sum(f[g_inds])*np.sum(vd_vd[g_inds]*g_df['vx [m/s]'])
    D_l = 1/np.sum(f[l_inds])*np.sum(vd_vd[l_inds]*l_df['vx [m/s]'])
    D_x = 0
    D_th = np.sum((f[g_inds]))/np.sum(f)*D_g + np.sum((f[l_inds]))/np.sum(f)*D_l
    if pp.getX:
        x_df = df.loc[x_inds]
        D_x = 1/np.sum(f[x_inds])*np.sum(vd_vd[x_inds]*x_df['vx [m/s]'])
        D_th = np.sum((f[g_inds]))/np.sum(f)*D_g + np.sum((f[l_inds]))/np.sum(f)*D_l + np.sum((f[x_inds]))/np.sum(f)*D_x

    return D_th, D_g, D_l, D_x


def intervalley_diffusion_two_valley(ng_ng,chi,df):
    """Calculate the low-frequency non-eq diffusion coefficent as per my derivation.
    Parameters:
        ng_ng (nparray): Two-particle, one-time, correlation function for Gamma valley number fluctuations
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        D (double): The value of the non-eq diffusion coefficient in m^2/s.
    """
    g_inds,l_inds,x_inds = utilities.split_valleys(df,False)
    f = chi + df['k_FD']
    g_df = df.loc[g_inds]
    l_df = df.loc[l_inds]
    vd_g = utilities.mean_velocity(chi[g_inds],g_df)
    vd_l = utilities.mean_velocity(chi[l_inds],l_df)
    D_tr = (vd_g-vd_l)**2/np.sum(f)*np.sum(ng_ng[g_inds])

    return D_tr


def intervalley_diffusion_three_valley(ng_nl,ng_nx,nl_nx,chi,df):
    """Calculate the low-frequency non-eq diffusion coefficent as per my derivation.
    Parameters:
        ng_nl (nparray): Two-particle, one-time, correlation function for Gamma_l valley number fluctuations
        ng_nx (nparray): Two-particle, one-time, correlation function for Gamma-X valley number fluctuations
        nl_nx (nparray): Two-particle, one-time, correlation function for L-X valley number fluctuations

        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        D (double): The value of the non-eq diffusion coefficient in m^2/s.
    """
    g_inds,l_inds,x_inds = utilities.split_valleys(df,False)
    f = chi + df['k_FD']
    g_df = df.loc[g_inds]
    l_df = df.loc[l_inds]
    x_df = df.loc[x_inds]
    vd_g = utilities.mean_velocity(chi[g_inds],g_df)
    vd_l = utilities.mean_velocity(chi[l_inds],l_df)
    vd_x = utilities.mean_velocity(chi[x_inds],x_df)
    D_tr_gl = -(vd_g-vd_l)**2/np.sum(f)*np.sum(ng_nl[g_inds])
    D_tr_gx = -(vd_g-vd_x)**2/np.sum(f)*np.sum(ng_nx[g_inds])
    D_tr_lx = -(vd_l-vd_x)**2/np.sum(f)*np.sum(nl_nx[l_inds])
    D_tr = D_tr_gl + D_tr_gx + D_tr_lx

    return D_tr,D_tr_gl,D_tr_gx,D_tr_lx


def noiseT(inLoc,D,mobility,df, applyscmFac = False):
    """Calculates the noise temperature using the Price relationship.
    Parameters:
        inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
        linearization by default.
        D (double): The value of the non-eq diffusion coefficient in m^2/s.
        mobility (df): The value of the mobility in m^2/V-s.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.

    Returns:
        None. Just writes the chi solutions to file. chi_#_EField. #1 corresponds to low-field RTA, #2 corresponds to
        low-field iterative, #3 corresponds to full finite-difference iterative.
    """
    nkpts = len(df)
    scm = np.memmap(inLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    if applyscmFac:
        scmfac = pp.scmVal
    else:
        scmfac = 1
    invdiag = (np.diag(scm) * scmfac) ** (-1)
    f0 = df['k_FD'].values
    tau = -np.sum(f0 * invdiag) / np.sum(f0)
    n = utilities.calculate_density(df)
    D_eq = c.kb_joule*pp.T*tau/(0.063*9.11e-31)

    con_eq = c.e*tau/(0.063*9.11e-31) * c.e * n
    con_neq = mobility * c.e * n
    Tn = D/D_eq*pp.T*con_eq/con_neq
    return Tn


if __name__ == '__main__':
    out_Loc = pp.outputLoc
    in_Loc = pp.inputLoc
    utilities.read_problem_params(in_Loc)
