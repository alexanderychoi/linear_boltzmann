import numpy as np
import pandas as pd
import time
import constants as c
import utilities
import problemparameters as pp
import matplotlib.pyplot as plt
import numpy.linalg
from scipy.sparse import linalg
import inspect


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


def low_field_GMRES(df, scm, simplelin=True, applyscmFac=False, convergence=1e-6):
    """GMRES solver hard coded for solving the BTE in low field approximation. Sign convention by Wu Li. Returns f,
    which is equal to chi/(eE/kT).
    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        scm (memmap): Memory-mapped array containing the scattering matrix, assumed simple linearization by default.
        simplelin (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        linearization, assumed simple by default (i.e. simplelin=True).
        applyscmFac (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.
        convergence (dbl): Convergence tolerance associated with the solver.
    Returns:
        f_next (nparray): Numpy array containing the (hopefully) converged GMRES solution as psi/(eE/kT).
        f_0 (nparray): Numpy array containing the RTA solution as psi_0/(eE/kT).
    """
    counter = gmres_counter()
    if applyscmFac:
        scmfac = pp.scmVal
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1
    b = (-1)* np.squeeze(df['vx [m/s]'])
    # chi2psi is used to give the finite difference matrix the right factors in front since substitution made
    chi2psi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
    invdiag = (np.diag(scm) * scmfac) ** (-1)
    f_0 = b * invdiag
    loopstart = time.time()
    f_next, criteria = linalg.gmres(scm*scmfac, b,x0=f_0,tol=convergence,callback=counter)
    print('Convergence? 0 = Successful')
    print(criteria)
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    b_check = np.dot(scm*scmfac,f_next)
    error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
    print('Residual error is {:3E}'.format(error))
    if not simplelin:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in canonical linearization')
        f_next = f_next * chi2psi
        f_0 = f_0 * chi2psi
    return f_next, f_0


def write_lowfield_GMRES(outLoc,inLoc,df,simplelin2=True,applyscmFac2=False,convergence2=1E-6):
    """Calls the iterative solver hard coded for solving the BTE in low field approximation and writes the single F_psi
     solution to file.
    Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions.
        inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
        linearization by default.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        simplelin2 (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        linearization, assumed simple by default (i.e. canonical=False).
        applyscmFac2 (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.
        convergence (dbl): Convergence tolerance associated with the solver.
    Returns:
        None. Just writes the F_psi solution to file. FPsi_#. #1 corresponds to low-field RTA, #2 corresponds to
        low-field iterative.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(inLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    utilities.check_matrix_properties(scm)
    f_next, f_0 = low_field_GMRES(df, scm, simplelin=simplelin2, applyscmFac=applyscmFac2,convergence=convergence2)
    np.save(outLoc +'f_' + '1_gmres', f_0)
    np.save(outLoc +'f_' + '2_gmres', f_next)
    print('f solutions written to file.')


def apply_centraldiff_matrix(matrix,fullkpts_df,E,step_size=1):
    """Given a scattering matrix, calculate a modified matrix using the central difference stencil and apply bc. In the
    current version, bc is not applied to points in the L, X valleys.

    Parameters:
        matrix (memmap): Scattering matrix in simple linearization.
        fullkpts_df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        E (dbl): Value of the electric field in V/m.
        scm (memmap): Memory-mapped array containing the scattering matrix, assumed simple linearization by default.
        step_size (dbl): Specifies the spacing between consecutive k-pts for the integration.

    Returns:
        shortslice_inds (numpyarray): Array of indices associated with the points along ky-kz lines with fewer than 5 pts.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        matrix (memmap): Memory-mapped array containing the modified scattering matrix, accounting for the FDM.
    """
    # Do not  flush the memmap it will overwrite consecutively.
    # Get the first and last rows since these are different because of the IC. Go through each.
    # Get the unique ky and kz values from the array for looping.

    if pp.gD == 160:
        step_size = 0.0070675528500652425 * 1E10  # 1/Angstrom to 1/m (for 160^3)
    if pp.gD == 200:
        step_size = 0.005654047459752398 * 1E10  # 1/Angstrom to 1/m (for 200^3)
    if pp.gD == 80:
        step_size = 0.0070675528500652425*2*1E10 #1/Angstron for 1/m (for 80^3)

    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]']]
    kptdata['kpt_mag'] = np.sqrt(kptdata['kx [1/A]'].values**2 + kptdata['ky [1/A]'].values**2 +
                                 kptdata['kz [1/A]'].values**2)
    kptdata['ingamma'] = kptdata['kpt_mag'] < 0.3  # Boolean. In gamma if kpoint magnitude less than some amount
    uniq_yz = np.unique(kptdata[['ky [1/A]', 'kz [1/A]']].values, axis=0)
    # If there are too few points in a slice < 5, we want to keep track of those points
    shortslice_inds = []
    l_icinds = []
    r_icinds = []
    lvalley_inds = []
    start = time.time()
    # Loop through the unique ky and kz values
    for i in range(len(uniq_yz)):
        kind = i + 1
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        # Grab the "slice" of points in k space with the same ky and kz coordinate
        slice_df = kptdata.loc[(kptdata['ky [1/A]'] == ky) & (kptdata['kz [1/A]'] == kz)]
        slice_inds = slice_df['k_inds'].values
        # if 0 in slice_inds or 1 in slice_inds or len(kptdata) in slice_inds or len(kptdata)-1 in slice_inds:
        #     lastslice_inds.append(slice_inds)
        #     continue
        # Skip all slices that intersect an L valley. Save the L valley indices
        if np.any(slice_df['ingamma'] == 0):
            lvalley_inds.append(slice_inds)
            # print('Not applied to {:d} slice because skip L valley'.format(kind))
            continue
        if len(slice_inds) > 4:
            # Subset is the slice sorted by kx value in ascending order. The index of subset still references kptdata.
            subset = slice_df.sort_values(by=['kx [1/A]'], ascending=True)
            ordered_inds = subset['k_inds'].values - 1  # indices of matrix (zero indexed)
            l_icinds.append(ordered_inds[0])  # +1 to get the k_inds values (one indexed)
            r_icinds.append(ordered_inds[-1])
            # Set the "initial condition" i.e. the point with the most negative kx value is treated as being zero
            # (and virtual point below)
            matrix[ordered_inds[0], ordered_inds[1]] += 1/(2*step_size)*c.e*E/c.hbar_joule
            matrix[ordered_inds[1], ordered_inds[2]] += 1/(2*step_size)*c.e*E/c.hbar_joule
            # Set the other "boundary condition" i.e. the point with the most positive kx value is treated as being zero
            # (and virtual point above)
            last = len(ordered_inds) - 1
            slast = len(ordered_inds) - 2
            matrix[ordered_inds[last], ordered_inds[slast]] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule
            matrix[ordered_inds[slast], ordered_inds[slast-1]] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule
            # Set the value of all other points in the slice
            inter_inds = ordered_inds[2:slast]
            inter_inds_up = ordered_inds[3:last]
            inter_inds_down = ordered_inds[1:slast-1]
            matrix[inter_inds, inter_inds_up] += 1/(2*step_size)*c.e*E/c.hbar_joule
            matrix[inter_inds, inter_inds_down] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule
        else:
            shortslice_inds.append(slice_inds)
        if kind % 10 == 0:
            pass
    print('Scattering matrix modified to incorporate central difference contribution.')
    print('Not applied to {:d} points because fewer than 5 points on the slice.'.format(len(shortslice_inds)))
    print('Finite difference not applied to L valleys. Derivative treated as zero for these points.')
    return shortslice_inds, np.array(l_icinds), np.array(r_icinds), lvalley_inds, matrix


def steady_state_full_drift_GMRES(matrix_sc, matrix_fd, kptdf, field, canonical=False, applyscmFac=False,
                                             convergence=1E-6):
    """Generalized minimal residual solver for calculating steady BTE solution in the form of Chi using the full finite
    difference matrix.
    Parameters:
        matrix_sc (memmap): Scattering matrix in simple linearization by default..
        matrix_fd (memmap): Finite difference matrix, generated by apply_centraldiff_matrix.
        kptdf (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
        canonical (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        applyscmFac (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.
        convergence (dbl): Specifies the percentage threshold for convergence.
    Returns:
        x_next (nparray): Numpy array containing the (hopefully) converged iterative solution as chi.
        x_0 (nparray): Numpy array containing the RTA solution as chi.
    """
    counter = gmres_counter()
    print('Starting steady_state_full_drift_iterative solver for {:.3E}'.format(field))
    if applyscmFac:
        scmfac = pp.scmVal
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1

    _, icinds_l,icinds_r, _, matrix_fd = apply_centraldiff_matrix(matrix_fd, kptdf, field)
    b = (-1)*c.e*field/c.kb_joule/pp.T * np.squeeze(kptdf['vx [m/s]'] * kptdf['k_FD']) * (1 - kptdf['k_FD'])
    # chi2psi is used to give the finite difference matrix the right factors in front since substitution made
    chi2psi = np.squeeze(kptdf['k_FD'] * (1 - kptdf['k_FD']))
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    x_0 = b * invdiag

    # M2 = linalg.spilu(matrix_sc * scmfac - matrix_fd)
    # M_x = lambda x: M2.solve(x)
    # M = linalg.LinearOperator((len(kptdf), len(kptdf)), M_x)
    # print('Obtained preconditioner')
    # x_0 = np.load(pp.outputLoc + 'chi_3_gmres_{:.1e}.npy'.format(7.5e4))

    loopstart = time.time()
    x_next, criteria = linalg.gmres(matrix_sc*scmfac-matrix_fd, b,x0=x_0,tol=convergence,callback=counter)
    print('Convergence?')
    print(criteria)
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    b_check = np.dot(matrix_sc*scmfac-matrix_fd,x_next)
    error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
    print('Residual error is {:3E}'.format(error))

    if canonical:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in canonical linearization')
        x_next = x_next * chi2psi
        x_0 = x_0 * chi2psi
    return x_next, x_0, error, icinds_l, icinds_r


def write_fdm_GMRES(outLoc, inLoc, fieldVector, df, canonical2=False, applyscmFac2=False,
                               convergence2=1E-6):
    """Calls the GMRES solver hard coded for solving the BTE with full FDM and writes the chis to file.
    Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions.
        inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
        linearization by default.
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        canonical2 (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        linearization, assumed simple by default (i.e. canonical=False).
        applyscmFac2 (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.
        convergence2 (dbl): Specifies the percentage threshold for convergence.
    Returns:
        None. Just writes the chi solutions to file. chi_#_EField. #1 corresponds to low-field RTA, #2 corresponds to
        low-field iterative, #3 corresponds to full finite-difference iterative.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(inLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    error = []
    for i in range(len(fieldVector)):
        fdm = np.memmap(inLoc + '/finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
        EField = fieldVector[i]
        x_next, _, temp_error, icinds_l, icinds_r = steady_state_full_drift_GMRES(scm, fdm, df, EField, canonical2, applyscmFac2, convergence2)
        error.append(temp_error)
        del fdm
        np.save(outLoc + 'chi_' + '3_gmres_' + "{:.1e}".format(EField), x_next)
        print('Solution written to file for ' + "{:.1e}".format(EField))
    np.save(outLoc + 'left_icinds', icinds_l)
    np.save(outLoc + 'right_icinds', icinds_r)

    plt.figure()
    plt.plot(fieldVector* 1E-5,error)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel('|Ax-b|/|b|')
    plt.show()


def eff_distr_GMRES(chi, matrix_sc, matrix_fd, df, field, simplelin=True, applyscmFac=False,
                                 convergence=1E-6):
    """Iterative solver for calculating effective BTE solution in the form of g_Chi using the full finite difference matrix.

    Parameters:
        chi (nparray): Solution for the steady distribution function in chi form.
        matrix_sc (memmap): Scattering matrix in simple linearization by default..
        matrix_fd (memmap): Finite difference matrix, generated by apply_centraldiff_matrix.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
        simplelin (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        applyscmFac (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.
        convergence (dbl): Specifies the percentage threshold for convergence.
    Returns:
        g_next (nparray): Numpy array containing the (hopefully) converged iterative solution as g_chi.
        g_0 (nparray): Numpy array containing the RTA solution as g0_chi.
    """
    print('Starting eff_distr_g_iterative_solver solver for {:.3E}'.format(field))
    if applyscmFac:
        scmfac = pp.scmVal
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1
    _, icinds_l,icinds_r, _, matrix_fd  = apply_centraldiff_matrix(matrix_fd, df, field)
    # Will only be able to run if you have a precalculated chi stored on file
    vd = utilities.drift_velocity(chi, df)
    f0 = df['k_FD'].values
    f = chi + f0
    b = (-1) * ((df['vx [m/s]'] - vd) * f)
    # chi2psi is used to give the finite difference matrix the right factors in front since substitution made
    psi2chi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    g_0 = b * invdiag
    loopstart = time.time()
    g_next, criteria = linalg.gmres(matrix_sc * scmfac - matrix_fd, b, x0=g_0, tol=convergence)
    print('Convergence?')
    print(criteria)
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if not simplelin:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in simple linearization')
        g_next = g_next * psi2chi
        g_0 = g_0 * psi2chi
    b_check = np.dot(matrix_sc*scmfac-matrix_fd,g_next)
    error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
    print('Residual error is {:3E}'.format(error))
    return g_next, g_0,error



def eff_distr_RTA(chi,matrix_sc,df,simplelin=True,applyscmFac=False):
    """For calculating effective BTE solution in the form of g_Chi using the low-field RTA solution.
    Parameters:
        chi (nparray): Solution for the steady distribution function in chi form.
        matrix_sc (memmap): Scattering matrix in simple linearization by default..
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        simplelin (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        applyscmFac (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.

    Returns:
        g_linear (nparray): Numpy array containing the effective distribution function for the low-field approx."""
    if applyscmFac:
        scmfac = pp.scmVal
        print('Applying 2 Pi-squared factor.')
    else:
        scmfac = 1
    psi2chi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
    f = chi + df['k_FD']
    vd = utilities.drift_velocity(chi,df)
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    g_lin = (vd-df['vx [m/s]']) * invdiag * f
    g0 = (-df['vx [m/s]']) * invdiag * df['k_FD']
    if not simplelin:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in simple linearization')
        g_lin = g_lin * psi2chi
    return g_lin, g0


def write_g_GMRES(outLoc, inLoc, fieldVector, df, simplelin2=True, applyscmFac2 = False,
                               convergence2=1E-6):
    """Calls the iterative solver hard coded for solving the effective BTE w/FDM and writes the chis to file.

    Parameters:
        outLoc (str): String containing the location of the directory to write the chi solutions and ready steady state chis.
        inLoc (str): String containing the location of the directory containing the scattering matrix, assumed simple
        linearization by default.
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        simplelin2 (bool): Boolean that specifies whether the scattering matrix is assumed simple or canonical
        linearization, assumed simple by default (i.e. canonical=False).
        applyscmFac2 (bool): Boolean that specifies whether or not to apply the 2*pi squared factor.
        convergence2 (dbl): Specifies the percentage threshold for convergence.

    Returns:
        None. Just writes the chi solutions to file. chi_#_EField. #1 corresponds to low-field RTA, #2 corresponds to
        low-field iterative, #3 corresponds to full finite-difference iterative.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(inLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    error = []
    for i in range(len(fieldVector)):
        EField = fieldVector[i]
        chi = np.load(outLoc + 'chi_3_gmres_{:.1e}.npy'.format(EField))
        fdm = np.memmap(inLoc + 'finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
        g_next, g_3_johnson,temp_error = eff_distr_GMRES(chi, scm, fdm, df, EField, simplelin2, applyscmFac2, convergence2)
        error.append(temp_error)
        del fdm
        f_i = np.load(outLoc + 'f_1.npy')
        chi_1_i = utilities.f2chi(f_i, df, EField)
        g_rta, g_1_johnson = eff_distr_RTA(chi_1_i, scm, df, simplelin2, applyscmFac2)
        np.save(outLoc + 'g_' + '1_' + "{:.1e}".format(EField), g_rta)
        np.save(outLoc + 'g_' + '3_gmres_' + "{:.1e}".format(EField), g_next)
        print('Solution written to file for ' + "{:.1e}".format(EField))
    np.save(outLoc + 'g_' + '1_johnson', g_1_johnson)
    np.save(outLoc + 'g_' + '3_johnson', g_3_johnson)

    plt.figure()
    plt.plot(fieldVector* 1E-5,error)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel('|Ag-b|/|b|')
    plt.show()


if __name__ == '__main__':
    # Right now, the functions are hardcoded to look for a scattering matrix named 'scattering_matrix_5.87_simple.mmap'
    # in the in_Loc. This can be modified later to look for just scattering_matrix_simple, or the name can be passed as
    # an argument. I kind of lean towards the argument, because it would force you to change the name each time you ran
    # with a new scattering matrix, which is probably good so we don't mess up.

    # Point to inputs and outputs
    out_Loc = pp.outputLoc
    in_Loc = pp.inputLoc

    # Read problem parameters and specify electron DataFrame
    # utilities.load_electron_df(in_Loc)
    utilities.read_problem_params(in_Loc)
    electron_df = pd.read_pickle(in_Loc+'electron_df.pkl')
    electron_df = utilities.fermi_distribution(electron_df)

    # Steady state solutions
    # fields = np.array([1])
    # fields = np.array([1e2,1e3,1e4,2.5e4,5e4,7.5e4,1e5,2e5,3e5])
    # fields = np.array([1e2,1e3,1e4,2.5e4,5e4,7.5e4,1e5])
    # fields = np.array([1e2, 1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5])
    # fields = np.array([3.5e5,4e5,4.5e5,5e5,5.5e5,6e5])
    # fields = np.geomspace(1e1,2.9e5,20)
    # fields = np.array([1.5e4,2e4])
    # fields = np.array([1.4e4])
    fields = np.array([1e2, 1e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5,2e5])

    applySCMFac = pp.scmBool
    simpleLin = pp.simpleBool
    writeLowfield = False
    writeFDM = True
    writeEffective = False

    if writeLowfield:
        write_lowfield_GMRES(out_Loc, in_Loc, electron_df, simpleLin, applySCMFac,1E-8)
        print('Low field solutions written to file as Fs.')

    if writeFDM:
        write_fdm_GMRES(out_Loc, in_Loc, fields, electron_df, not simpleLin, applySCMFac, 1E-5)
        print('FDM solutions written to file as chis.')

    if writeEffective:
        write_g_GMRES(out_Loc, in_Loc, fields, electron_df, simpleLin, applySCMFac, 1E-5)
        print('Effective distribution solutions written to file.')


