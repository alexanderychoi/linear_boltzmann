import numpy as np
import time
import constants as c
import utilities
import problem_parameters as pp
import matplotlib.pyplot as plt
import numpy.linalg
from scipy.sparse import linalg
import preprocessing

# PURPOSE: THIS MODULE CALCULATES SOLUTIONS TO THE BOLTZMANN EQUATION UNDER VARIOUS APPROXIMATIONS AND STORES THE
# SOLUTIONS AS A FUNCTION OF FREQUENCY AND ELECTRIC FIELD STRENGTH. RIGHT NOW THIS IS HARDCODED FOR GAAS WITH THE FIELD
# APPLIED IN THE [1 0 0]. TOGGLES SET THE FINITE DIFFERENCE SCHEME AND THE CONVERGENCE PARAMETERS FOR GMRES.

# ORDER: THIS MODULE CAN BE RUN AFTER PROBLEM_PARAMETERS.PY HAS SPECIFIED THE SIMULATION AND PHYSICAL PARAMETERS.

# OUTPUT: THIS MODULE STORES SOLUTIONS TO OUTPUT LOC. TOGGLES FOR CALCULATING TRANSIENT AND STEADY SOLUTIONS. IT CAN
# GENERATE SOME PLOTS FOR ERROR ANALYSIS IF pp.verboseError IS TOGGLED. THESE PLOTS ARE RENDERED BUT NOT SAVED.


class gmres_counter(object):
    """A class object that can be called during GMRES to print stepwise iterative residual."""
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


def gaas_inverse_relaxation_operator(b, matrix_sc, matrix_fd, kptdf, field, freq):
    """Generalized minimal residual solver for calculating the solution to the matix equation Ax = b, where b is to be
    specified and A is the transient relaxation operator (time + drift + scm). There are many problems that can be cast
    to this type, solving the steady solution to the BTE, solving the transient solution to the BTE, solving for the
    effective distribution. It is convenient to have a single structure to do these calculations and only vary the
    forcing vectors (b) and frequencies (freq).

    Parameters:
        b (nparray) : The forcing for the matrix equation.
        matrix_sc (memmap): Scattering matrix in simple linearization by default..
        matrix_fd (memmap): Finite difference matrix, generated by apply_centraldiff_matrix.
        kptdf (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        field (dbl): Value of the electric field in V/m.
        freq (dbl): Frequency in GHz to be used for the transient solution
    Returns:
        x_next (nparray): Numpy array containing the (hopefully) converged iterative solution as chi.
        x_0 (nparray): Numpy array containing the RTA solution as chi.
        counter.niter (dbl): Number of iterations to reach desired relative convergence
    """
    counter = gmres_counter()
    print('Starting inverse relaxation solver for {:.3E} and {:.3E} GHz'.format(field,freq))
    if pp.scmBool:
        scmfac = pp.scmVal
        print('Applying correction factor to the scattering matrix.')
    else:
        scmfac = 1
        print('Not applying correction factor to the scattering matrix.')
    _,_,_,matrix_fd = gaas_gamma_fdm(matrix_fd, kptdf, field)
    if pp.derL:
        _,_,_,matrix_fd = gaas_l_fdm(matrix_fd, kptdf, field)
    loopstart = time.time()
    chi2psi = np.squeeze(kptdf['k_FD'] * (1 - kptdf['k_FD']))
    invdiag = (np.diag(matrix_sc) * scmfac) ** (-1)
    x_0 = b * invdiag
    tau_0 = (np.sum(np.diag(matrix_sc)*kptdf['k_FD'])/np.sum(kptdf['k_FD']))**(-1)

    x_smrta = b * np.ones(len(kptdf))*tau_0
    if freq > 0:
        freq_matrix = np.diag(np.ones(len(kptdf))*1j*10**9*2*np.pi*freq)  # Positive quantity
        print('Starting GMRES solver.')
        x_next, criteria = linalg.gmres(freq_matrix+matrix_sc*scmfac-matrix_fd, b,x0=x_0,tol=pp.relConvergence,
                                        callback=counter,atol=pp.absConvergence)
    if freq == 0:
        print('Starting GMRES solver.')
        x_next, criteria = linalg.gmres(matrix_sc*scmfac-matrix_fd, b,x0=x_0,tol=pp.relConvergence,
                                        callback=counter,atol=pp.absConvergence)
        freq_matrix = np.diag(np.zeros(len(kptdf)))
    print('GMRES convergence criteria: {:3E}'.format(criteria))
    # The following step is the calculation of the relative residual, which involves another MVP. This adds expense. If
    # we're confident in the convergence, we can omit this check to increase speed.
    if pp.verboseError:
        b_check = np.dot(freq_matrix+matrix_sc*scmfac-matrix_fd,x_next)
        error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
        print('Norm of b is {:3E}'.format(np.linalg.norm(b)))
        print('Absolute residual error is {:3E}'.format(np.linalg.norm(b_check-b)))
        print('Relative residual error is {:3E}'.format(error))
    else:
        error = 0
        print('Error not stored.')
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if not pp.simpleBool:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in canonical linearization')
        x_next = x_next * chi2psi
        x_0 = x_0 * chi2psi
        x_smrta = x_smrta * chi2psi
    return x_next, x_smrta, error, counter.niter


def gaas_gamma_fdm(matrix,fullkpts_df,E):
    """Calculate a finite difference matrix using the specified difference stencil and apply bc. In the current version,
    it is only applied in the kx direction, but it is easy to generalize for a general axis. Only applied to the Gamma
    valley of GaAs.

    Parameters:
        matrix (memmap): Scattering matrix in simple linearization.
        fullkpts_df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        E (dbl): Value of the electric field in V/m.
        scm (memmap): Memory-mapped array containing the scattering matrix, assumed simple linearization by default.
    Returns:
        shortslice_inds (numpyarray): Array of indices associated with the points along ky-kz lines with fewer than 5 pts.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        matrix (memmap): Memory-mapped array containing the modified scattering matrix, accounting for the FDM.
    TODO: CURRENTLY THE SPACING PARAMETER IS HARDCODED FOR DIFFERENT GRID SIZES. WOULD BE NICE TO MAKE THIS MORE ROBUST
    IN THE FUTURE.
    """
    # Do not  flush the memmap it will overwrite consecutively.
    # Get the first and last rows since these are different because of the IC. Go through each.
    # Get the unique ky and kz values from the array for looping.
    # This is not robust and should be replaced.
    if pp.kgrid == 160:
        step_size = 0.0070675528500652425 * 1E10  # 1/Angstrom to 1/m (for 160^3)
    if pp.kgrid == 200:
        step_size = 0.005654047459752398 * 1E10  # 1/Angstrom to 1/m (for 200^3)
    if pp.kgrid == 80:
        step_size = 0.0070675528500652425*2*1E10  # 1/Angstron for 1/m (for 80^3)
    print('Generating GaAs Gamma Valley finite difference matrix using ' + pp.fdmName)

    print('Step size is {:.3f} inverse Angstroms'.format(step_size / 1e10))
    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]']]
    g_inds,l_inds,x_inds=utilities.gaas_split_valleys(fullkpts_df,False)
    g_df = kptdata.loc[g_inds]  # Only apply condition in the Gamma valley
    uniq_yz = np.unique(g_df[['ky [1/A]', 'kz [1/A]']].values, axis=0)

    # If there are too few points in a slice < 4, we want to keep track of those points
    shortslice_inds = []
    l_icinds = []
    r_icinds = []

    start = time.time()
    # Loop through the unique ky and kz values in the Gamma valley. Right now field applied only in [1 0 0]
    for i in range(len(uniq_yz)):
        ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
        # Grab the "slice" of points in k space with the same ky and kz coordinate
        slice_df = g_df.loc[(g_df['ky [1/A]'] == ky) & (g_df['kz [1/A]'] == kz)]
        slice_inds = slice_df['k_inds'].values-1
        if len(slice_inds) > 3:
            # Subset is the slice sorted by kx value in ascending order. The index of subset still references kptdata.
            subset = slice_df.sort_values(by=['kx [1/A]'], ascending=True)
            ordered_inds = subset['k_inds'].values - 1  # indices of matrix (zero indexed)
            l_icinds.append(ordered_inds[0])  # +1 to get the k_inds values (one indexed)
            r_icinds.append(ordered_inds[-1])
            last = len(ordered_inds) - 1
            slast = len(ordered_inds) - 2
            if pp.fdmName == 'Column Preserving Central Difference':
                # Set the "initial condition" i.e. the point with the most negative kx value is treated as being zero
                # (and virtual point below)
                matrix[ordered_inds[0], ordered_inds[1]] += 1/(2*step_size)*c.e*E/c.hbar_joule
                matrix[ordered_inds[1], ordered_inds[2]] += 1/(2*step_size)*c.e*E/c.hbar_joule
                # Set the other "boundary condition" i.e. the point with the most positive kx value is treated as being zero
                # (and virtual point above)
                matrix[ordered_inds[last], ordered_inds[slast]] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule
                matrix[ordered_inds[slast], ordered_inds[slast-1]] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule
                # Set the value of all other points in the slice
                inter_inds = ordered_inds[2:slast]
                inter_inds_up = ordered_inds[3:last]
                inter_inds_down = ordered_inds[1:slast-1]
                matrix[inter_inds, inter_inds_up] += 1/(2*step_size)*c.e*E/c.hbar_joule
                matrix[inter_inds, inter_inds_down] += -1 * 1/(2*step_size)*c.e*E/c.hbar_joule
            if pp.fdmName == 'Backwards Difference':
                # Set the "initial condition" i.e. the point with the most negative kx value has virtual point below
                # that is assumed to be zero
                inter_inds = ordered_inds[1:last+1]
                inter_inds_down = ordered_inds[0:last]
                matrix[ordered_inds, ordered_inds] += 1/(step_size)*c.e*E/c.hbar_joule
                matrix[inter_inds, inter_inds_down] += -1 * 1/(step_size)*c.e*E/c.hbar_joule
            if pp.fdmName == 'Hybrid Difference':
                matrix[ordered_inds[0],ordered_inds[0]] = -1 * 1/(step_size)*c.e*E/c.hbar_joule
                matrix[ordered_inds[0], ordered_inds[1]] = 1 / (step_size) * c.e * E / c.hbar_joule
                matrix[ordered_inds[1], ordered_inds[0]] = -1 / (2*step_size) * c.e * E / c.hbar_joule
                matrix[ordered_inds[slast],ordered_inds[last]] = 1/(2*step_size)*c.e*E/c.hbar_joule
        else:
            shortslice_inds.append(slice_inds)
    print('Scattering matrix modified to incorporate central difference contribution.')
    shortslice_inds = np.concatenate(shortslice_inds,axis=0)
    print('Not applied to {:d} Gamma points because fewer than 4 points on the slice.'.format(len(shortslice_inds)))
    print('This represents {:1f} % of points in the Gamma valley.'.format(len(shortslice_inds)/len(g_df)*100))
    end = time.time()
    print('Finite difference generation took {:.2f}s'.format(end - start))

    return shortslice_inds, np.array(l_icinds), np.array(r_icinds), matrix


def gaas_l_fdm(matrix,fullkpts_df,E):
    """Calculate a finite difference matrix using the specified difference stencil and apply bc. In the current version,
    it is only applied in the kx direction, but it is easy to generalize for a general axis. Only applied to the L
    valleys of GaAs.

    Parameters:
        matrix (memmap): Scattering matrix in simple linearization.
        fullkpts_df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        E (dbl): Value of the electric field in V/m.

    Returns:
        shortslice_inds (numpyarray): Array of indices associated with the points along ky-kz lines with fewer than 5 pts.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        np.array(icinds) (numpyarray): Array of indices associated with the points where the ic is applied.
        matrix (memmap): Memory-mapped array containing the modified scattering matrix, accounting for the FDM.

    TODO: CURRENTLY THE SPACING PARAMETER IS HARDCODED FOR DIFFERENT GRID SIZES. WOULD BE NICE TO MAKE THIS MORE ROBUST
    IN THE FUTURE. WEEKLY UPDATES HAVE SHOWN THAT THE CENTRAL DIFFERENCE FDM APPROX. IS NOT GREAT IN THE L VALLEY.
    AT THE MOMENT, THE SCATTERING MATRIX APPROX DELTA F << F0 IS BAD IF THERE ARE ANY ELECTRONS IN THE L SO THIS IS
    PROBABLY MORE PRESSING THAN THE FDM BOUNDARY CONDITIONS AT THE MOMENT.
    """
    # Do not  flush the memmap it will overwrite consecutively.
    # Get the first and last rows since these are different because of the IC. Go through each.
    # Get the unique ky and kz values from the array for looping.
    # This is not robust and should be replaced.
    if pp.kgrid == 160:
        step_size = 0.0070675528500652425 * 1E10  # 1/Angstrom to 1/m (for 160^3)
    if pp.kgrid == 200:
        step_size = 0.005654047459752398 * 1E10  # 1/Angstrom to 1/m (for 200^3)
    if pp.kgrid == 80:
        step_size = 0.0070675528500652425*2*1E10  # 1/Angstron for 1/m (for 80^3)
    print('Step size is {:.3f} inverse Angstroms'.format(step_size/1e10))
    kptdata = fullkpts_df[['k_inds', 'kx [1/A]', 'ky [1/A]', 'kz [1/A]','energy [eV]']]
    _,L_inds,_=utilities.gaas_split_valleys(kptdata,False)
    l1_inds,l2_inds,l3_inds,l4_inds,l5_inds,l6_inds,l7_inds,l8_inds = utilities.split_L_valleys(kptdata,False)
    L_list = [l1_inds,l2_inds,l3_inds,l4_inds,l5_inds,l6_inds,l7_inds,l8_inds]
    shortslice_inds = []
    l_icinds = []
    r_icinds = []
    if pp.fdmName == 'Hybrid Difference':
        print('Applying hybrid FDM scheme.')
    for i1 in range(len(L_list)):
        print('Applying to {} L valley'.format(i1))
        l_df = kptdata.loc[L_list[i1]]
        uniq_yz = np.unique(l_df[['ky [1/A]', 'kz [1/A]']].values, axis=0)
        # If there are too few points in a slice < 5, we want to keep track of those points
        start = time.time()
        # Loop through the unique ky and kz values in the Gamma valley
        for i in range(len(uniq_yz)):
            kind = i + 1
            ky, kz = uniq_yz[i, 0], uniq_yz[i, 1]
            # Grab the "slice" of points in k space with the same ky and kz coordinate
            slice_df = l_df.loc[(l_df['ky [1/A]'] == ky) & (l_df['kz [1/A]'] == kz)]
            slice_inds = slice_df['k_inds'].values
            if len(slice_inds) > 3:
                # Subset is the slice sorted by kx value in ascending order. The index of subset still references kptdata.
                subset = slice_df.sort_values(by=['kx [1/A]'], ascending=True)
                ordered_inds = subset['k_inds'].values - 1  # indices of matrix (zero indexed)
                l_icinds.append(ordered_inds[0])  # +1 to get the k_inds values (one indexed)
                r_icinds.append(ordered_inds[-1])
                last = len(ordered_inds) - 1
                slast = len(ordered_inds) - 2
                if pp.fdmName == 'Column Preserving Central Difference':
                    # Set the "initial condition" i.e. point with most negative kx value is treated as being zero
                    # (and virtual point below)
                    matrix[ordered_inds[0], ordered_inds[1]] += 1 / (2 * step_size) * c.e * E / c.hbar_joule
                    matrix[ordered_inds[1], ordered_inds[2]] += 1 / (2 * step_size) * c.e * E / c.hbar_joule
                    # Set the other "boundary condition" i.e. point with most positive kx value is treated as being zero
                    # (and virtual point above)
                    matrix[ordered_inds[last], ordered_inds[slast]] += -1 * 1 / (2 * step_size) * c.e * E / c.hbar_joule
                    matrix[ordered_inds[slast], ordered_inds[slast - 1]] += -1 * 1 / (2 * step_size) *c.e*E/c.hbar_joule
                    # Set the value of all other points in the slice
                    inter_inds = ordered_inds[2:slast]
                    inter_inds_up = ordered_inds[3:last]
                    inter_inds_down = ordered_inds[1:slast - 1]
                    matrix[inter_inds, inter_inds_up] += 1 / (2 * step_size) * c.e * E / c.hbar_joule
                    matrix[inter_inds, inter_inds_down] += -1 * 1 / (2 * step_size) * c.e * E / c.hbar_joule
                if pp.fdmName == 'Backwards Difference':
                    # Set the "initial condition" i.e. the point with the most negative kx value has virtual point below
                    # that is assumed to be zero
                    inter_inds = ordered_inds[1:last + 1]
                    inter_inds_down = ordered_inds[0:last]
                    matrix[ordered_inds, ordered_inds] += 1 / (step_size) * c.e * E / c.hbar_joule
                    matrix[inter_inds, inter_inds_down] += -1 * 1 / (step_size) * c.e * E / c.hbar_joule
            else:
                shortslice_inds.append(slice_inds)
            if kind % 10 == 0:
                pass
    print('Scattering matrix modified to incorporate central difference contribution.')
    shortslice_inds = np.concatenate(shortslice_inds, axis=0)
    print('Not applied to {:d} L valley points because fewer than 4 points on the slice.'.format(len(shortslice_inds)))
    print('This represents {:1f} % of points in the L valley.'.format(len(shortslice_inds) / len(fullkpts_df.loc[L_inds]) * 100))
    print('Finite difference applied to L valleys.')
    if not pp.getX:
        pass
    else:
        print('Finite difference not applied to X valleys. Derivative treated as zero for these points.')
    end = time.time()
    print('Finite difference generation took {:.2f}s'.format(end - start))

    return shortslice_inds, np.array(l_icinds), np.array(r_icinds), matrix


def steady_low_field(df, scm):
    """GMRES solver hard coded for solving the BTE in low field approximation. Sign convention by Wu Li. Returns F,
    which is equal to chi/(eE/kT). Since the finite difference matrix is not applied and the solution is calculated as F.
    we have a separate operation to calculate low field approximation quantities.

    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        scm (memmap): Memory-mapped array containing the scattering matrix, assumed simple linearization by default.
    Returns:
        f_next (nparray): Numpy array containing the (hopefully) converged GMRES solution as psi/(eE/kT).
        f_0 (nparray): Numpy array containing the RTA solution as psi_0/(eE/kT).
    """
    counter = gmres_counter()
    print('Starting steady BTE low-field occupancy solver')
    if pp.scmBool:
        scmfac = pp.scmVal
        print('Applying correction factor to the scattering matrix.')
    else:
        scmfac = 1
        print('Not applying correction factor to the scattering matrix.')
    loopstart = time.time()
    invdiag = (np.diag(scm) * scmfac) ** (-1)
    b = (-1) * np.squeeze(df['vx [m/s]'] * df['k_FD']) * (1 - df['k_FD'])
    f_0 = b * invdiag
    f_next, criteria = linalg.gmres(scm*scmfac, b, x0=f_0, tol=pp.relConvergence, atol=pp.absConvergence,
                                    callback=counter)
    print('GMRES convergence criteria: {:3E}'.format(criteria))
    if pp.verboseError:
        b_check = np.dot(scm*scmfac,f_next)
        error = np.linalg.norm(b_check - b)/np.linalg.norm(b)
        print('Norm of b is {:3E}'.format(np.linalg.norm(b)))
        print('Absolute residual error is {:3E}'.format(np.linalg.norm(b_check-b)))
        print('Relative residual error is {:3E}'.format(error))
    else:
        error = 0
        print('Error not stored.')
    loopend = time.time()
    print('Convergence took {:.2f}s'.format(loopend - loopstart))
    if pp.simpleBool:
        # Return chi in all cases so there's not confusion in plotting
        print('Converting psi to chi since matrix in simple linearization')
        chi2psi = np.squeeze(df['k_FD'] * (1 - df['k_FD']))
        f_next = f_next / chi2psi
        f_0 = f_0 / chi2psi
    return f_next, f_0, error, counter.niter


def write_steady(fieldVector, df):
    """Calls the GMRES solver hard coded for solving the BTE with full FDM and writes the chis to file. Also calculates
    low-field and RTA chis.
    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
    Returns:
        None. Just writes the chi solutions to file. chi_#_EField_freq. #0 corresponds to single mode RTA with full
        finite difference, #1 corresponds to RTA with low-field, #2 corresponds to low-field with full scattering, #3
        corresponds to full finite difference and full scattering.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    error = []
    iteration_count = []
    f_next, f_0,_,_ = steady_low_field(df, scm)
    np.save(pp.outputLoc + 'Steady/' + 'f_1',f_0)
    np.save(pp.outputLoc + 'Steady/' + 'f_2',f_next)
    tau_0 = -np.sum(np.diag(scm)**(-1)*df['k_FD'])/np.sum(df['k_FD'])
    print('The thermally averaged relaxation time is {:.1f} ps'.format(tau_0*1e12))
    for ee in fieldVector:
        fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
        b = (-1) * c.e * ee / c.kb_joule / pp.T * np.squeeze(df['vx [m/s]'] * df['k_FD']) * (1 - df['k_FD'])
        # Frequency set as zero for steady solutions in the inverse operator
        x_next,x_smrta, temp_error, iterations = gaas_inverse_relaxation_operator(b, scm, fdm, df, ee, 0)
        error.append(temp_error)
        iteration_count.append(iterations)
        del fdm
        np.save(pp.outputLoc + 'Steady/' + 'chi_' + '3_' + "E_{:.1e}".format(ee), x_next)
        np.save(pp.outputLoc + 'Steady/' + 'chi_' + '2_' + "E_{:.1e}".format(ee), utilities.f2chi(f_next,df,ee))
        np.save(pp.outputLoc + 'Steady/' + 'chi_' + '1_' + "E_{:.1e}".format(ee), utilities.f2chi(f_0,df, ee))
        np.save(pp.outputLoc + 'Steady/' + 'chi_' + '0_' + "E_{:.1e}".format(ee), x_smrta)
        print('Steady occupation solutions written to file for ' + "{:.1e} V/m ".format(ee))
        print('\n \n')
    if pp.verboseError:
        plt.figure()
        plt.plot(fieldVector*1E-5,error)
        plt.xlabel('EField (kV/cm)')
        plt.ylabel(r'$|Ax_{f}-b|/|b|$')
        plt.title('Steady Occupation' + pp.title_str)
    plt.figure()
    plt.plot(fieldVector*1E-5, iteration_count)
    plt.xlabel('EField (kV/cm)')
    plt.ylabel('Iterations to convergence')
    plt.title('Steady Occupation' + pp.title_str)


def write_transient(fieldVector, df, freqVector):
    """Calls the GMRES solver hard coded for solving the BTE with full FDM and writes the chis to file.
    Parameters:
        fieldVector (nparray): Vector containing the values of the electric field to be evaluated in V/m.
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
        freqVector (dbl): Vector containing frequency in GHz to be used for the transient solution
    Returns:
        None. Just writes the chi solutions to file. Right now only implemented for #3 corresponding to full finite
        difference with full scattering matrix.
    """
    nkpts = len(np.unique(df['k_inds']))
    scm = np.memmap(pp.inputLoc + pp.scmName, dtype='float64', mode='r', shape=(nkpts, nkpts))
    error = []
    iteration_count =[]
    for freq in freqVector:
        for ee in fieldVector:
            fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
            b = (-1) * c.e * ee / c.kb_joule / pp.T * np.squeeze(df['vx [m/s]'] * df['k_FD']) * (1 - df['k_FD'])
            x_next, x_0, temp_error, iterations = gaas_inverse_relaxation_operator(b, scm, fdm, df, ee, freq)
            error.append(temp_error)
            iteration_count.append(iterations)
            del fdm
            np.save(pp.outputLoc + 'Transient/' + 'chi_' + '3_' + "f_{:.1e}_E_{:.1e}".format(freq,ee), x_next)
            print('Transient solution written to file for ' + "{:.1e} V/m and {:.1e} GHz".format(ee,freq))
            print('\n \n')
    if pp.verboseError:
        plt.figure()
        plt.plot(fieldVector*1E-5,error)
        plt.xlabel('EField (kV/cm)')
        plt.ylabel(r'$|Ax_{f}-b|/|b|$')
        plt.title('Occupation {:.1e} GHz'.format(freq) + pp.title_str)
    # plt.figure()
    # plt.plot(fieldVector*1E-5, iteration_count)
    # plt.xlabel('EField (kV/cm)')
    # plt.ylabel('Iterations to convergence')
    # plt.title('Occupation {:.1e} GHz'.format(freq) + pp.title_str)


def write_icinds(df):
    """Write the location of the initial conditions to file and plot.
    Parameters:
        df (dataframe): Electron DataFrame indexed by kpt containing the energy associated with each state in eV.
    Returns:
        None. Just writes the icinds to file.
    """
    nkpts = len(np.unique(df['k_inds']))
    fdm = np.memmap(pp.inputLoc + '/finite_difference_matrix.mmap', dtype='float64', mode='w+', shape=(nkpts, nkpts))
    _, g_l_inds, g_r_inds, _ = gaas_gamma_fdm(fdm, df, 1)
    np.save(pp.outputLoc + 'Gamma_left_icinds', g_l_inds)
    np.save(pp.outputLoc + 'Gamma_right_icinds', g_r_inds)
    if pp.derL:
        _, l_l_inds, l_r_inds, _ = gaas_l_fdm(fdm, df, 1)
        np.save(pp.outputLoc + 'L_left_icinds', l_l_inds)  # All 8 L valley indices saved together
        np.save(pp.outputLoc + 'L_right_icinds', l_r_inds)  # All 8 L valley indices saved together



def writeOutputFile():
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    outF = open(pp.outputLoc+"Boltzmann_Transport_log.txt", "w")

    outF.write('Boltzmann Transport Equation solved at: ' + dt_string + '\n')
    outF.write('FDM scheme: ' + pp.fdmName + '\n')
    outF.write('SCM name: ' + pp.scmName + '\n')
    outF.write('GNRES rel: ' + str(pp.relConvergence) + '\n')
    outF.write('GMRES abs: ' + str(pp.absConvergence) + '\n')
    outF.write('X Valleys: ' + str(pp.getX) + '\n')
    outF.write('L derivative applied: ' + str(pp.derL) + '\n')
    outF.write('SCM Bool: ' + str(pp.scmBool) + '\n')
    outF.write('SCM Val: ' + str(pp.scmVal) + '\n')
    outF.close()


if __name__ == '__main__':
    # Create electron and phonon dataframes
    preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)
    fields = pp.small_signal_fields
    freqs = pp.freqVector

    print(utilities.calculate_density(electron_df))

    # Toggle to run calculations for transient or steady BTE solutions.
    writeTransient = False
    writeSteady = True
    writeIcinds = False
    if writeSteady:
        write_steady(fields, electron_df)
    if writeTransient:
        write_transient(fields, electron_df, freqs)
    if writeIcinds:
        write_icinds(electron_df)
    writeOutputFile()
    plt.show()

