import numpy as np
import pandas as pd
import time
import constants as c
import utilities
import problemparameters as pp
import matplotlib.pyplot as plt
import occupation_solver
import occupation_plotter


if __name__ == '__main__':
    # Create electron and phonon dataframes
    # preprocessing.create_el_ph_dataframes(pp.inputLoc, overwrite=True)
    electron_df, phonon_df = utilities.load_el_ph_data(pp.inputLoc)
    electron_df = utilities.fermi_distribution(electron_df)

    fields = np.array([1])
    freq = pp.freqGHz

    writeSteady = True

    if writeSteady:
        occupation_solver.write_stegit mady(fields, electron_df)
    plt.show()

    occupation_plotter.bz_3dscatter(electron_df,True,True)
