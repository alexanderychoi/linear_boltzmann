#!/usr/bin/env python
import preprocessing_largegrid
import numpy as np
import multiprocessing as mp
from functools import partial
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import constants as c


def relaxation_times_parallel(k,nlambda):
