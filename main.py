import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
from astropy.io import ascii
import os

# open source module found for converting MJD/JD to DateTimes
import jdutil as jd

# Ben's module -- THANKS BEN
import LCtools

# my preprocessing module
from preproc import *

import glob

################## START PREPROCESSING ###################
path_to_data = "TDSS_data/"

#lightcurve paths
lightcurves = glob.glob('{}*LC*.csv'.format(path_to_data))
# remove extras
lightcurves = [ x for x in lightcurves if "Phase" not in x]
lightcurves = [ x for x in lightcurves if "phase" not in x]

#spectra paths
spectra = glob.glob('{}*spec*.csv'.format(path_to_data))

lenpath = len(path_to_data)
#object names
names = [n[lenpath:-9] for n in spectra]

#preprocessing loop
for n in names:
    # Filepaths for each object
    lc_path = [ x for x in lightcurves if n in x][0]
    spec_path = [ x for x in spectra if n in x][0]
    
    #load into pandas dataframes
    lc = pd.read_csv(lc_path)
    spec = pd.read_csv(spec_path)
    
    #do the preprocessing
    time_preproc_lc = LC_timesort_preproc(lc)
    phase_preproc_lc = LC_phasesort_preproc(lc, bins=phase_bins())
    phasefit_preproc_lc = LC_phasefit_preproc(lc_path, bins=phase_bins())
    preproc_spec = spectra_preproc(spec, bins=wavelength_bins())

    #create output directory
    outdir = 'preproc/{}/'.format(n)
    if not os.path.exists(outdir):
        os.makedirs(outdir)    
    
    #save preprocessed files
    time_preproc_lc.to_csv("preproc/{}/{}_LC_timesort.csv".format(n,n), index=False)
    phase_preproc_lc.to_csv("preproc/{}/{}_LC_phasesort.csv".format(n,n), index=False)
    phasefit_preproc_lc.to_csv("preproc/{}/{}_LC_phasefit.csv".format(n,n), index=False)
    preproc_spec.to_csv("preproc/{}/{}_spec.csv".format(n,n), index=False)

#preprocessed .csvs should be saved in *workingdirectory*/preproc/ folder
