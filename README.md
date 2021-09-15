# SDU-preproc

## Sensing the Dynamic Universe project

The Sensing the Dynamic Universe (SDU) project seeks to make accessible to the widest possible audience the marvelous menagerie of celestial variables, with a taste of the astrophysics that makes them vary. SDU wants to help you see, hear, and understand many types of variables, including single stars, binary stars, neutron stars, black holes, supernovae, quasars.

For SDU, the sonification of astronomical data is done using [sonoUno](https://github.com/sonounoteam/sonouno), a software tool written in Python with some important built-in features. 

## Preprocessing

Our raw lightcurve and spectra data was obtained from the Time-Domain Spectroscopic Survey -- a subproject of the Sloan Digital Sky Survey, which is perhaps the most successful observational surveys in the history of astronomy. From the lightcurves and spectral, we hope to obtain four types of sonified videos:

1. Lightcurve sorted by time of observation
2. Phased lightcurve
3. Fourier decomposed, phase fitted lightcurve
4. Spectrum

Therefore, we preprocess the raw data in certain ways to ensure that the inputs to sonoUno will generate videos that are both scientifically accurate and aesthetically pleasing. That is the purpose of this repository.

## Running the Code

**To run the code, the user must have _astropy_ installed**

To run the preprocessing code, simply move into this directory and run:

```bash
python preproc.py
```

The four preprocessed .csvs for each type of object, ready to be input into sonoUno, should be 

## To-Do

- Make the output names of the preprocessed .csv files (and the column names within) more polished, so that the automatic titles and labels that sonoUno generates are publication ready.

- Currently, the preprocessed time-sorted LC is automatically resampled to a timebase of days. This can be unwieldy and generate sonified videos that are too long for our purposes. In the future, perhaps we could automate the code to vary the time-base depending on the length of time that observations span?

## Notes

- If the path to the raw data changes, change the `path_to_data` string within `preproc.py`

- `preproc.py` is written to match the naming structure of the .csv files within the TDSS_data folder. When adding data, please stick to the conventions
    - Lightcurves:  *"{object acronym}_LC\_{something}.csv"*
    - Spectra:  *"{object acronym}_spec.csv"*

- This code is tailored towards _periodic_ variable objects. _Non-periodic_ variables should be treated differently. This code will still most likely be a good start for preprocessing those as well.
