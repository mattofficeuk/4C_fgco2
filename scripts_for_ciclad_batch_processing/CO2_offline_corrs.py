process_data = False
process_corrs = True

import netCDF4
import os
import cartopy.crs as ccrs
import numpy as np
import scipy.interpolate.ndgriddata as ndgriddata
import pickle
import sys
from scipy import stats
import mfilter

data_dir = '/data/mmenary/fgco2'
var_names = ['dpco2', 'sfcWind', 'siconc', 'sos', 'tos', 'fgco2']
constant_vars = ['none', 'dpco2', 'sfcWind', 'siconc', 'sos', 'tos']
output_dir = '/home/mmenary/python/notebooks/CO2_Py3x/output'
year0s = [1850, 2100, 2350, 2600, 2850, 3100, 3350, 3600]

#########
year0 = 1850
oneConstant = 'False'
season = 1
########
nyrs = 250

def compute_fgco2(input_data, verbose=False):
    '''
    Reads in the various components and computes the fgco2.
    tos: DegC
    sos: PSU g/kg
    wind: m/s
    dpco2: Pa
    sea_ice: % (will be converted to fraction)
    '''
    ### Constantes
    rhop = 1025  ## What is this? According to Fortran, potential volumic mass [kg/m3]
#     constants = 1000 * 0.01 / 3600. * 0.251

    ## tos is potential temp., rather than in situ (required), but as we're only at surface it doesn't matter

    ### KO
    if verbose: print("Computing K0")
    tos_kel = input_data['tos'] + 273.15  # OK
    zcek1 = 9345.17/tos_kel - 60.2409 + 23.3585 * np.log(tos_kel*0.01) + input_data['sos'] * (0.023517 - 0.00023656 * tos_kel + 0.0047036e-4 * tos_kel * tos_kel)  ## OK
    chemc = np.exp(zcek1) * rhop / 1000   ## OK
    del tos_kel, zcek1  # save memory

    ### Schmidt Number
    if verbose: print("Computing Schmidt No.")
    ztc = np.minimum(35, input_data['tos'])  ## OK
    sch_co2 = 2116.8 - 136.25 * ztc + 4.7353 * ztc * ztc - 0.092307 * ztc * ztc * ztc + 0.0007555 * ztc * ztc * ztc *ztc  ## OK
    del ztc  # save memory

    ### Wind
    if verbose: print("Computing Wind squared")
    wind_contrib = input_data['sfcWind'] * input_data['sfcWind'] * 0.251 * 0.01 / 3600.
    sea_ice_contrib = (1 - input_data['siconc'] / 100)

    ### delta pCOs
    dpco2_2 = input_data['dpco2'] / 101325.  # From Pa to atm

    ## Notes:
    ## 0.251 (cm/h)(m/s)-2

    ### Calcul
    if verbose: print("Doing final calculation")
    fgco2_calc = wind_contrib * sea_ice_contrib * dpco2_2 * chemc * np.sqrt(660. / sch_co2) * 1e3  # molC/m2/s (Laurent)
    ## Not sure about this  1e3. But it could be conversion from litre to m3

    fgco2_calc *= (12 / 1000)  # kgC/m2/s
    return fgco2_calc

def regrid_to_ocean_grid(fld, lon_wind, lat_wind):
    '''
    Regrid surface winds
    '''
    nt, nj, ni = fld.shape

    lon_wind_shifted = lon_wind.copy()
    lon_wind_shifted[lon_wind_shifted >= 180] -= 360
    lon_wind2d = np.repeat(lon_wind_shifted[np.newaxis, :], nj, axis=0)
    lat_wind2d = np.repeat(lat_wind[:, None], ni, axis=1)

    nj2, ni2 = nav_lon.shape
    fld_regridded = np.ma.masked_all(shape=(nt, nj2, ni2))
    for tt in range(nt):
        fld_regridded[tt, :, :] = ndgriddata.griddata((lon_wind2d.flatten(), lat_wind2d.flatten()),
                                                      fld[tt, :, :].flatten(), (nav_lon, nav_lat), method="linear")

    return fld_regridded

def regrid_for_plotting(fld):
    '''
    Regrid for plotting
    '''
    fld_regridded = ndgriddata.griddata((nav_lon.flatten(), nav_lat.flatten()),
                                        fld.flatten(), (lon_re, lat_re), method="linear")
    mask_regridded = ndgriddata.griddata((nav_lon.flatten(), nav_lat.flatten()),
                                         fld.mask.flatten(), (lon_re, lat_re), method="linear")

    fld_regridded = np.ma.array(fld_regridded, mask=mask_regridded)

    return fld_regridded

def spatial_correlation(fgco2_fld, other_fld, testing=False, smoothing=1, detrend=False):
    nyrs, nj, ni = fgco2_fld.shape
    slope_fld = np.ma.masked_all(shape=(nj, ni))
    inter_fld = np.ma.masked_all(shape=(nj, ni))
    corr_fld  = np.ma.masked_all(shape=(nj, ni))
    for jj in range(nj):
        if testing:
            if jj < 250: continue
            if jj > 300: continue
        for ii in range(ni):
            if testing:
                if ii < 220: continue
                if ii > 300: continue
            if np.ma.is_masked(fgco2_fld[0, jj, ii]):
                continue
            fgco2_ts = fgco2_fld[:, jj, ii]
            other_ts = other_fld[:, jj, ii]

            if detrend:
                fgco2_longperiod = mfilter.smooth1d(fgco2_ts, 71, mask_ends=True)
                other_longperiod = mfilter.smooth1d(other_ts, 71, mask_ends=True)
                fgco2_ts = fgco2_ts - (fgco2_longperiod - fgco2_longperiod.mean())
                other_ts = other_ts - (other_longperiod - other_longperiod.mean())

            if smoothing > 1:
                fgco2_ts = mfilter.smooth1d(fgco2_ts, smoothing, mask_ends=True)
                other_ts = mfilter.smooth1d(other_ts, smoothing, mask_ends=True)

            real = np.nonzero(fgco2_ts * other_ts)
            slope, inter, corr, _, _ = stats.linregress(other_ts[real], fgco2_ts[real])
            slope_fld[jj, ii] = slope
            inter_fld[jj, ii] = inter
            corr_fld[jj, ii]  = corr

    return [slope_fld, inter_fld, corr_fld]

def get_t_indices(iyr2, season):
    if season == 0:
        tt0, tt1 = iyr2 * 12, (iyr2 + 1) * 12
    else:
        iseason = season - 1
        tt0, tt1 = iyr2 * 12 + iseason * 3, iyr2 * 12 + (iseason + 1) * 3
    return tt0, tt1

nj, ni = 180, 360
lon_re = np.repeat((np.arange(-180, 180) + 0.5)[np.newaxis, :], nj, axis=0)
lat_re = np.repeat((np.arange(-90, 90) + 0.5)[:, None], ni, axis=1)

# ============================
# This bit is not parallelised but slow, so it is also in a batch script: CO2_offline_corrs.py
# ============================
if process_corrs:
    detrend = sys.argv[1]
    if detrend == "True":
        detrend = True
    elif detrend == "False":
        detrend = False
    else:
        raise ValueError('Huh?')
    season = np.long(sys.argv[2])

    chunks = []
    nchunks_i = 4
    nchunks_j = 4

    nj2, ni2 = 331, 360
    di = np.long(np.ceil(ni2 / nchunks_i))
    dj = np.long(np.ceil(nj2 / nchunks_j))
    for ii in range(nchunks_i):
        ii0 = ii * di
        ii1 = min((ii + 1) * di, ni2)
        for jj in range(nchunks_j):
            jj0 = jj * dj
            jj1 = min((jj + 1) * dj, nj2)
            new_chunk = [ii0, ii1, jj0, jj1]
            chunks.append(new_chunk)

    print("Chunks",  chunks)
    print("Total chunks = {:d}".format(nchunks_i * nchunks_j))

    remake_corr_save = False

    if detrend:
        detrend_string = '_detrended71Y'
    else:
        detrend_string = ''

    if season == 0:
        season_string = ''
    else:
        season_string = '_Season{:d}'.format(season)

    save_file_corrs = '/data/mmenary/python_saves/CO2_ConstantVarCorrs{:s}{:s}.pkl'.format(detrend_string, season_string)
    print(save_file_corrs)
    if os.path.isfile(save_file_corrs) and not remake_corr_save:
        print("Loading corr save file: {:s}".format(save_file_corrs))
        with open(save_file_corrs, 'rb') as handle:
            slopes_oneConstant, inters_oneConstant, corrs_oneConstant, slopes_oneVary, inters_oneVary, \
            corrs_oneVary, smoothings, nav_lon, nav_lat = pickle.load(handle)
    else:
        smoothings = [1, 3, 5, 7, 11, 15, 21, 31, 41, 51, 71]
        slopes_oneConstant, inters_oneConstant, corrs_oneConstant = {}, {}, {}
        slopes_oneVary, inters_oneVary, corrs_oneVary = {}, {}, {}
        for var in constant_vars:
            slopes_oneConstant[var] = np.ma.masked_all(shape=(len(smoothings), nj2, ni2))
            inters_oneConstant[var] = np.ma.masked_all(shape=(len(smoothings), nj2, ni2))
            corrs_oneConstant[var]  = np.ma.masked_all(shape=(len(smoothings), nj2, ni2))
            slopes_oneVary[var] = np.ma.masked_all(shape=(len(smoothings), nj2, ni2))
            inters_oneVary[var] = np.ma.masked_all(shape=(len(smoothings), nj2, ni2))
            corrs_oneVary[var]  = np.ma.masked_all(shape=(len(smoothings), nj2, ni2))

        for ichunk, chunk in enumerate(chunks):
            ###################################################
    #         if ichunk != 15: continue

            ii0, ii1, jj0, jj1 = chunk
            fgco2_ann_oneConstant = {}
            fgco2_ann_oneVary = {}
            for var in constant_vars:
                fgco2_ann_oneConstant[var] = np.ma.masked_all(shape=(2000, jj1-jj0, ii1-ii0))
                fgco2_ann_oneVary[var] = np.ma.masked_all(shape=(2000, jj1-jj0, ii1-ii0))

            print("Working on chunk: [{:d}, {:d}, {:d}, {:d}]".format(ii0, ii1, jj0, jj1))
            for iyr, year0 in enumerate(year0s):
                ###################################################
    #             if iyr > 0: continue
                year1 = year0 + nyrs
                tt0 = year0 - 1850
                tt1 = year1 - 1850
                save_file_constant = '/data/mmenary/python_saves/CO2_OneConstant_{:d}-{:d}{:s}.pkl'.format(year0, year1, season_string)
                save_file_vary = '/data/mmenary/python_saves/CO2_OneVaries_{:d}-{:d}{:s}.pkl'.format(year0, year1, season_string)

                if os.path.isfile(save_file_constant):
                    print("Loading: {:s}".format(save_file_constant))
                    with open(save_file_constant, 'rb') as handle:
                        fgco2_ann_oneConstant_in, _, nav_lon, nav_lat = pickle.load(handle)
                    for key in constant_vars:
                        fgco2_ann_oneConstant[key][tt0:tt1, :, :] = fgco2_ann_oneConstant_in[key][:, jj0:jj1, ii0:ii1]
                        fgco2_ann_oneConstant_in[key] = 0
                else:
                    print("Could not find: {:s}".format(save_file_constant))

                if os.path.isfile(save_file_vary):
                    print("Loading: {:s}".format(save_file_vary))
                    with open(save_file_vary, 'rb') as handle:
                        fgco2_ann_oneVary_in, _, nav_lon, nav_lat = pickle.load(handle)
                    for key in constant_vars:
                        fgco2_ann_oneVary[key][tt0:tt1, :, :] = fgco2_ann_oneVary_in[key][:, jj0:jj1, ii0:ii1]
                        fgco2_ann_oneVary_in[key] = 0
                else:
                    print("Could not find: {:s}".format(save_file_vary))

            for var in constant_vars:
                if var == 'none':
                    continue
                print('Correlations for: {:s}'.format(var))
                for ismooth, smoothing in enumerate(smoothings):
                    slope, inter, corr = spatial_correlation(fgco2_ann_oneConstant['none'], fgco2_ann_oneConstant[var],
                                                             testing=False, smoothing=smoothing, detrend=detrend)
                    slopes_oneConstant[var][ismooth, jj0:jj1, ii0:ii1] = slope
                    inters_oneConstant[var][ismooth, jj0:jj1, ii0:ii1] = inter
                    corrs_oneConstant[var][ismooth, jj0:jj1, ii0:ii1]  = corr

                    ## Note that the target is fgco2_ann_oneConstant['none'] as that has nothing constant
                    slope, inter, corr = spatial_correlation(fgco2_ann_oneConstant['none'], fgco2_ann_oneVary[var],
                                                             testing=False, smoothing=smoothing, detrend=detrend)
                    slopes_oneVary[var][ismooth, jj0:jj1, ii0:ii1] = slope
                    inters_oneVary[var][ismooth, jj0:jj1, ii0:ii1] = inter
                    corrs_oneVary[var][ismooth, jj0:jj1, ii0:ii1]  = corr

        with open(save_file_corrs, 'wb') as handle:
            print("Saving to: {:s}".format(save_file_corrs))
            pickle.dump([slopes_oneConstant, inters_oneConstant, corrs_oneConstant,
                         slopes_oneVary, inters_oneVary, corrs_oneVary, smoothings, nav_lon, nav_lat],
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Done!")
