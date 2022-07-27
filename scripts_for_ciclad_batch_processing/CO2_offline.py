process_data = True
process_corrs = False

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
# This bit is parallelised in a batch script: CO2_offline.py
# ============================
year0 = np.long(sys.argv[1])
oneConstant = sys.argv[2]
season = np.long(sys.argv[3])
if process_data:
    if oneConstant == 'True':
        oneConstant = True
    else:
        oneConstant = False

    if season == 0:
        season_string = ''
    else:
        season_string = '_Season{:d}'.format(season)

    year1 = year0 + nyrs
    if oneConstant:
        save_file = '/data/mmenary/python_saves/CO2_OneConstant_{:d}-{:d}{:s}.pkl'.format(year0, year1, season_string)
    else:
        save_file = '/data/mmenary/python_saves/CO2_OneVaries_{:d}-{:d}{:s}.pkl'.format(year0, year1, season_string)

    years_full = 1850 + np.arange(2000)
    years = year0 + np.arange(nyrs)

    ##################
    remake_first_save = True

    print(save_file)
    if os.path.isfile(save_file) and not remake_first_save:
        with open(save_file, 'rb') as handle:
            fgco2_ann_oneConstantVary, fgco2_ann_online, nav_lon, nav_lat = pickle.load(handle)
    else:
        # =================
        # Make monthly climatologies
        # =================
        file_end = '_MonClim.nc'
        clim_data = {}
        for ivar, var_name in enumerate(var_names):
            file_name = os.path.join(data_dir, var_name + file_end)
            print(file_name)

            loaded = netCDF4.Dataset(file_name)

            if var_name == 'sfcWind':
                wind_raw = loaded.variables[var_name][:]
                lat_wind = loaded.variables['lat'][:]
                lon_wind = loaded.variables['lon'][:]
                clim_data[var_name] = regrid_to_ocean_grid(wind_raw, lon_wind, lat_wind)
            else:
                nav_lat = loaded.variables['nav_lat'][:-1, 1:-1]
                nav_lon = loaded.variables['nav_lon'][:-1, 1:-1]
                clim_data[var_name] = loaded.variables[var_name][:, :-1, 1:-1]

        # =================
        # Compute the ann mean from online fgco2
        # =================
        file_end = '_temp.nc'
        nj, ni = nav_lon.shape
        fgco2_ann_online = np.ma.masked_all(shape=(nyrs, nj, ni))
        for iyr, year in enumerate(years):
            if year not in years_full:
                continue

            print('Processing (fgco2 online): {:d}'.format(year))
            iyr2 = iyr + (year0 - 1850)
            tt0, tt1 = get_t_indices(iyr2, season)

            file_name = os.path.join(data_dir, 'fgco2' + file_end)
            loaded = netCDF4.Dataset(file_name)
            input_data = loaded.variables['fgco2'][tt0:tt1, :-1, 1:-1]

            fgco2_ann_online[iyr, :, :] = input_data.mean(axis=0)

        # =================
        # Prepare dictionaries
        # =================
        fgco2_ann_oneConstantVary = {}
        for constant_var in constant_vars:
            nj, ni = nav_lon.shape
            fgco2_ann_oneConstantVary[constant_var] = np.ma.masked_all(shape=(nyrs, nj, ni))

        # =================
        # Compute ann mean fgco2 contributions (from mon mean contributions)
        # =================
        if oneConstant:
            for constant_var in constant_vars:
                for iyr, year in enumerate(years):
                    if year not in years_full:
                        continue

                    print('Processing (one constant) {:s}: {:d}'.format(constant_var, year))
                    iyr2 = iyr + (year0 - 1850)
                    tt0, tt1 = get_t_indices(iyr2, season)

                    input_data = {}
                    for ivar, var_name in enumerate(var_names):
                        if constant_var == var_name:
                            if season == 0:
                                input_data[var_name] = clim_data[var_name].copy()
                            else:
                                cc0 = (season - 1) * 3
                                cc1 = cc0 + 3
                                input_data[var_name] = clim_data[var_name][cc0:cc1, :, :].copy()
                        else:
                            file_name = os.path.join(data_dir, var_name + file_end)
                            loaded = netCDF4.Dataset(file_name)
                            if var_name == 'sfcWind':
                                input_data[var_name] = loaded.variables[var_name][tt0:tt1, :, :]
                                input_data[var_name] = regrid_to_ocean_grid(input_data[var_name], lon_wind, lat_wind)
                            else:
                                input_data[var_name] = loaded.variables[var_name][tt0:tt1, :-1, 1:-1]

                    fgco2_ann_oneConstantVary[constant_var][iyr, :, :] = compute_fgco2(input_data).mean(axis=0)

        else:
            for constant_var in constant_vars:
                for iyr, year in enumerate(years):
                    if year not in years_full:
                        continue

                    print('Processing (one varies) {:s}: {:d}'.format(constant_var, year))
                    iyr2 = iyr + (year0 - 1850)
                    tt0, tt1 = get_t_indices(iyr2, season)

                    input_data = {}
                    for ivar, var_name in enumerate(var_names):
                        if constant_var == var_name:
                            file_name = os.path.join(data_dir, var_name + file_end)
                            loaded = netCDF4.Dataset(file_name)
                            if var_name == 'sfcWind':
                                input_data[var_name] = loaded.variables[var_name][tt0:tt1, :, :]
                                input_data[var_name] = regrid_to_ocean_grid(input_data[var_name], lon_wind, lat_wind)
                            else:
                                input_data[var_name] = loaded.variables[var_name][tt0:tt1, :-1, 1:-1]
                        else:
                            if season == 0:
                                input_data[var_name] = clim_data[var_name].copy()
                            else:
                                cc0 = (season - 1) * 3
                                cc1 = cc0 + 3
                                input_data[var_name] = clim_data[var_name][cc0:cc1, :, :].copy()

                    fgco2_ann_oneConstantVary[constant_var][iyr, :, :] = compute_fgco2(input_data).mean(axis=0)

        with open(save_file, 'wb') as handle:
            pickle.dump([fgco2_ann_oneConstantVary, fgco2_ann_online, nav_lon, nav_lat],
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Done!")
