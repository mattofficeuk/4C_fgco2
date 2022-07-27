print("Starting!")

#########  GISS has a different latitude dimension for tos compared to the other variables
#########  It is -89, -87, -85, -83, -81, -79, ...
#########  compared to -90, -88.5, -87.5, -86.5, -85.5, -84.5, ...
#########  I have used remapbil to regrid the data BEFORE processing. Far too complicated otherwise

import netCDF4
import os
import numpy as np
import pickle
import sys
from scipy import stats
from sklearn import linear_model
import statsmodels.api as sm
import mfilter

block_number = np.long(sys.argv[1])  ###################################################################
# block_number = 112

model = sys.argv[2]

smoothing = 5

detrending = 100

## This results in using seasonal data that is anomalies relative to that year.
## For monthly data set this to False and look at season=5
do_deannualise = False

data_dir = '/thredds/ipsl/mmenary/fgco2/{:s}'.format(model)
var_names = ['dpco2', 'tos', 'mlotst', 'intpp']

remake_saves = True
do_scaling = True

block_len = 30
# block_len = 50  # Takes <6hrs except for block 19!
# block_len = 29  # About 1.1Gb
# block_len = 4

def get_ji_for_block(block_number, block_len):
    nblocks_j = np.int(np.ceil(nj / np.float(block_len)))
    nblocks_i = np.int(np.ceil(ni / np.float(block_len)))

    block_j = np.int(block_number // nblocks_i)
    block_i = np.int(block_number - (block_j * nblocks_i))

    j0 = block_j * block_len
    i0 = block_i * block_len
    j1 = np.min((((block_j + 1) * block_len), nj))
    i1 = np.min((((block_i + 1) * block_len), ni))

    return [j0, j1, i0, i1]

def make_annual_mean(in_arr, season):
    nt_in = len(in_arr)
    nt_out = nt_in // 12
    out_arr = np.ma.masked_all(shape=nt_out)
    if season == 4:
        ## Uses December of previous year (but keep array the same size)
        nt_out -= 1

    if season == 0:
        for tt in range(nt_out):
            out_arr[tt] = in_arr[(tt * 12):(tt + 1) * 12].mean()
    elif season == 4:
        for tt in range(nt_out):
            out_arr[tt] = in_arr[(tt * 12) + 11:(tt * 12) + 11 + 3].mean()
    else:
        season_offset = season * 3 - 1
        for tt in range(nt_out):
            out_arr[tt] = in_arr[(tt * 12) + season_offset:(tt * 12) + season_offset + 3].mean()

    return out_arr

def deannualise(in_arr, ann_arr, limit_length=False, input1_is_monthly=False):
    nt_in = len(in_arr)
    if limit_length or input1_is_monthly:
        nt_in = nt_in // 12
    out_arr = np.ma.masked_all(shape=(nt_in))

    if input1_is_monthly:
        for tt in range(nt_in):
            out_arr[tt] = in_arr[tt] - ann_arr[tt // 12]
    else:
        for tt in range(nt_in):
            out_arr[tt] = in_arr[tt] - ann_arr[tt]

    return out_arr

def smooth_and_shorten(in_arr, smoothing):
    nt = len(in_arr)
    nt_out = nt // smoothing
    out_arr = np.ma.masked_all(shape=nt_out)

    for tt in range(nt_out):
        out_arr[tt] = in_arr[tt * smoothing:(tt + 1) * smoothing].mean()

    return out_arr

def detrend(in_arr, detrending):
    if in_arr.count() < 10:
        out_arr = np.ma.masked_all(shape=in_arr.shape)
    else:
        smoothed = mfilter.smooth1d(in_arr, detrending, mask_ends=True)
        out_arr = in_arr - smoothed
    return out_arr

output_dir = '/data/mmenary/fgco2_blocks/{:s}'.format(model)
block_number_filled = '{:d}'.format(block_number).zfill(6)

smoothing_string = ''
if smoothing > 1:
    smoothing_string = '_Smo{:d}'.format(smoothing)

detrending_string = ''
if detrending > 1:
    detrending_string = '_det{:d}'.format(detrending)

deannualised_string = ''
if do_deannualise:
    deannualised_string = '_AnnAnoms'

output_file = 'dpco2_MLR_Block{:s}-{:d}{:s}{:s}{:s}.pkl'.format(block_number_filled, block_len, deannualised_string, smoothing_string, detrending_string)
out_save_file = os.path.join(output_dir, output_file)
# out_save_file += '.TEST'  ########################################################################################
print("Will save to: {:s}".format(out_save_file))

if not remake_saves:
    if os.path.isfile(out_save_file):
        raise ValueError("Already created this file")

if model == "IPSLCM6A":
    nj = 332
    ni = 362
elif model == "UKESM1-0-LL":
    nj = 330
    ni = 360
elif model == "EC-Earth3-CC":
    nj = 292
    ni = 362
elif (model == "CESM2") or (model == "CESM2-FV2") or (model == "CESM2-WACCM") or (model == "CESM2-WACCM-FV2"):
    nj = 384
    ni = 320
elif model == "NorESM2-LM":
    nj = 385
    ni = 360
elif model == "CNRM-ESM2-1":
    nj = 294
    ni = 362
elif model == "CMCC-ESM2":
    nj = 292
    ni = 362
elif (model == "GISS-E2-1-G") or (model == "GISS-E2-1-G-CC"):
    nj = 180  ## Regridded
    ni = 360  ## Regridded

j0, j1, i0, i1 = get_ji_for_block(block_number, block_len)

regr = linear_model.LinearRegression(fit_intercept=True)

## Read the data in
data = {}
for ivar, var_name in enumerate(var_names):
    print("Reading {:s}".format(var_name))

    if (model == "GISS-E2-1-G") or (model == "GISS-E2-1-G-CC"):
        suffix = 'temp_re.nc'
    else:
        suffix = 'temp.nc'

    filepath = os.path.join(data_dir, '{:s}_{:s}'.format(var_name, suffix))
    loaded = netCDF4.Dataset(filepath)
    data[var_name] = loaded.variables[var_name][:, j0:j1, i0:i1]

    # Make the arrays
nseasons = 6  # 4 + 1 annual + 1 seasonal cycle
ncoefs = 4  ## 3 scalings and 1 intercept
npreds = 4  ## 1 full prediction and 3 with different variables held constant
coefficients_map = np.ma.masked_all(shape=(nseasons, nj, ni, ncoefs))

## Can't save this as it would be huge
# predictions_map = np.ma.masked_all(shape=(TIME, nj, ni, npreds))

correlations_map = np.ma.masked_all(shape=(nseasons, nj, ni, npreds))

# Loop and do MLR
for season in range(nseasons):
    print(season)
    # if season != 5: continue ######################################################################################
    for jj in range(j0, j1):
        print("Calculating coefficients for j={:d} ({:d}/{:d})".format(jj, j0, j1))
        j_sub = jj - j0
        for ii in range(i0, i1):
            i_sub = ii - i0
            tos = data['tos'][:, j_sub, i_sub]
            mlotst = data['mlotst'][:, j_sub, i_sub]
            intpp = data['intpp'][:, j_sub, i_sub]
            dpco2 = data['dpco2'][:, j_sub, i_sub]

            if np.ma.is_masked(tos[0]):
                continue

            if do_deannualise or (season == 5):
                tos_ann = make_annual_mean(tos, 0)
                mlotst_ann = make_annual_mean(mlotst, 0)
                intpp_ann = make_annual_mean(intpp, 0)
                dpco2_ann = make_annual_mean(dpco2, 0)

            if season != 5:
                tos = make_annual_mean(tos, season)
                mlotst = make_annual_mean(mlotst, season)
                intpp = make_annual_mean(intpp, season)
                dpco2 = make_annual_mean(dpco2, season)

                if do_deannualise:
                    tos = deannualise(tos, tos_ann)
                    mlotst = deannualise(mlotst, mlotst_ann)
                    intpp = deannualise(intpp, intpp_ann)
                    dpco2 = deannualise(dpco2, dpco2_ann)
            elif season == 5:
                tos = deannualise(tos, tos_ann, limit_length=True, input1_is_monthly=True)
                mlotst = deannualise(mlotst, mlotst_ann, limit_length=True, input1_is_monthly=True)
                intpp = deannualise(intpp, intpp_ann, limit_length=True, input1_is_monthly=True)
                dpco2 = deannualise(dpco2, dpco2_ann, limit_length=True, input1_is_monthly=True)

            if (detrending > 1) and (season != 5):
                tos = detrend(tos, detrending)
                mlotst = detrend(mlotst, detrending)
                intpp = detrend(intpp, detrending)
                dpco2 = detrend(dpco2, detrending)

            if smoothing > 1:
                real = np.nonzero(tos * mlotst * intpp * dpco2)
                if len(real[0]) < 10:
                    continue
                tos = tos[real]
                mlotst = mlotst[real]
                intpp = intpp[real]
                dpco2 = dpco2[real]
                tos = smooth_and_shorten(tos, smoothing)
                mlotst = smooth_and_shorten(mlotst, smoothing)
                intpp = smooth_and_shorten(intpp, smoothing)
                dpco2 = smooth_and_shorten(dpco2, smoothing)

            if do_scaling:
                tos -= tos.mean()
                mlotst -= mlotst.mean()
                intpp -= intpp.mean()
                tos /= tos.std()
                mlotst /= mlotst.std()
                intpp /= intpp.std()

            nt = len(tos)

            X = np.transpose(np.array((tos, mlotst, intpp)))
            Y = dpco2
            regr.fit(X, Y)

            ## Full
            Y2full = regr.predict(X)

            ## With SST as mean
            tos_mn = np.repeat(tos.mean(), nt)
            Xtos = np.transpose(np.array((tos_mn, mlotst, intpp)))
            Y2tos = regr.predict(Xtos)

            ## With mlotst as mean
            mlotst_mn = np.repeat(mlotst.mean(), nt)
            Xmlotst = np.transpose(np.array((tos, mlotst_mn, intpp)))
            Y2mlotst = regr.predict(Xmlotst)

            ## With intpp as mean
            intpp_mn = np.repeat(mlotst.mean(), nt)
            Xintpp = np.transpose(np.array((tos, mlotst, intpp_mn)))
            Y2intpp = regr.predict(Xintpp)

            ## Put back in to maps
            for index in range(3):
                coefficients_map[season, jj, ii, index] = regr.coef_[index]
            coefficients_map[season, jj, ii, 3] = regr.intercept_
            correlations_map[season, jj, ii, 0] = np.corrcoef(Y2full, Y)[0][1]
            correlations_map[season, jj, ii, 1] = np.corrcoef(Y2tos, Y)[0][1]
            correlations_map[season, jj, ii, 2] = np.corrcoef(Y2mlotst, Y)[0][1]
            correlations_map[season, jj, ii, 3] = np.corrcoef(Y2intpp, Y)[0][1]

with open(out_save_file, 'wb') as handle:
    pickle.dump([coefficients_map.data, coefficients_map.mask, correlations_map.data, correlations_map.mask],
                handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done!")
