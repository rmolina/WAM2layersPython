#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:56:58 2017

@author: rmolina
"""

# standard imports
import datetime
import os.path
import collections
# third party imports
import netCDF4
import numpy as np
from ecmwfapi import ECMWFDataServer
from scipy.constants import g
import cartopy.io
import shapely.ops
import shapely.geometry
# our own imports
import wam2layers_config


WATER_DENSITY = 1000.  # kg/m3
AUTHALIC_EARTH_RADIUS = 6371007.2  # [m]
IS_GLOBAL = wam2layers_config.region[1] - wam2layers_config.region[0] == 360


def download_era_interim_data(year, grid=1.5, just_one_day=False):
    """ downloads data from ERA Interim """

    # FIXME: we are only checking that the file exists.
    # we need to check whether the data is actually available
    # i.e,, open the dataset, check the time dmension ,an return if ok
    # or keep going and download
    # This wil also help to identify cases when we peviously downloaded
    # just_one_day but we want to re-download the files for the full year!

    # FIXME: download only the region!!!

    if just_one_day:
        date_param = "%s0101" % year
    else:
        date_param = "%s0101/to/%s1231" % (year, year)

    server = ECMWFDataServer()

    server_config = {
        'dataset' : "interim",
        'date'    : date_param,
        'stream'  : "oper",
        'time'    : "00:00:00/06:00:00/12:00:00/18:00:00",
        'step'    : "0",
        'type'    : "an",
        'grid'    : "%s/%s" % (grid, grid),
        'levtype' : "sfc",
        'class'   : "ei",
        #'area'    : "%s/%s/%s/%s" % (20,-90,-40,-30)  # (north, west, south, east)
        'format'  : "netcdf",
    }

    sfc_an = {
        'viwve': 71.162,  # Vertical integral of eastward water vapour flux (ewvf)
        'viwvn': 72.162,  # Vertical integral of northward water vapour flux (nwvf)
        'vilwe': 88.162,  # Vertical integral of eastward cloud liquid water flux (eclwf)
        'vilwn': 89.162,  # Vertical integral of northward cloud liquid water flux (nclwf)
        'viiwe': 90.162,  # Vertical integral of eastward cloud frozen water flux (ecfwf)
        'viiwn': 91.162,  # Vertical integral of northward cloud frozen water flux (ncfwf)
        'sp': 134.128,  # Surface pressure
        'tcw': 136.128,  # Total column water
        'tcwv': 137.128,  # Total column water vapour
    }

    sfc_fc = {
        'tp': 228.128,  # Total precipitation
        'e': 182.128,  # Evaporation
    }

    ml_an = {
        'u': 131.128,  # U component of wind
        'v': 132.128,  # V component of wind
        'q': 133.128,  # Specific humidity
    }

    for key, value in sfc_an.items():
        filename = "%s/%s.%s.nc" % (wam2layers_config.data_dir, year, key)
        if not os.path.isfile(filename):
            server_config.update({'param': value,
                                  'target': filename})
            server.retrieve(server_config)

    for key, value in sfc_fc.items():
        filename = "%s/%s.%s.nc" % (wam2layers_config.data_dir, year, key)
        if not os.path.isfile(filename):
            server_config.update({'param': value,
                                  'target': filename,
                                  'type': "fc",
                                  'time': "00:00:00/12:00:00",
                                  'step': "3/6/9/12"})
            server.retrieve(server_config)

    for key, value in ml_an.items():
        filename = "%s/%s.%s.nc" % (wam2layers_config.data_dir, year, key)
        if not os.path.isfile(filename):
            server_config.update(
                {'param': "%s" % value, 'target': filename,
                 'type': "an", 'levtype': "ml", 'step': "0",
                 'time' : "00:00:00/06:00:00/12:00:00/18:00:00",
                 'levelist': '/'.join(str(level) for level in wam2layers_config.levels)
                })
            server.retrieve(server_config)


def read_netcdf(var, current_date):
    """ reads ERA Interim data from a netCDF file """

    # TIP: some variables are stored in the netCDF file using common
    # short names (as: u, v, q), but other variables are stored as
    # "p" + number. This is the case for the vertical integrals of fluxes.

    # this dictionary relates short names and their numeric codes
    codes = {
        'viwve' :  71.162,  # eastward water vapour flux
        'viwvn' :  72.162,  # northward water vapour flux
        'vilwe' :  88.162,  # eastward cloud liquid water flux
        'vilwn' :  89.162,  # northward cloud liquid water flux
        'viiwe' :  90.162,  # eastward cloud frozen water flux
        'viiwn' :  91.162,  # northward cloud frozen water flux
    }

    # test if we should use a short name or a code to access the variable
    if var in codes.keys():
        var_name = 'p%s' % codes[var]
    else:
        var_name = var

    # define start and end times for the time-slice of the array
    # TIP: we use a timedelta of 25 hours to catch the 00 hours of the
    # next day (when available)
    start = current_date
    end = start + datetime.timedelta(hours=25)

    # TIP; for evaporation and precipitation, the first register of each day
    # is at 03 hours instead of 00 hours
    if var in ['e', 'tp']:
        start += datetime.timedelta(hours=3)

    with netCDF4.MFDataset("%s/*.%s.nc" % (wam2layers_config.data_dir, var)) as dataset:

        time_dimension = netCDF4.num2date(dataset.variables["time"][:],
                                          dataset.variables["time"].units)

        start, end = np.searchsorted(time_dimension, [start, end])

        latnrs, lonnrs = get_latnrs_lonnrs()
        if var in ['q', 'v', 'u']:  # var has levels
            data = dataset.variables[var_name][start:end, :, latnrs, lonnrs]
        else:
            data = dataset.variables[var_name][start:end, latnrs, lonnrs]

    return data


def get_atmospheric_pressure(current_date, levels):
    """ calculates atmospheric pressure [Pa] at model levels """
    # Source for A and B vertical discretisation values:
    # http://www.ecmwf.int/en/forecasts/documentation-and-support/60-model-levels
    vertical_discretisation_a = np.array(
        [0, 20, 38.425343, 63.647804, 95.636963, 134.483307, 180.584351,
         234.779053, 298.495789, 373.971924, 464.618134, 575.651001,
         713.218079, 883.660522, 1094.834717, 1356.474609, 1680.640259,
         2082.273926, 2579.888672, 3196.421631, 3960.291504, 4906.708496,
         6018.019531, 7306.631348, 8765.053711, 10376.126953, 12077.446289,
         13775.325195, 15379.805664, 16819.474609, 18045.183594, 19027.695313,
         19755.109375, 20222.205078, 20429.863281, 20384.480469, 20097.402344,
         19584.330078, 18864.75, 17961.357422, 16899.46875, 15706.447266,
         14411.124023, 13043.21875, 11632.758789, 10209.500977, 8802.356445,
         7438.803223, 6144.314941, 4941.77832, 3850.91333, 2887.696533,
         2063.779785, 1385.912598, 855.361755, 467.333588, 210.39389,
         65.889244, 7.367743, 0, 0])

    vertical_discretisation_b = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0.000076, 0.000461, 0.001815, 0.005081, 0.011143, 0.020678,
         0.034121, 0.05169, 0.073534, 0.099675, 0.130023, 0.164384, 0.202476,
         0.243933, 0.288323, 0.335155, 0.383892, 0.433963, 0.484772, 0.53571,
         0.586168, 0.635547, 0.683269, 0.728786, 0.771597, 0.811253, 0.847375,
         0.879657, 0.907884, 0.93194, 0.951822, 0.967645, 0.979663, 0.98827,
         0.994019, 0.99763, 1])

    surface_pressure = read_netcdf('sp', current_date)

    return (vertical_discretisation_a[np.newaxis, levels, np.newaxis, np.newaxis] +
            vertical_discretisation_b[np.newaxis, levels, np.newaxis, np.newaxis] *
            surface_pressure[:, np.newaxis])


def get_water(current_date, gridcell, boundary):
    """ calculate water volumes for the two layers """

    levels = np.array([0] + wam2layers_config.levels)

    atmospheric_pressure = get_atmospheric_pressure(current_date, levels)
    specific_humidity = read_netcdf('q', current_date)

    cwv = specific_humidity * np.diff(atmospheric_pressure, axis=1) / g  # [kg/m2]

    # total column water vapor = sum cwv over the vertical [kg/m2]
    total_column_water_vapor = np.sum(cwv, axis=1)

    # Total column water
    total_column_water = read_netcdf('tcw', current_date)
    calculated_total_column_water = \
        (total_column_water / total_column_water_vapor)[:, np.newaxis] * cwv

    # water volumes
    vapor_top = np.sum(cwv[:, :boundary], axis=1)
    vapor_bottom = np.sum(cwv[:, boundary:], axis=1)
    vapor = vapor_top + vapor_bottom

    water_top_layer = (total_column_water * (vapor_top / vapor) *
                       gridcell.area / WATER_DENSITY)

    water_bottom_layer = (total_column_water * (vapor_bottom / vapor) *
                          gridcell.area / WATER_DENSITY)

    return calculated_total_column_water, water_top_layer, water_bottom_layer


def get_horizontal_fluxes(total_column_water, current_date):
    """ calculates horizontal water fluxes """

    # sum water states [kg*m-1*s-1]
    eastward_water_flux = (
        read_netcdf('viwve', current_date) +  # water vapour
        read_netcdf('vilwe', current_date) +  # liquid water
        read_netcdf('viiwe', current_date))   # frozen water

    northward_water_flux = (
        read_netcdf('viwvn', current_date) +  # water vapour
        read_netcdf('vilwn', current_date) +  # liquid water
        read_netcdf('viiwn', current_date))   # frozen water

    # eastward and northward fluxes
    u_wind_component = read_netcdf('u', current_date)
    v_wind_component = read_netcdf('v', current_date)
    eastward_tcw_flux = u_wind_component * total_column_water
    northward_tcw_flux = v_wind_component * total_column_water

    return (eastward_water_flux, northward_water_flux, eastward_tcw_flux,
            northward_tcw_flux)


def get_two_layer_fluxes(water_flux, tcw_flux, boundary):
    """ split the flux into two layers """
    # TIP: All units: [kg*m-1*s-1]

    # uncorrected bottom and top fluxes
    bottom_layer_flux = np.sum(tcw_flux[:, boundary:], axis=1)
    top_layer_flux = np.sum(tcw_flux[:, :boundary], axis=1)

    # corrected total fluxes
    corrected_total = water_flux / (bottom_layer_flux + top_layer_flux)
    corrected_total[corrected_total < 0] = 0
    corrected_total[corrected_total > 2] = 2

    # corrected bottom and top fluxes
    bottom_layer_flux = corrected_total * bottom_layer_flux
    top_layer_flux = corrected_total * top_layer_flux

    # fluxes during the timestep
    bottom_layer_flux = 0.5 * (bottom_layer_flux[:-1] + bottom_layer_flux[1:])
    top_layer_flux = 0.5 * (top_layer_flux[:-1] + top_layer_flux[1:])

    return top_layer_flux, bottom_layer_flux


def get_evaporation_precipitation(current_date, gridcell):
    """ get evaporation and precipitation data from ERA Interim netCDF files,
    disaggregate the data into 3h accumulated values and transfer invalid
    (by sign convention) evaporation into precipitation """

    evaporation = read_netcdf('e', current_date)
    precipitation = read_netcdf('tp', current_date)

    # TIP: ERA Interim's evaporation and precipitation are stored as values
    # accumulated in the ranges: 0h-3h, 0h-6h, 0h-9h, 0h-12h, and 12h-15h,
    # 12h-18h, 12h-21h, 12h-24h

    # calculate differences in time in order to obtain 3h acccumulations
    e_diff = np.diff(evaporation, axis=0)
    p_diff = np.diff(precipitation, axis=0)

    # TIP: indices 0 and 4 in evaporation an precipitation are already
    # expressed as 3-hour accumulation (0h-3h and 12h-15h) and therefore they
    # won't be replaced

    # disaggregate evaporation
    evaporation[1:4] = e_diff[0:3]  # [1:4] <==> 06h, 09h, 12h
    evaporation[5:8] = e_diff[4:7]  # [5:8] <==> 18h, 21h, 24h

    # disaggregate total precipitation into 3h accumulated values
    precipitation[1:4] = p_diff[0:3]  # [1:4] <==> 03h, 06h, 09h
    precipitation[5:8] = p_diff[4:7]  # [5:8] <==> 15h, 18h, 21h

    # TIP: ERA Interim's vertical fluxes are positive downwards. Therefore,
    # precipitation should be positive, and evaporation should be negative.

    # positive values of evaporation are transferred to precipitation and
    # negative values in the resulting precipitation are discarded
    precipitation = np.maximum(precipitation + np.maximum(evaporation, 0), 0)

    # positive evaporation (already transferred) is discarded, and the sign is
    # reversed to use positive evaporation
    evaporation = -np.minimum(evaporation, 0)

    # calculate volumes
    evaporation_volume = evaporation * gridcell.area
    precipitation_volume = precipitation * gridcell.area

    return evaporation_volume, precipitation_volume


def refine_fluxes(eastward_top, northward_top, eastward_bottom,
                  northward_bottom, divt):
    """ refine horizontal fluxes into a smaller timestep """
    # TIP: fluxes are just repeated 'divt' times
    eastward_bottom = np.repeat(eastward_bottom, repeats=divt, axis=0)
    northward_bottom = np.repeat(northward_bottom, repeats=divt, axis=0)
    eastward_top = np.repeat(eastward_top, repeats=divt, axis=0)
    northward_top = np.repeat(northward_top, repeats=divt, axis=0)

    return eastward_top, northward_top, eastward_bottom, northward_bottom


def refine_evap_precip(evaporation, precipitation, divt):
    """ refine evaporation and precipitation  into a smaller timestep """
    # TIP: E and P values are evenly distributed in 'divt2' parts
    divt2 = divt/2  # TIP: E and P data is available every 3h (not every 6h)
    evaporation = np.repeat(evaporation, repeats=divt2, axis=0) / divt2
    precipitation = np.repeat(precipitation, repeats=divt2, axis=0) / divt2

    return evaporation, precipitation


def refine_water(water, divt):
    """ refine water volumes into a smaller timestep """
    # TIP: water volumes are accumulated: the difference between two observed
    # volumes (which are 6 hours apart) is evenly distributed in 'divt' parts,
    # and a part is added at each step except the first one (at the first step
    # we are in an observed volume, there is no need to add anything)

    # TIP: Do not assume each day will have 4 observations.
    # This might not be the case for other datasets.
    # FIXME: should we include num_obs in wam2layers_config.py ?
    num_obs = water[:-1].shape[0]
    # TIP: partvector stores how many parts must be added at each step
    partvector = np.tile(np.arange(divt), num_obs)

    water_refined = (np.repeat(water[:-1], repeats=divt, axis=0) +
                     partvector[:, np.newaxis, np.newaxis] *
                     np.repeat(np.diff(water, axis=0) / divt,
                               repeats=divt, axis=0))

    water_refined = np.concatenate((water_refined,
                                    water[-1][np.newaxis]))

    return water_refined


def stable_layer_fluxes(water, eastern_flux, northern_flux, refined_timestep,
                        gridcell):
    """ get stable eastern and northen fluxes for one layer """

    # convert to m3
    # TIP: [(kg*m^-1*s^-1) * s * m * (kg^-1*m^3)] = [m3]
    eastern = (eastern_flux * refined_timestep * gridcell.side_length / WATER_DENSITY)

    northern = (northern_flux * refined_timestep * 0.5 *
                (gridcell.top_length + gridcell.bottom_length) /
                WATER_DENSITY)

    # find out where the negative fluxes are
    eastern_posneg = np.sign(eastern)
    northern_posneg = np.sign(northern)

    # make everything absolute
    eastern_abs = np.abs(eastern)
    northern_abs = np.abs(northern)

    # stabilize the outfluxes / influxes
    stab = 1./2. # during the reduced timestep the water cannot move further
    # than 1/x * the gridcell, in other words at least x * the reduced
    # timestep is needed to cross a gridcell

    # suppress 'invalid value encountered' warnings.
    # nan values are handled later with np.nan_to_num()
    with np.errstate(invalid='ignore'):
        eastern = np.minimum(eastern_abs, water[:-1] * stab *
                             (eastern_abs / (eastern_abs + northern_abs)))
        northern = np.minimum(northern_abs, water[:-1] * stab *
                              (northern_abs / (eastern_abs + northern_abs)))

    #get rid of the nan values
    eastern = np.nan_to_num(eastern)
    northern = np.nan_to_num(northern)

    #redefine
    eastern *= eastern_posneg
    northern *= northern_posneg

    return eastern, northern


def balance_horizontal_fluxes(eastern, northern, water):
    """ define the horizontal fluxes over the boundaries and calculates
    the water balance for one layer """

    # fluxes over the eastern boundary

    eastern_boundary = np.zeros_like(eastern)
    eastern_boundary[:, :, :-1] = 0.5 * (eastern[:, :, :-1] + eastern[:, :, 1:])
    if IS_GLOBAL:
        eastern_boundary[:, :, -1] = 0.5 * (eastern[:, :, -1] + eastern[:, :, 0])

    # separate directions west-east (all positive numbers)
    eastern_we = np.maximum(eastern_boundary, 0)
    eastern_ew = np.maximum(-eastern_boundary, 0)

    # fluxes over the western boundary
    western_we = np.zeros_like(eastern_we)
    western_we[:, :, 1:] = eastern_we[:, :, :-1]
    western_we[:, :, 0] = eastern_we[:, :, -1]
    western_ew = np.zeros_like(eastern_ew)
    western_ew[:, :, 1:] = eastern_ew[:, :, :-1]
    western_ew[:, :, 0] = eastern_ew[:, :, -1]

    # fluxes over the northern boundary

    northern_boundary = np.zeros_like(northern)
    northern_boundary[:, 1:] = 0.5 * (northern[:, :-1] + northern[:, 1:])

    # separate directions south-north (all positive numbers)
    northern_sn = np.maximum(northern_boundary, 0)
    northern_ns = np.maximum(-northern_boundary, 0)

    # fluxes over the southern boundary
    southern_sn = np.zeros_like(northern_sn)
    southern_sn[:, :-1] = northern_sn[:, 1:]
    southern_ns = np.zeros_like(northern_ns)
    southern_ns[:, :-1] = northern_ns[:, 1:]

    # calculate balance with moisture fluxes:
    sa_after_fa = (water[:-1] +
                   eastern_ew - eastern_we +
                   western_we - western_ew +
                   northern_ns - northern_sn +
                   southern_sn - southern_ns)

    return sa_after_fa


def get_vertical_fluxes_new(evap, precip, w_top, w_bottom,
                            sa_after_fa_bottom, sa_after_fa_top):
    """ update balances with evaporation and precipitation (at both layers),
    calculates balance residuals, and compute the resulting vertical flux """

    # update balances

    # total moisture in the column
    w_total = w_top + w_bottom

    # bottom: substract precipitation and add evaporation
    sa_after_all_bottom = (sa_after_fa_bottom - precip *
                           (w_bottom[:-1] / w_total[:-1]) + evap)

    # top: substract precipitation
    sa_after_all_top = sa_after_fa_top - precip * (w_top[:-1] / w_total[:-1])


    # check the water balance

    residual_bottom = np.zeros_like(sa_after_fa_bottom)  # residual factor [m3]
    residual_top = np.zeros_like(sa_after_fa_top)  # residual factor [m3]
    # bottom: calculate the residual
    residual_bottom[:, 1:-1] = w_bottom[1:, 1:-1] - sa_after_all_bottom[:, 1:-1]
    # top: calculate the residual
    residual_top[:, 1:-1] = w_top[1:, 1:-1] - sa_after_all_top[:, 1:-1]

    # compute the resulting vertical moisture flux
    # the vertical velocity so that the new residual_bottom/w_bottom =
    # residual_top/w_top (positive downward)

    fa_vert_raw = ((w_bottom[1:] / w_total[1:]) *
                   (residual_bottom + residual_top) - residual_bottom)

    # find out where the negative vertical flux is
    fa_vert_posneg = np.sign(fa_vert_raw)

    # make the vertical flux absolute
    fa_vert_abs = np.abs(fa_vert_raw)

    # stabilize the outfluxes / influxes
    stab = 1./4.
    # during the reduced timestep the vertical flux can maximally empty/fill
    # 1/x of the top or down storage

    fa_vert_stable = np.minimum(fa_vert_abs,
                                np.minimum(stab * w_top[1:],
                                           stab * w_bottom[1:]))

    # redefine the vertical flux
    fa_vert = fa_vert_stable * fa_vert_posneg

    return fa_vert


def fluxes_and_storages(day, gridcell, boundary,
                        divt, timestep):
    """ gets all done for a single day """

    # integrate specific humidity to get the (total) column water (vapor)
    w_total, w_top, w_bottom = get_water(day, gridcell, boundary)

    # calculate horizontal moisture fluxes
    ewf, nwf, eastward_tcw, northward_tcw = \
         get_horizontal_fluxes(w_total, day)

    east_top, east_bottom = \
        get_two_layer_fluxes(ewf, eastward_tcw, boundary)

    north_top, north_bottom = \
        get_two_layer_fluxes(nwf, northward_tcw, boundary)

    # evaporation and precipitation
    evaporation, precipitation = \
        get_evaporation_precipitation(day, gridcell)

    # put data on a smaller time step
    east_top, north_top, east_bottom, north_bottom = \
        refine_fluxes(east_top, north_top, east_bottom, north_bottom, divt)

    evaporation, precipitation = refine_evap_precip(evaporation, precipitation, divt)

    w_top = refine_water(w_top, divt)
    w_bottom = refine_water(w_bottom, divt)

    # stabilize horizontal fluxes and get everything in (m3 per smaller timestep)

    east_top, north_top = \
        stable_layer_fluxes(w_top, east_top, north_top, timestep / divt,
                            gridcell)

    east_bottom, north_bottom = \
        stable_layer_fluxes(w_bottom, east_bottom, north_bottom,
                            timestep / divt, gridcell)

    #7 determine the vertical moisture flux
    sa_after_fa_top = \
        balance_horizontal_fluxes(east_top, north_top, w_top)
    sa_after_fa_bottom = \
        balance_horizontal_fluxes(east_bottom, north_bottom, w_bottom)
    vertical_flux = \
        get_vertical_fluxes_new(evaporation, precipitation, w_top, w_bottom,
                                sa_after_fa_bottom, sa_after_fa_top)

    return (east_top, north_top, east_bottom, north_bottom, vertical_flux,
            evaporation, precipitation, w_top, w_bottom)


def get_gridcell_geometry():
    """
    Calculates the grid cell area and dimensions.

    The length expressions are derived from the haversine formula:
        hav(d/r)= hav(lat2-lat1) + cos(lat1) * cos(lat2) * hav(lon2-lon1)
    See: http://math.stackexchange.com/a/479459

    The expression for the area comes from:  http://gis.stackexchange.com/a/29743
    See also: https://badc.nerc.ac.uk/help/coordinates/cell-surf-area.html
    """
    # FIXME: these 3 lines should be a get_lat_lon() function
    latnrs, lonnrs = get_latnrs_lonnrs()
    with netCDF4.MFDataset("%s/*.sp.nc" % wam2layers_config.data_dir) as dataset:
        latitude = dataset.variables['latitude'][latnrs]

    geometry = collections.namedtuple('geometry', ['area', 'top_length',
                                                   'bottom_length', 'side_length'])

    grid_size = np.abs(latitude[0] - latitude[1])

    tops = np.radians(np.minimum(latitude + 0.5 * grid_size, +90))
    bottoms = np.radians(np.maximum(latitude - 0.5 * grid_size, -90))

    grid_size = np.radians(grid_size)

    # define the haversine and inverse haversine functions:
    hav = lambda x: np.sin(x/2)**2
    inv_hav = lambda x: 2 * np.arcsin(np.sqrt(x))

    # For the "side" length, lon2 = lon1, then: hav(lon2-lon1) = 0, and the
    # haversine formula becomes: hav(d/r) = hav(lat2-lat1)
    # i.e.: d = r * (lat2-lat1)
    side_length = AUTHALIC_EARTH_RADIUS * grid_size

    # For the top and bottom lengths, lat2 = lat1, then:
    #    hav(lat2-lat1) = 0
    #    cos(lat1)*cos(lat2) = cos(lat1)**2
    # and the haversine formula becomes: hav(d/r) = cos(lat1)**2 * hav(lon2-lon1)
    # i.e.: d = r * inv_hav(cos(lat1)**2 * hav(lon2-lon1))
    top_length = AUTHALIC_EARTH_RADIUS * inv_hav(np.cos(tops)**2 * hav(grid_size))
    bottom_length = AUTHALIC_EARTH_RADIUS * inv_hav(np.cos(bottoms)**2 * hav(grid_size))

    area = (AUTHALIC_EARTH_RADIUS ** 2 * grid_size * np.abs(np.sin(tops) - np.sin(bottoms)))

    return geometry(area[np.newaxis, :, np.newaxis],
                    top_length[np.newaxis, :, np.newaxis],
                    bottom_length[np.newaxis, :, np.newaxis],
                    side_length)


def wrap_lon360(lon):
    """ Source: https://github.com/pyoceans/python-oceans/blob/master/oceans/ocfis/ocfis.py """
    lon = np.atleast_1d(lon).copy()
    positive = lon > 0
    lon = lon % 360
    lon[np.logical_and(lon == 0, positive)] = 360
    return lon

def wrap_lon180(lon):
    """ Source: https://github.com/pyoceans/python-oceans/blob/master/oceans/ocfis/ocfis.py """
    lon = np.atleast_1d(lon).copy()
    angles = np.logical_or((lon < -180), (lon > 180))
    lon[angles] = wrap_lon360(lon[angles] + 180) - 180
    return lon

def inpolygon(polygon, points):
    """ https://ocefpaf.github.io/python4oceanographers/blog/2015/08/17/shapely_in_polygon/ """
    return np.array([shapely.geometry.Point(x, y).intersects(polygon) for x, y in points],
                    dtype=np.bool)

def land_sea_mask():
    """ builds the boolean land sea mask """
    with netCDF4.MFDataset("%s/*.sp.nc" % wam2layers_config.data_dir) as dataset:
        latitude = dataset.variables['latitude'][:]
        longitude = wrap_lon180(dataset.variables['longitude'][:])

    gridsize = latitude[0] - latitude[1]
    xgrd, ygrd = np.meshgrid(longitude, latitude)

    filename = '%s/lsm.%s.npy' % (wam2layers_config.data_dir, gridsize)

    if os.path.isfile(filename):
        mask = np.unpackbits(np.load(filename)).astype('bool')[:xgrd.size]
    else:
        shp = cartopy.io.shapereader.natural_earth(resolution='110m',
                                                   category='physical',
                                                   name='land')
        # load the shapefile to use as mask and build the polygons
        shp = cartopy.io.shapereader.Reader(shp)
        geoms = shp.geometries()
        polygon = shapely.ops.cascaded_union(list(geoms))
        mask = inpolygon(polygon, zip(xgrd.ravel(), ygrd.ravel()))
        np.save(filename, np.packbits(mask))

    return xgrd, ygrd, mask.reshape(xgrd.shape)


def get_latnrs_lonnrs():
    """ provides indices for slicing the lat and lon dimensions """
    # FIXME: we should have a single function returning lat and lon
    with netCDF4.MFDataset("%s/*.sp.nc" % wam2layers_config.data_dir) as dataset:
        latitude = dataset.variables['latitude'][:]
        longitude = dataset.variables['longitude'][:]

    weast, east, north, south = wam2layers_config.region

    west_idx, east_idx = np.searchsorted(longitude, wrap_lon360([weast, east]))
#    print "lonnrs = %d:%d" % (west_idx, east_idx)
#    print "longitude[%d:%d] = %s" % (west_idx, east_idx,
#                                     wrap_lon180(longitude[west_idx: east_idx]))

    north_idx, south_idx = latitude.size - np.searchsorted(latitude[::-1], [north, south])
#    print "latnrs = %d:%d" % (north_idx, south_idx)
#    print "latitude[%d:%d] = %s" % (north_idx, south_idx,
#                                    latitude[north_idx: south_idx])

    return range(north_idx, south_idx + 1), range(west_idx, east_idx + 1)
