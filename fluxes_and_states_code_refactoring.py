#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:56:58 2017

@author: rmolina
"""


import datetime

import netCDF4
import numpy as np
from scipy.constants import g

WATER_DENSITY = 1000.  # kg/m3


def read_netcdf(var, year, day, latnrs, lonnrs, input_folder):
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
    start = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day)
    end = start + datetime.timedelta(hours=25)

    # TIP; for evaporation and precipitation, the first register of each day
    # is at 03 hours instead of 00 hours
    if var in ['e', 'tp']:
        start += datetime.timedelta(hours=3)

    with netCDF4.MFDataset("%s/*.%s.nc" % (input_folder, var)) as dataset:

        time_dimension = netCDF4.num2date(dataset.variables["time"][:],
                                          dataset.variables["time"].units)

        start, end = np.searchsorted(time_dimension, [start, end])

        if var in ['q', 'v', 'u']:  # var has levels
            data = dataset.variables[var_name][start:end, :, latnrs, lonnrs]
        else:
            data = dataset.variables[var_name][start:end, latnrs, lonnrs]

    return data


def get_atmospheric_pressure(year, day, levels, latnrs, lonnrs,
                                   input_folder):
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

    surface_pressure = read_netcdf('sp', year, day, latnrs, lonnrs,
                                   input_folder)

    return (vertical_discretisation_a[np.newaxis, levels, np.newaxis, np.newaxis] +
            vertical_discretisation_b[np.newaxis, levels, np.newaxis, np.newaxis] *
            surface_pressure[:, np.newaxis, :, :])


def get_water(latnrs, lonnrs, day, year, gridcell_area, boundary,
              input_folder):
    """ calculate water volumes for the two layers """

    k = np.array([0, 17, 27, 32, 35, 38, 41, 44, 47, 48, 51, 54, 55, 56, 57,
                  58, 59, 60])

    atmospheric_pressure = \
        get_atmospheric_pressure(year, day, k, latnrs, lonnrs, input_folder)

    specific_humidity = read_netcdf('q', year, day, latnrs, lonnrs,
                                    input_folder)

    cwv = specific_humidity * np.diff(atmospheric_pressure, axis=1) / g  # [kg/m2]

    # total column water vapor = sum cwv over the vertical [kg/m2]
    total_column_water_vapor = np.sum(cwv, axis=1)

    # Total column water
    total_column_water = read_netcdf('tcw', year, day, latnrs, lonnrs,
                                     input_folder)
    calculated_total_column_water = \
        (total_column_water / total_column_water_vapor)[:, np.newaxis, :, :] * cwv

    # water volumes
    vapor_top = np.sum(cwv[:, :boundary, :, :], axis=1)
    vapor_down = np.sum(cwv[:, boundary:, :, :], axis=1)
    vapor = vapor_top + vapor_down

    water_top_layer = (total_column_water * (vapor_top / vapor) *
                       gridcell_area / WATER_DENSITY)

    water_bottom_layer = (total_column_water * (vapor_down / vapor) *
                          gridcell_area / WATER_DENSITY)

    return calculated_total_column_water, water_top_layer, water_bottom_layer
