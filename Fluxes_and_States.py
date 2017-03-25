#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:56:37 2017

@author: rmolina
"""

import datetime
import calendar
from timeit import default_timer as timer
#import scipy.io as sio
from getconstants import getconstants
from Fluxes_and_States_Masterscript import (data_path, getW, getwind,
                                            getFa, getEP, getrefined,
                                            get_stablefluxes, getFa_Vert)
import numpy as np

from fluxes_and_states_code_refactoring import (get_water,
                                                get_horizontal_fluxes,
                                                correct_horizontal_fluxes,
                                                get_evaporation_precipitation)

#BEGIN OF INPUT (FILL THIS IN)
years = np.arange(2010, 2011) #fill in the years

yearpart = np.arange(0, 364) # for a full (leap)year fill in np.arange(0,366)

boundary = 8
# with 8 the vertical separation is at 812.83 hPa for surface 
# pressure = 1031.25 hPa, which corresponds to k=47 (ERA-Interim)

divt = 24
# division of the timestep, 24 means a calculation timestep of
# 6/24 = 0.25 hours (numerical stability purposes)

count_time = 4
# number of indices to get data from (for six hourly data this
# means everytime one day)

# Manage the extent of your dataset (FILL THIS IN)
# Define the latitude and longitude cell numbers to consider and corresponding lakes that should be considered part of the land
latnrs = np.arange(7, 114)
lonnrs = np.arange(0, 240)
isglobal = 1 # fill in 1 for global computations (i.e. Earth round), fill in 0 for a local domain with boundaries

# the lake numbers below belong to the ERA-Interim data on 1.5 degree starting at Northern latitude 79.5 and longitude 0
lake_mask_1 = np.array([9,9,9,12,12,21,21,22,22,23,24,25,23,23,25,25,53,54,61,23,24,23,24,25,27,22,23,24,25,26,27,28,22,25,26,27,28,23,23,12,18])
lake_mask_2 = np.array([120+19,120+40,120+41,120+43,120+44,120+61,120+62,120+62,120+63,120+62,120+62,120+62,120+65,120+66,120+65,120+66,142-120,142-120,143-120,152-120,152-120,153-120,153-120,153-120,153-120,154-120,154-120,154-120,154-120,154-120,154-120,154-120,155-120,155-120,155-120,155-120,155-120,159-120,160-120,144-120,120+55])
lake_mask = np.transpose(np.vstack((lake_mask_1,lake_mask_2))) #recreate the arrays of the matlab model

#END OF INPUT

# Datapaths (FILL THIS IN)
invariant_data = 'input/lsm.nc' #invariants
interdata_folder = 'interdata'
input_folder = 'input'





start1 = timer()

# obtain the constants
latitude, longitude, lsm, g, density_water, timestep, A_gridcell, \
    L_N_gridcell, L_S_gridcell, L_EW_gridcell, gridcell = \
    getconstants(latnrs, lonnrs, lake_mask, invariant_data)

# loop through the years
for yearnumber in years:

    ly = int(calendar.isleap(yearnumber))
    final_time = 364 + ly  # number of parts-1 to divide a year in
    
    for a in yearpart:  # a > 365 (366th index) and not a leapyear
        start = timer()

        datapath = data_path(yearnumber, a, input_folder, interdata_folder)

        if a > final_time:
            pass
            # do nothing
        else:
            begin_time = a*4 # first index to get data from (netcdf is zero based) (leave at a*16)
            
            #1 integrate specific humidity to get the (total) column water (vapor)
            cw, W_top, W_down = \
                getW(latnrs, lonnrs, final_time, a, yearnumber, begin_time,
                     count_time, density_water, latitude, longitude, g,
                     A_gridcell, boundary,datapath)

            day = datetime.datetime(yearnumber, 1, 1) + datetime.timedelta(days=a)

            cw_new, W_top_new, W_down_new = \
                get_water(day, latnrs, lonnrs, A_gridcell, boundary, input_folder)
            assert np.allclose(cw_new, cw)
            assert np.allclose(W_top_new, W_top)
            assert np.allclose(W_down_new, W_down)

            #2 wind in between pressure levels
            U, V = getwind(latnrs, lonnrs, final_time, a, yearnumber,
                           begin_time, count_time, datapath)
            
            #3 calculate horizontal moisture fluxes
            Fa_E_top, Fa_N_top, Fa_E_down, Fa_N_down = \
                getFa(latnrs, lonnrs, boundary, cw, U, V, count_time,
                      begin_time, yearnumber, a, final_time, datapath,
                      latitude, longitude)

            (ewf, nwf, eastward_tcw, northward_tcw) = \
                 get_horizontal_fluxes(cw, day, latnrs, lonnrs,
                                       input_folder)
            east_top, north_top, east_bottom, north_bottom = \
                correct_horizontal_fluxes(ewf, nwf, eastward_tcw,
                                          northward_tcw, boundary)
            assert np.allclose(east_top, Fa_E_top)
            assert np.allclose(north_top, Fa_N_top)
            assert np.allclose(east_bottom, Fa_E_down)
            assert np.allclose(north_bottom, Fa_N_down)

            #4 evaporation and precipitation
            E, P = getEP(latnrs, lonnrs, yearnumber, begin_time, count_time,
                         latitude, longitude, A_gridcell, datapath)

            evaporation, precipitation = \
                get_evaporation_precipitation(day, latnrs, lonnrs,
                                              A_gridcell, input_folder)
            assert np.allclose(evaporation, E)
            assert np.allclose(precipitation, P)
            
            #5 put data on a smaller time step
            Fa_E_top_1, Fa_N_top_1, Fa_E_down_1, Fa_N_down_1, E, P, W_top, \
                W_down = getrefined(Fa_E_top, Fa_N_top, Fa_E_down, Fa_N_down,
                                    W_top, W_down, E, P, divt, count_time,
                                    latitude, longitude)

            #6 stabilize horizontal fluxes and get everything in (m3 per smaller timestep)
            Fa_E_top, Fa_E_down, Fa_N_top, Fa_N_down = \
                get_stablefluxes(W_top, W_down, Fa_E_top_1, Fa_E_down_1,
                                 Fa_N_top_1, Fa_N_down_1, timestep, divt,
                                 L_EW_gridcell,density_water, L_N_gridcell,
                                 L_S_gridcell, latitude, longitude, count_time)
            
            #7 determine the vertical moisture flux
            Fa_Vert_raw, Fa_Vert = getFa_Vert(Fa_E_top, Fa_E_down, Fa_N_top,
                                              Fa_N_down, E,P,W_top, W_down,
                                              divt, count_time, latitude,
                                              longitude, isglobal)
            
#            sio.savemat(datapath[23], {'Fa_E_top':Fa_E_top, 'Fa_N_top':Fa_N_top, 'Fa_E_down':Fa_E_down,'Fa_N_down':Fa_N_down, 'Fa_Vert':Fa_Vert, 'E':E, 'P':P, 
#                                                                                    'W_top':W_top, 'W_down':W_down}, do_compression=True)
            
            # alternative, but slower and more spacious
            # np.savez_compressed(datapath[23],Fa_E_top,Fa_N_top,Fa_E_down,Fa_N_down,Fa_Vert,E,P,W_top,W_down)
            
        end = timer()
        print 'Runtime fluxes_and_storages for day ' + str(a+1) + ' in year ' + str(yearnumber) + ' is',(end - start),' seconds.'
end1 = timer()
print 'The total runtime is',(end1-start1),' seconds.'
