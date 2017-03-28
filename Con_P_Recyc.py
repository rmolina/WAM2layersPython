#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 21:29:59 2017

@author: rmolina
"""

import calendar
from getconstants import getconstants
from timeit import default_timer as timer
import numpy as np
import scipy.io as sio
from Con_P_Recyc_Masterscript import data_path_ea, create_empty_array, data_path, get_Sa_track_forward, get_Sa_track_forward_TIME


#%% BEGIN OF INPUT1 (FILL THIS IN)
years = np.arange(2010,2011) #fill in the years
yearpart = np.arange(0,364) # for a full (leap)year fill in np.arange(0,366)
boundary = 8 # with 8 the vertical separation is at 812.83 hPa for surface pressure = 1031.25 hPa, which corresponds to k=47 (ERA-Interim)
divt = 24 # division of the timestep, 24 means a calculation timestep of 6/24 = 0.25 hours (numerical stability purposes)
count_time = 4 # number of indices to get data from (for six hourly data this means everytime one day)

# Manage the extent of your dataset (FILL THIS IN)
# Define the latitude and longitude cell numbers to consider and corresponding lakes that should be considered part of the land
latnrs = np.arange(7,114)
lonnrs = np.arange(0,240)

# the lake numbers below belong to the ERA-Interim data on 1.5 degree starting at Northern latitude 79.5 and longitude -180
lake_mask_1 = np.array([9,9,9,12,12,21,21,22,22,23,24,25,23,23,25,25,53,54,61,23,24,23,24,25,27,22,23,24,25,26,27,28,22,25,26,27,28,23,23,12,18])
lake_mask_2 = np.array([120+19,120+40,120+41,120+43,120+44,120+61,120+62,120+62,120+63,120+62,120+62,120+62,120+65,120+66,120+65,120+66,142-120,142-120,143-120,152-120,152-120,153-120,153-120,153-120,153-120,154-120,154-120,154-120,154-120,154-120,154-120,154-120,155-120,155-120,155-120,155-120,155-120,159-120,160-120,144-120,120+55])
lake_mask = np.transpose(np.vstack((lake_mask_1,lake_mask_2))) #recreate the arrays of the matlab model

# obtain the constants

invariant_data = 'input/lsm.nc' #invariants
interdata_folder = 'interdata'
input_folder = 'input'


latitude,longitude,lsm,g,density_water,timestep,A_gridcell,L_N_gridcell,L_S_gridcell,L_EW_gridcell,gridcell = getconstants(latnrs,lonnrs,lake_mask,invariant_data)

# BEGIN OF INPUT 2 (FILL THIS IN)
Region = lsm
Kvf = 3 # vertical dispersion factor (advection only is 0, dispersion the same size of the advective flux is 1, for stability don't make this more than 3)
timetracking = 1 # 0 for not tracking time and 1 for tracking time
veryfirstrun = 1 # type '1' if no run has been done before from which can be continued, otherwise type '0'


#END OF INPUT

#%% Runtime & Results

start1 = timer()

# The two lines below create empty arrays for first runs/initial values are zero. 
datapathea = data_path_ea(years,yearpart) #define paths for empty arrays
if veryfirstrun == 1:
    create_empty_array(count_time,divt,latitude,longitude,yearpart,years,datapathea) #creates empty arrays for first day run

# loop through the years
for yearnumber in years:
    
    if (yearpart[-1] == 365) & (calendar.isleap(yearnumber) == 0):
        thisyearpart = yearpart[:-1]
    else: # a leapyear
        thisyearpart = yearpart
        
    for a in thisyearpart:
        start = timer()

        if a == 0: # a == 1 January
            previous_data_to_load = (str(yearnumber-1) + '-' + str(364+calendar.isleap(yearnumber-1)))
        else: # a != 1 January
            previous_data_to_load = (str(yearnumber) + '-' + str(a-1))
        print 'previous_data_to_load', previous_data_to_load

        datapath = data_path(previous_data_to_load,yearnumber,a)
        print 'datapath', datapath
        
        # Sa_track.mat
        loading_ST = sio.loadmat(datapath[0],verify_compressed_data_integrity=False)
        Sa_track_top = loading_ST['Sa_track_top']
        Sa_track_down = loading_ST['Sa_track_down']
        Sa_track_top_last_scheef = Sa_track_top[-1,:,:]
        Sa_track_down_last_scheef = Sa_track_down[-1,:,:]
        Sa_track_top_last =  np.reshape(Sa_track_top_last_scheef, (1,len(latitude),len(longitude)))
        Sa_track_down_last =  np.reshape(Sa_track_down_last_scheef, (1,len(latitude),len(longitude)))
        
        # fluxes_storages.mat
        loading_FS = sio.loadmat(datapath[1],verify_compressed_data_integrity=False)
        Fa_E_top = loading_FS['Fa_E_top']
        Fa_N_top = loading_FS['Fa_N_top']
        Fa_E_down = loading_FS['Fa_E_down']
        Fa_N_down = loading_FS['Fa_N_down']
        Fa_Vert = loading_FS['Fa_Vert']
        E = loading_FS['E']
        P = loading_FS['P']
        W_top = loading_FS['W_top']
        W_down = loading_FS['W_down']
        
        # call the forward tracking function
        if timetracking == 0:
            (Sa_track_top, Sa_track_down, north_loss, south_loss,
             down_to_top, top_to_down, water_lost) = \
                 get_Sa_track_forward(latitude,longitude,count_time,divt,Kvf,
                                      Region,Fa_E_top,Fa_N_top,Fa_E_down,
                                      Fa_N_down,Fa_Vert,E,P,W_top,W_down,
                                      Sa_track_top_last,Sa_track_down_last)   
        elif timetracking == 1:
            loading_STT = sio.loadmat(datapath[2],verify_compressed_data_integrity=False)
            Sa_time_top = loading_STT['Sa_time_top'] # [seconds]
            Sa_time_down = loading_STT['Sa_time_down']
            Sa_time_top_last_scheef = Sa_time_top[-1,:,:]
            Sa_time_down_last_scheef = Sa_time_down[-1,:,:]
            Sa_time_top_last =  np.reshape(Sa_time_top_last_scheef, (1,len(latitude),len(longitude)))
            Sa_time_down_last =  np.reshape(Sa_time_down_last_scheef, (1,len(latitude),len(longitude)))
            Sa_time_top,Sa_time_down,Sa_track_top,Sa_track_down,north_loss,south_loss,down_to_top,top_to_down,water_lost = get_Sa_track_forward_TIME(latitude,longitude,count_time,divt,timestep,Kvf,Region,Fa_E_top,Fa_N_top,Fa_E_down,Fa_N_down,Fa_Vert,E,P,
                                       W_top,W_down,Sa_track_top_last,Sa_track_down_last,Sa_time_top_last,Sa_time_down_last)
            
        # save this data 
        sio.savemat(datapath[3], {'Sa_track_top':Sa_track_top,'Sa_track_down':Sa_track_down,'north_loss':north_loss, 'south_loss':south_loss,'down_to_top':down_to_top,'top_to_down':top_to_down,'water_lost':water_lost}, do_compression=True)
        if timetracking == 1:
            sio.savemat(datapath[4], {'Sa_time_top':Sa_time_top,'Sa_time_down':Sa_time_down},do_compression=True)
            
        end = timer()
        print 'Runtime Sa_track for day ' + str(a+1) + ' in year ' + str(yearnumber) + ' is',(end - start),' seconds.\n'
        
end1 = timer()
print 'The total runtime of Con_P_Recyc_Masterscript is',(end1-start1),' seconds.'

