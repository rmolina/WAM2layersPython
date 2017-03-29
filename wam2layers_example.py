#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:54:46 2017

@author: rmolina
"""

import datetime
import calendar
from timeit import default_timer as timer
import scipy.io
import numpy as np
import wam2layers


def main(years, boundary=8, divt=24, timestep=6*3600):
    """ Reimplements the Fluxes_and_States_Masterscript.py functionality from
    https://github.com/ruudvdent/WAM2layersPython """

    print "[*] Checking data files"
    for year in years:
        print "    [+] Checking whether we have data for %s" % year
        wam2layers.download_era_interim_data(year)
    print "    [+] Checking whether we have data for %s-01-01" % (year+1)
    wam2layers.download_era_interim_data(years[-1] + 1, just_one_day=True)

    main_start = timer()
    gridcell_geometry = wam2layers.get_gridcell_geometry()

    for year in years:

        print "[*] Running fluxes_and_storages() for %s" % year

        days = np.arange(366 if calendar.isleap(year) else 365)

        for day in days:
            day_start = timer()
            day_as_dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day)

            (east_top, north_top, east_bottom, north_bottom, vertical_flux,
             evaporation, precipitation, water_top, water_bottom) = \
             wam2layers.fluxes_and_storages(day_as_dt, gridcell_geometry, boundary, divt, timestep)

            scipy.io.savemat('interdata/%s-%sfluxes_storages.mat' % (year, day),
                             {'Fa_E_top': east_top,
                              'Fa_N_top': north_top,
                              'Fa_E_down': east_bottom,
                              'Fa_N_down': north_bottom,
                              'Fa_Vert': vertical_flux,
                              'E': evaporation,
                              'P': precipitation,
                              'W_top': water_top,
                              'W_down': water_bottom
                             }, do_compression=True)

            print '    [+] fluxes_and_storages() runtime for %s is %.2f seconds.' % \
                (day_as_dt.date(), timer() - day_start)

    print '[*] Total runtime is %.2f seconds.' % (timer() - main_start)


main(years=np.arange(2010, 2011), boundary=8)


def lsm_demo():
    """ plots a masked layer of time-averaged surface pressure """

    import netCDF4
    import cartopy.crs
    import matplotlib.pyplot as plt
    import wam2layers_config
    import numpy.ma as ma


    xgrd, ygrd, mask = wam2layers.land_sea_mask()

    with netCDF4.MFDataset("%s/*.sp.nc" % wam2layers_config.data_dir) as dataset:
        pres = np.average(dataset.variables['sp'][:], axis=0)

    projection = cartopy.crs.PlateCarree()

    # plot mask
    fig, axis = plt.subplots(subplot_kw=dict(projection=projection))
    axis.plot(xgrd[mask], ygrd[mask], 'k.', alpha=0.25)

    # plot masked data
    fig, axis = plt.subplots(subplot_kw=dict(projection=projection))
    #cs = ax.pcolormesh(longitude, latitude, ma.masked_array(pressure, ~m))
    plt.contourf(ma.masked_array(xgrd, mask),
                 ma.masked_array(ygrd, mask),
                 ma.masked_array(pres, ~mask))
    axis.coastlines(resolution='50m')
    axis.set_extent([-90, -30, -40, +20])


#lsm_demo()
