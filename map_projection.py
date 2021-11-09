#!/bin/python3

import datetime
import sunpy.map
import astropy.units as u
import matplotlib.pyplot as plt
import nustar_pysolar # for solar to RA/Dec coordinate transform

from reproject import reproject_interp
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from sunpy.coordinates import Helioprojective, RotatedSunFrame, transform_with_sun_center
from sunpy.net import Fido, attrs as a
from aiapy.calibrate import register, update_pointing, normalize_exposure
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import warnings
warnings.filterwarnings("ignore")


# Define constants
X_LABEL = 'X (arcseconds)'
Y_LABEL = 'Y (arcseconds)'
AIA_WAVELENGTH = 94 # Units of angstroms
STEREO_MIN_WAVELENGTH = 171  # Units of angstroms
STEREO_MAX_WAVELENGTH = 195 # Units of angstroms

# Define the universal axes limits to be used by all plots.
# In units of arcseconds.
X_MIN = -1100
Y_MIN = -1100
X_MAX = 1100
Y_MAX = 1100


def add_minor_ticks(ax):
    """
    Adds minor ticks to the plot on the provided axes.
    """

    (ax.coords[0]).display_minor_ticks(True)
    (ax.coords[0]).set_minor_frequency(5)
    (ax.coords[1]).display_minor_ticks(True)
    (ax.coords[1]).set_minor_frequency(5)
    ax.tick_params(which='minor', length=1.5)
    

def most_recent_map(_map):

    current = datetime.datetime.now()
    current_date = current.strftime("%Y-%m-%dT%H:%M:%S")

    if _map.lower()=="aia":
        info = (a.Instrument("aia") & a.Wavelength(AIA_WAVELENGTH*u.angstrom))
        look_back = {"minutes":30}
    elif _map.lower()=="stereo":
        info = (a.Source('STEREO_A') & a.Instrument("EUVI") & \
            a.Wavelength(STEREO_MIN_WAVELENGTH*u.angstrom, STEREO_MAX_WAVELENGTH*u.angstrom)) 
        look_back = {"days":5}
    else:
        print("Don't know what to do! Please set _map=\"aia\" or \"stereo\".")
        
    past = current-datetime.timedelta(**look_back)
    past_date = past.strftime("%Y-%m-%dT%H:%M:%S")
    startt = str(past_date)
    endt = str(current_date)

    result = Fido.search(a.Time(startt, endt), info)
    file_download = Fido.fetch(result[0, -1], site='ROB')

    data_map = sunpy.map.Map(file_download[-1])

    # Set the axes limits.
    bl = SkyCoord(X_MIN*u.arcsec, Y_MIN*u.arcsec, frame=data_map.coordinate_frame)
    tr = SkyCoord(X_MAX*u.arcsec, Y_MAX*u.arcsec, frame=data_map.coordinate_frame)
    data_map = data_map.submap(bottom_left=bl, top_right=tr)
    
    return(data_map)


def project_map(in_map, future_time):
    """
    Create a projection of an input map at the given input time.

    Parameters
    ----------
    in_map : Sunpy map
        The input map to be projected.
    future_time : str
        The time of the projected map. 
    """

    in_time = in_map.date
    out_frame = Helioprojective(observer='earth', obstime=future_time,
                                rsun=in_map.coordinate_frame.rsun)
    rot_frame = RotatedSunFrame(base=out_frame, rotated_time=in_time)

    out_shape = in_map.data.shape
    out_center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=out_frame)
    header = sunpy.map.make_fitswcs_header(out_shape,
                                           out_center,
                                           scale=u.Quantity(in_map.scale))
    
    out_wcs = WCS(header)
    out_wcs.coordinate_frame = rot_frame

    with transform_with_sun_center():
        arr, _ = reproject_interp(in_map, out_wcs, out_shape)
    
    out_warp = sunpy.map.Map(arr, out_wcs)
    out_warp.plot_settings = in_map.plot_settings

    # Set the axes limits.
    bl = SkyCoord(X_MIN*u.arcsec, Y_MIN*u.arcsec, frame=out_warp.coordinate_frame)
    tr = SkyCoord(X_MAX*u.arcsec, Y_MAX*u.arcsec, frame=out_warp.coordinate_frame)
    
    out_warp = out_warp.submap(bottom_left=bl, top_right=tr)

    return out_warp


def draw_nustar_fov(in_map, ax, center_x, center_y, layers=[-100, 0, 100], colors='red'):
    """
    Draw squares representing NuSTAR's field of view on the current map.
    
    By default, three squares are drawn: one that is equal
    to the 12x12 arcminute FOV and two with side lengths
    of +-100 arcseconds from the actual side lengths.

    Adds a text box describing the boxes.
    
    Parameters
    ----------
    in_map : Sunpy map
        The input map on which the squares will be overlaid.
    center_x : float
        The x position, in arcseconds, of the squares' center point.
    center_y : float
        The y position, in arcseconds, of the squares' center point.
    layers : list
        List of values, in arcseconds, containing the adjustments
        to the side lengths of the drawn squares. Each value results
        in a new square drawn on the map.
    colors : str or list of str
        The colors of the drawn squares. If colors is a string,
        then each square will be drawn with that color.
        Otherwise, a list can be provided to customize the color
        of each layer. The index of the color will match the index
        of the layer.
    """
    
    # Change the colors variable to a list if it's not already one.
    if not isinstance(colors, list):
        colors = [colors]*len(layers)
    
    FOV_SIDE_LENGTH = 12*60 # Units of arcseconds

    for i, diff in enumerate(layers):
        # Translate to bottom left corner of rectangle
        bottom_left = ((center_x - FOV_SIDE_LENGTH/2 - diff)*u.arcsec,
                       (center_y - FOV_SIDE_LENGTH/2 - diff)*u.arcsec)
        rect_bl = SkyCoord(*bottom_left, frame=in_map.coordinate_frame)

        in_map.draw_quadrangle(bottom_left=rect_bl,
                                 width=(FOV_SIDE_LENGTH+2*diff)*u.arcsec,
                                 height=(FOV_SIDE_LENGTH+2*diff)*u.arcsec,
                                 color=colors[i])

    # Add text on plot.
    # Determine the position of the text box.
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()
    text_x = ax_xlim[1] * (center_x-X_MIN)/(X_MAX-X_MIN)
    text_y = ax_ylim[1] * (center_y-Y_MIN-600)/(Y_MAX-Y_MIN)
    
    # To make the text dynamic, we need to format
    # the text string based on the layers list.
    layers_copy = layers.copy()
    if 0 in layers_copy:
        layers_copy.remove(0)

    # Convert the list of int to list of str, add arcsecond unit symbols,
    # and add '+' to positive numbers
    list_of_strings = ['+'+str(x)+'\"' if x>0 else str(x)+'\"' for x in layers_copy]

    # Format the string by removing the residual brackets and quotes.
    layers_str = str(list_of_strings).replace('[','').replace(']','').replace('\'','')
    text_str = 'Center: ('+str(center_x)+'\",'+str(center_y)+'\")'+'\nBoxes 12\', ' + layers_str
    
    # Add the text to the plot.
    plt.text(text_x, text_y, text_str, color='red',
             horizontalalignment='center', verticalalignment='center')


def reprojection(obstime:str, center_x, center_y, layers):
    
    fig = plt.figure(figsize=(12, 12))

    # For AIA.
    aiamap = most_recent_map(_map="aia")
    projected_aiamap = project_map(aiamap, obstime)
    print("Got AIA map.")

    reversed_aia_cmap = (aiamap.cmap).reversed()

    ax1 = fig.add_subplot(2, 2, 1, projection=aiamap)
    aiamap.plot(cmap=reversed_aia_cmap, title=f"Original AIA Map\nAIA " + \
        str(AIA_WAVELENGTH) + f" {aiamap.date}")
    aiamap.draw_limb(color='black')
    ax1.tick_params(which='major', direction='in')
    ax1.grid(False)
    ax1.set_xlabel(X_LABEL)
    ax1.set_ylabel(Y_LABEL)
    add_minor_ticks(ax1)
    plt.colorbar(fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 2, 2, projection=projected_aiamap)
    projected_aiamap.plot(cmap=reversed_aia_cmap, title="Reprojected to an Earth Observer\nAIA " + \
        str(AIA_WAVELENGTH) + f" {projected_aiamap.date}")
    projected_aiamap.draw_limb(color='black')
    ax2.tick_params(which='major', direction='in')
    ax2.grid(False)
    ax2.set_xlabel(X_LABEL)
    ax2.set_ylabel(Y_LABEL)
    add_minor_ticks(ax2)
    plt.colorbar(fraction=0.046, pad=0.04)

    draw_nustar_fov(projected_aiamap, ax2, center_x, center_y, layers)


    # For STEREO.
    stereomap = most_recent_map(_map="stereo")
    projected_stereomap = project_map(stereomap, obstime)
    print("Got STEREO-A map.")

    ax3 = fig.add_subplot(2, 2, 3, projection=stereomap)
    stereomap.plot(title=f"Original STEREO Map\nSTEREO " + \
        str(STEREO_MIN_WAVELENGTH) + "-" + str(STEREO_MAX_WAVELENGTH) + f" {stereomap.date}")
    stereomap.draw_limb(color='black')
    ax3.tick_params(which='major', direction='in')
    ax3.grid(False)
    ax3.set_xlabel(X_LABEL)
    ax3.set_ylabel(Y_LABEL)
    add_minor_ticks(ax3)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.set_cmap('YlGn')

    ax4 = fig.add_subplot(2, 2, 4, projection=projected_stereomap)
    projected_stereomap.plot(title="Reprojected to an Earth Observer\nSTEREO " + \
        str(STEREO_MIN_WAVELENGTH) + "-" + str(STEREO_MAX_WAVELENGTH) + f" {projected_stereomap.date}")
    projected_stereomap.draw_limb(color='black')
    ax4.tick_params(which='major', direction='in')
    ax4.grid(False)
    ax4.set_xlabel(X_LABEL)
    ax4.set_ylabel(Y_LABEL)
    add_minor_ticks(ax4)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.set_cmap('YlGn')

    draw_nustar_fov(projected_stereomap, ax4, center_x, center_y, layers)

    plt.savefig('aia_stereo_projection.jpg')


if __name__ == '__main__':

    future_time = (datetime.datetime.now()+datetime.timedelta(days=4)).strftime("%Y-%m-%dT%H:%M:%S")
    reprojection(future_time, -300, -300, [-100, 0, 100])