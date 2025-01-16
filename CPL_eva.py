# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:15:44 2024

@author: Milon
"""

#%%
#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import scipy.odr
from scipy.signal import butter, filtfilt
import numpy as np
from scipy.interpolate import interp1d


def read_CPL_data(file):
    
    '''
    Read CPL data from file in csv format of Kitzmann et al.
    
    param file: Path to file
    type file: String
    
    return data: Data with 
    
                columns added by Kitzmann et al.:
            
                WL: Wavelength (WL)
                DC: DC component LogIn signal 
                DC_std: Error of DC component
                AC: AC component LogIn signal 
                AC_std: Error of AC component
                I_L: Left-handed circular-polarized intensity
                I_l_std: Error of I_L 
                I_R: Right-handed circular-polarized intensity
                I_R_std: Error of I_R
                glum: Dissymmetry factor
                glum_std: Error of glum
                lp_r: Amplitude of PEM-modulated linear polarization
                lp_r: Error of lp_r
                lp_theta: Phase of PEM-modulated linear polarization
                lp_theta_std: Error of lp_theta
                lp: Sign-corrected PEM-modulated linear polarization
                lp_std: Error of lp
                
                columns added additionally:
                    
                PL: Photoluminescence 
                CPL: Circularly Polarized Luminescence
                PL_CPL_std: Error of both PL and CPL (same error propagation through I_R and I_L)
                
    type data: pandas.core.frame.DataFrame
    '''
    
    #Read Data
    data = pd.read_csv(file)
    
    #Add Columns for easy access to interesting data
    data['PL'] = (data['I_L'] + data['I_R'])/2
    data['CPL'] = (data['I_L'] - data['I_R'])/2
    data['PL_CPL_std'] = 0.5*np.sqrt(data['I_L_std']**2+data['I_R_std']**2)
    
    return data

def longpassfilter(data_input, N = 1, Wn = 1/10):
    
    '''
    Apply long pass (Butterworth) filter to CPL, glum and lp in given CPL dataset
    
    param data_input: Path to file or data 
    type data_input: String or pandas.core.frame.DataFrame in output format of read_CPL_data()
    param N: Order of filter
    type N: int
    param Wn: critical/cut-off frequency
    type Wn: float
    
    return filtered_data: Filtered data with same columns as in output of read_CPL_data 
    type filtered_data: pandas.core.frame.DataFrame
    '''
    
    #Read data from file or 
    if type(data_input) == str:
        data_input = read_CPL_data(data_input)

    #Allocate all columns of data_input to filtered data in order to keep all properties where no filter has to be applied 
    filtered_data = data_input.copy()
    
    #Create Butterworth filter
    b, a = butter(N = N, Wn = Wn, btype = 'lowpass')
    filtered_data['CPL'] = filtfilt(b, a, data_input['CPL'])
    filtered_data['glum'] = filtfilt(b, a, data_input['glum']) 
    filtered_data['lp'] = filtfilt(b, a, data_input['lp']) 
    
    return filtered_data


#%% Function for plotting PL
def plot_single_var(data_input, var, ax = None, title = '', label = None, cutoff_WL = None):
    
    '''
    Plot data of desired variable on given axis
    
    param data_input: Path to file or data 
    type data_input: String or pandas.core.frame.DataFrame in output format of read_CPL_data()
    param var: Variable to be plotted: 'PL', 'CPL', 'glum', 'lp' (Linear Artefacts)
    type var: String
    param ax: Axis for data to plotted on
    type ax: matplotlib.axes._subplots.AxesSubplot
    param title: Title of plot
    type title: String
    param label: Label to be displayed in legend (if desired)
    type label: String
    param cutoff_WL: Interval to cut glum plot as it is very noisy when there is no signal
    type cutoff_WL: list
    
    return data: Data with same columns as in output of read_CPL_data 
    type data: pandas.core.frame.DataFrame
    return ax: Axis of PL data
    type ax: matplotlib.axes._subplots.AxesSubplot
    '''
    
    #Read data from file
    if type(data_input) == str:
        data_input = read_CPL_data(data_input)
    
    #If no axis is given, create one 
    if ax == None:
        fig, ax = plt.subplots()
    
    #Apply if necessary cutoff wavelengths to glum data
    if cutoff_WL == None:
        cutoff_WL = [data_input['WL'][0], data_input['WL'].iloc[-1]]
        
    data_input['glum'] = data_input['glum'].where((data_input['WL'] >= cutoff_WL[0]) & (data_input['WL'] <= cutoff_WL[1]))

    #Carry out plotting, including formatting
    #Special Treatment for labeling of y-axis and setting title
    if var == 'lp':
        ylabel = 'Linear Artefacts scaling in PMT Voltage'
        fulltitle = 'Linear Artefacts ' + ': ' + title
    elif var == 'glum':
        ylabel = r'$g_{lum}$'
        fulltitle = r'$g_{lum}$' + ': ' + title
    else:
        ylabel = var + ' in PMT Voltage'
        fulltitle = var + ': ' + title
        
    ax.set_title(fulltitle)
    ax.set_xlabel('Wavelength in nm')
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.plot(data_input['WL'], data_input[var], label = label)
    ax.legend()
    
    return data_input, ax


def plot_all(data_input, axs = None, title = '', label = None, cutoff_WL = None, figsize = (15,10)):
     
    '''
    Plot data of desired variable on given axis
    
    param data_input: Path to file or data 
    type data_input: String or pandas.core.frame.DataFrame in output format of read_CPL_data()
    param axs: 2x2 grid of matplotlib subplot axes
    type ax: 2x2 numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
    param title: Title of plot
    type title: String
    param label: Label to be displayed in legend (if desired)
    type label: String
    param cutoff_WL: Interval to cut glum plot as it is very noisy when there is no signal
    type cutoff_WL: list
    
    return axs: 2x2 grid of matplotlib subplot axes in order PL, CPL, glum, lp when array is flatten
    type axs: 2x2 numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
    '''
       
    #Read data from file
    if type(data_input) == str:
        data_input = read_CPL_data(data_input)

    #Create subplot
    if axs is None:
        figure, axs = plt.subplots(2,2, sharex = True, figsize = figsize)
        axs = axs.flatten()
        axs[0].set(xlabel = 'Wavelength in nm', ylabel = ('PL in PMT Voltage'))
        axs[1].set(xlabel = 'Wavelength in nm', ylabel = ('CPL in PMT Voltage'))
        axs[2].set(xlabel = 'Wavelength in nm', ylabel = (r'$g_{lum}$'))
        axs[3].set(xlabel = 'Wavelength in nm', ylabel = ('Linear Artefacts in PMT Voltage'))
        figure.suptitle(title)
        
    #figure.suptitle(title)
    axs[0].plot(data_input['WL'], data_input['PL'], label = label)
    #axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[1].plot(data_input['WL'], data_input['CPL'], label = label)
    #axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    if cutoff_WL == None:
        cutoff_WL = [data_input['WL'][0], data_input['WL'].iloc[-1]]
        
    axs[2].plot(data_input[(data_input['WL'] > cutoff_WL[0]) & (data_input['WL'] < cutoff_WL[1])]['WL'], data_input[(data_input['WL'] > cutoff_WL[0]) & (data_input['WL'] < cutoff_WL[1])]['glum'], label = label)
    axs[3].plot(data_input['WL'], data_input['lp'], label = label)
    #axs[3].ticklabel_format(axis='y', style='sci', scilimits=(0,0))    
    
    for ax in axs:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.legend(fontsize = 10)
        ax.axhline(y = 0, color = "black")
        
    return axs


   
def plot_PL_CPL_LA(data_input, ax_PL = None, title = '', figsize = (10,5)):
                
                     
    '''
    Plot PL, CPL and LP data in same figure on potentially given axis
    
    param data_input: Path to file or data 
    type data_input: String or pandas.core.frame.DataFrame in output format of read_CPL_data()
    param axs: 2x2 grid of matplotlib subplot axes
    type ax: 2x2 numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
    param title: Title of plot
    type title: String
    param label: Label to be displayed in legend (if desired)
    type label: String
    param cutoff_WL: Interval to cut glum plot as it is very noisy when there is no signal
    type cutoff_WL: list
    
    return ax_PL: axis of PL
    type ax_PL: matplotlib.axes._subplots.AxesSubplot
    return ax_CPL: axis of CPL and LP
    type ax_CPL: matplotlib.axes._subplots.AxesSubplot
    '''
       
    #Read data from file
    if type(data_input) == str:
        data_input = read_CPL_data(data_input)
        
    #Create subplot
    if ax_PL is None:
        figure, ax_PL = plt.subplots(1, 1, sharex = True, figsize = figsize)
        ax_PL.set(xlabel = 'Wavelength in nm', ylabel = ('PL in PMT Voltage'))
        figure.suptitle(title)
        
    #Create twin acis for CPL data and set it to red color
    ax_CPL = ax_PL.twinx()
    ax_CPL.tick_params(axis='y', labelcolor='red') 
    ax_CPL.set_ylabel('CPL/Lin. Artefacts in PMT Voltage', color = 'black')
    line_PL, = ax_PL.plot(data_input['WL'], data_input['PL'], label = 'PL', color = 'black')
    ax_CPL.plot(data_input['WL'], data_input['CPL'], color = 'red', alpha = 0.5)
    
    
    ax_CPL.set_ylim(-max(abs(np.concatenate([data_input['CPL'], data_input['lp']]))), max(abs(np.concatenate([data_input['CPL'], data_input['lp']]))))
    ax_CPL.axhline(y = 0, color = 'black')
    ax_PL.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_CPL.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_CPL.plot(data_input['WL'], data_input['lp'], color = 'blue', alpha = 0.5)
    
 
    #Add longpassfiltered CPL
    filtered_data = longpassfilter(data_input)
    line_CPL, = ax_CPL.plot(filtered_data['WL'], filtered_data['CPL'], color = 'red', label = 'CPL')
    line_LA, = ax_CPL.plot(filtered_data['WL'], filtered_data['lp'], color = 'blue', label = 'Lin. Artefacts')
    ax_PL.legend([line_PL, line_CPL, line_LA], [line_PL.get_label(), line_CPL.get_label(), line_LA.get_label()])
    
    ax_PL.grid(axis = 'both')
    return ax_PL, ax_CPL
    
def plot_PL_glum_LA_norm(data_input, ax_PL = None, title = '', figsize = (10,5)):
                
                     
    '''
    Plot normalized PL and glum data in same figure on potentially given axis
    
    param data_input: Path to file or data 
    type data_input: String or pandas.core.frame.DataFrame in output format of read_CPL_data()
    param axs: 2x2 grid of matplotlib subplot axes
    type ax: 2x2 numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
    param title: Title of plot
    type title: String
    param label: Label to be displayed in legend (if desired)
    type label: String
    param cutoff_WL: Interval to cut glum plot as it is very noisy when there is no signal
    type cutoff_WL: list
    
    return ax_PL: axis of normalized PL
    type ax_PL: matplotlib.axes._subplots.AxesSubplot
    return ax_glum: axis of glum and LP
    type ax_glum: matplotlib.axes._subplots.AxesSubplot
    '''
       
    #Read data from file
    if type(data_input) == str:
        data_input = read_CPL_data(data_input)
        
    #Normalized PL data to PL maximum
    max_PL = data_input['PL'].max()
    max_index = data_input["PL"].idxmax()
    
        
    data_input['lp'] = data_input['lp']/data_input['PL']
    data_input['PL'] = data_input['PL']/max_PL
    
    #Add longpassfiltered CPL
    filtered_data = longpassfilter(data_input)
    
    #Cut wavelengths where filtered value is greater than 1% in glum and linear artefact
    filtered_data['glum'] = filtered_data['glum'].where(abs(filtered_data['glum']) < 1e-2)
    filtered_data['lp'] = filtered_data['lp'].where(filtered_data['glum'] < 1e-2)
    data_input['glum'] = data_input['glum'].where(abs(filtered_data['glum']) < 1e-2)
    data_input['lp'] = data_input['lp'].where(filtered_data['glum'] < 1e-2)
    
    #Create subplot
    if ax_PL is None:
        figure, ax_PL = plt.subplots(1, 1, sharex = True, figsize = figsize)
        ax_PL.set(xlabel = 'Wavelength in nm', ylabel = ('PL in PMT Voltage'))
        figure.suptitle(title)
        
    #Create twin acis for glum data and set it to red color
    ax_glum = ax_PL.twinx()
    ax_glum.tick_params(axis='y', labelcolor='red') 
    ax_glum.set_ylabel(r'$g_{lum}$/Lin. Artefacts in PMT Voltage', color = 'black')
    line_PL, = ax_PL.plot(data_input['WL'], data_input['PL'], label = 'PL', color = 'black')
    
    ax_glum.plot(data_input['WL'], data_input['glum'], color = 'red', alpha = 0.5)
    
    ax_glum.axhline(y = 0, color = 'black')
    ax_PL.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_glum.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_glum.plot(data_input['WL'], data_input['lp'], color = 'blue', alpha = 0.5)
    
 
    
    ax_glum.set_ylim(-1.2*np.nanmax(abs(np.concatenate([filtered_data['glum'], filtered_data['lp']]))), 1.2*np.nanmax(abs(np.concatenate([filtered_data['glum'], filtered_data['lp']]))))
    line_glum, = ax_glum.plot(filtered_data['WL'], filtered_data['glum'], color = 'red', label = r'$g_{lum}$')
    line_LA, = ax_glum.plot(filtered_data['WL'], filtered_data['lp'], color = 'blue', label = r'$g_{lin}$')
    ax_PL.legend([line_PL, line_glum, line_LA], [line_PL.get_label(), line_glum.get_label(), line_LA.get_label()])
    
    ax_PL.grid(axis = 'both')
    return ax_PL, ax_glum


def plot_PL_CPL(data_input, ax_PL = None, title = '', figsize = (10,5)):
                
                     
    '''
    Plot PL and CPL data in same figure on potentially given axis
    
    param data_input: Path to file or data 
    type data_input: String or pandas.core.frame.DataFrame in output format of read_CPL_data()
    param axs: 2x2 grid of matplotlib subplot axes
    type ax: 2x2 numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
    param title: Title of plot
    type title: String
    param label: Label to be displayed in legend (if desired)
    type label: String
    param cutoff_WL: Interval to cut glum plot as it is very noisy when there is no signal
    type cutoff_WL: list
    
    return ax_PL: axis of PL
    type ax_PL: matplotlib.axes._subplots.AxesSubplot
    return ax_CPL: axis of CPL
    type ax_CPL: matplotlib.axes._subplots.AxesSubplot
    '''
       
    #Read data from file
    if type(data_input) == str:
        data_input = read_CPL_data(data_input)
        
    #Create subplot
    if ax_PL is None:
        figure, ax_PL = plt.subplots(1, 1, sharex = True, figsize = figsize)
        ax_PL.set(xlabel = 'Wavelength in nm', ylabel = ('PL in PMT Voltage'))
        figure.suptitle(title)
        
    #Create twin acis for CPL data and set it to red color
    ax_CPL = ax_PL.twinx()
    ax_CPL.tick_params(axis='y', labelcolor='red') 
    ax_CPL.set_ylabel('CPL in PMT Voltage', color = 'red')
    line_PL, = ax_PL.plot(data_input['WL'], data_input['PL'], label = 'PL', color = 'black')
    line_CPL, = ax_CPL.plot(data_input['WL'], data_input['CPL'], color = 'red', label = 'CPL', alpha = 0.5)
    
    
    ax_CPL.set_ylim(-max(abs(data_input['CPL'])), max(abs(data_input['CPL'])))
    ax_CPL.axhline(y = 0, color = 'red')
    ax_PL.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_CPL.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_PL.legend([line_PL, line_CPL], [line_PL.get_label(), line_CPL.get_label()])
    
    #Add longpassfiltered CPL
    filtered_data = longpassfilter(data_input)
    ax_CPL.plot(filtered_data['WL'], filtered_data['CPL'], color = 'red')
    
    return ax_PL, ax_CPL
    
def plot_CPL_LA(data_input, ax_LA = None, title = '', figsize = (10,5)):
                
                     
    '''
    Plot PL and CPL data in same figure on potentially given axis
    
    param data_input: Path to file or data 
    type data_input: String or pandas.core.frame.DataFrame in output format of read_CPL_data()
    param axs: 2x2 grid of matplotlib subplot axes
    type ax: 2x2 numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
    param title: Title of plot
    type title: String
    param label: Label to be displayed in legend (if desired)
    type label: String
    param cutoff_WL: Interval to cut glum plot as it is very noisy when there is no signal
    type cutoff_WL: list
    
    return ax_PL: axis of PL
    type ax_PL: matplotlib.axes._subplots.AxesSubplot
    return ax_CPL: axis of CPL
    type ax_CPL: matplotlib.axes._subplots.AxesSubplot
    '''
       
    #Read data from file
    if type(data_input) == str:
        data_input = read_CPL_data(data_input)
        
    #Create subplot
    if ax_LA is None:
        figure, ax_LA = plt.subplots(1, 1, sharex = True, figsize = figsize)
        ax_LA.set(xlabel = 'Wavelength in nm', ylabel = ('Lin. Artefacts in PMT Voltage'))
        figure.suptitle(title)
        
    #Create twin acis for CPL data and set it to red color
    ax_CPL = ax_LA.twinx()
    ax_CPL.tick_params(axis='y', labelcolor='red') 
    ax_CPL.set_ylabel('CPL in PMT Voltage', color = 'red')
    line_CPL, = ax_CPL.plot(data_input['WL'], data_input['CPL'], label = 'CPL', color = 'red')
    line_LA, = ax_LA.plot(data_input['WL'], data_input['lp'], color = 'black', label = 'Lin. Artefacts')
    
    
    ax_CPL.set_ylim(-max(abs(data_input['CPL'])), max(abs(data_input['CPL'])))
    ax_CPL.axhline(y = 0, color = 'red')
    ax_CPL.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_LA.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_LA.legend([line_CPL, line_LA], [line_CPL.get_label(), line_LA.get_label()])
    
    return ax_LA, ax_CPL

def plot_PL_glum(data_input, ax_PL = None, title = '', cutoff_WL = None, figsize = (10,5)):
                
                     
    '''
    Plot PL and g_lum data in same figure on potentially given axis
    
    param data_input: Path to file or data 
    type data_input: String or pandas.core.frame.DataFrame in output format of read_CPL_data()
    param axs: 2x2 grid of matplotlib subplot axes
    type ax: 2x2 numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
    param title: Title of plot
    type title: String
    param label: Label to be displayed in legend (if desired)
    type label: String
    param cutoff_WL: Interval to cut glum plot as it is very noisy when there is no signal
    type cutoff_WL: list
    
    return ax_PL: axis of PL
    type ax_PL: matplotlib.axes._subplots.AxesSubplot
    return ax_g_lum: axis of g_lum
    type ax_g_lum: matplotlib.axes._subplots.AxesSubplot
    '''
       
    #Read data from file
    if type(data_input) == str:
        data_input = read_CPL_data(data_input)
        
    #Create subplot
    if ax_PL is None:
        figure, ax_PL = plt.subplots(1, 1, sharex = True, figsize = figsize)
        ax_PL.set(xlabel = 'Wavelength in nm', ylabel = ('PL in PMT Voltage'))
        figure.suptitle(title)
        
    wavelength_glum_cut = data_input[(data_input['WL'] > cutoff_WL[0]) & (data_input['WL'] < cutoff_WL[1])]['WL']
    glum_cut = data_input[(data_input['WL'] > cutoff_WL[0]) & (data_input['WL'] < cutoff_WL[1])]['glum'] 
        
    #Create twin acis for g_lum data and set it to red color
    ax_g_lum = ax_PL.twinx()
    ax_g_lum.tick_params(axis='y', labelcolor='red') 
    ax_g_lum.set_ylabel(r'$g_{lum}$', color = 'red')
    line_PL, = ax_PL.plot(data_input['WL'], data_input['PL'], label = 'PL', color = 'black')
    line_g_lum, = ax_g_lum.plot(wavelength_glum_cut, glum_cut, color = 'red', label = r'$g_{lum}$')
    
    
    
    ax_g_lum.set_ylim(-max(abs(glum_cut)), max(abs(glum_cut)))
    ax_g_lum.axhline(y = 0, color = 'red')
    ax_PL.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_g_lum.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_PL.legend([line_PL, line_g_lum], [line_PL.get_label(), line_g_lum.get_label()])
    return ax_PL, ax_g_lum    