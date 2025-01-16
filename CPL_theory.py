# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:51:45 2024

@author: Milon

File of several functions to calculate outcome through CPL Setup in 058 including PEM and Glan-Thompson (so far) 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#%%

plt.rcParams['font.size'] = 15.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 1
#plt.close('all')

#%% 

def phase_ret(time_array, f_mod = 50e3, phase_null = np.pi/2):
    '''
    Compute phase retardation of PEM of a given time interval
    
    param time_array: Time interval of relevance 
    type time_array: (len(time_array),) numpy.ndarray
    param f_mod: Modulation frequency of PEM
    type f_mod: float
    param phase_null: Maximal phase retardance introduced by the PEM; By default pi/2 corresponding to oscillating lambda/4 plate 
    type phase_null: float 
    
    return: Modulated phase for electric field component
    return type: (len(time_array),) numpy.ndarray
    '''
    return phase_null*np.cos(2*np.pi*f_mod * time_array)

def cos(angle):
    '''
    Cosine to treat angles in degree not radians
    
    angle: Angle in radians
    type angle: float, numpy.ndarray
    
    return: cosine 
    return type: float, numpy.ndarray
    '''
    return np.round(np.cos(2*np.pi/360*angle), 15)

def sin(angle):
    '''
    Sine to treat angles in degree not radians
    
    angle: Angle in radians
    type angle: float, numpy.ndarray
    
    return: sine 
    return type: float, numpy.ndarray
    '''
    return np.round(np.sin(2*np.pi/360*angle), 15)
    

def static_PEM_matrix(stat_bf_angle, stat_bf):
    '''
    Compute Mueller matrix of a (turned-off) PEM with static birefringence and fixed fast axis 
    Results taken from (Dekkers et al., 1985)
    
    param stat_bf_angle: Angle between horizontal axis and fast axis of static birefringence (in degree!)
    type stat_bf_angle: float
    param stat_bf: Absolute phase lag due to static birefringence
    type stat_bf: float
    
    return static_PEM: Mueller matrix of static component of birefringence of PEM
    return static_PEM type: (4, 4) numpy.ndarray
    '''
    # Modulate signal of static PEM component
    static_PEM = np.array([[1,0                                                                      ,0                                                                      ,0], 
                           [0,(cos(2*stat_bf_angle))**2 + (sin(2*stat_bf_angle))**2 * np.cos(stat_bf),cos(2*stat_bf_angle) * sin(2*stat_bf_angle) * (1-np.cos(stat_bf))      ,-sin(2*stat_bf_angle) * np.sin(stat_bf)],
                           [0,cos(2*stat_bf_angle) * sin(2*stat_bf_angle) * (1-np.cos(stat_bf))      ,(sin(2*stat_bf_angle))**2 + (cos(2*stat_bf_angle))**2 * np.cos(stat_bf),cos(2*stat_bf_angle)*np.sin(stat_bf)]   ,
                           [0,sin(2*stat_bf_angle) * np.sin(stat_bf)                                 ,-cos(2*stat_bf_angle)*np.sin(stat_bf)                                  ,np.cos(stat_bf)]])
    
    return static_PEM

def dynamic_PEM_matrix(time_array, f_mod = 50e3):
    '''
    Compute Mueller matrix of an ideal PEM with dynamic birefringence and fast axis fixed along horizontal direction
    Results taken from (Dekkers et al., 1985)
    
    param time_array: Time interval of relevance 
    type time_array: (len(time_array),) numpy.ndarray
    
    return dynamic_PEM: Mueller matrix of dynamic component of birefringence of PEM
    return dynamic_PEM type: (len(time_array), 4, 4) numpy.ndarray
    '''
    #Initialize result array
    dynamic_PEM = np.zeros((len(time_array), 4, 4))
    phases = phase_ret(time_array, f_mod = f_mod)

    #Create upper identity from downer phase retardation seperately
    top_left = np.eye(2)
    cos_phases = np.cos(phases)
    sin_phases = np.sin(phases)
    bottom_right = np.stack([np.stack([cos_phases, -sin_phases], axis = 1),np.stack([sin_phases, cos_phases], axis = 1)], axis = -1)

    dynamic_PEM[:, :2, :2] = top_left
    dynamic_PEM[:, 2:, 2:] = bottom_right
    
    return dynamic_PEM


def linear_polarizer_matrix(angle):
    '''
    Compute Mueller matrix of an ideal linear polarizer
    
    param angle: Angle of linear polarizer with respect to horizontal axis (in degree!)
    type angle: float
    
    return linear_polarizer: Mueller matrix of linear polarizer
    return linear_polarizer type: (4, 4) numpy.ndarray
    '''

    linear_polarizer = 0.5* np.array([[1                  , cos(2 * angle)                                                     , sin(2 * angle)                 , 0],
                            [cos(2 * angle)               , cos(2 * angle)**2                                                  , cos(2 * angle) * sin(2 * angle), 0],
                            [sin(2 * angle)               , cos(2 * angle) * sin(2 * angle)                                    , sin(2 * angle)**2              , 0],
                            [0                            , 0                                                                  , 0                              , 0]])
    
    return np.round(linear_polarizer, 10)

def calculate_final_stokes_vector(time_array, initial_stokes_vector, stat_bf_angle, stat_bf, gt_angle = 135, f_mod = 50e3):
    '''
    Compute Stokes vector, given a certain initial polarization, of setup including static and dynamic PEM component and linear (Glan-Thompson Polarizer) 
    
    param time_array: Time interval of relevance 
    type time_array: (len(time_array),) numpy.ndarray
    param initial_stokes_vector: Initial stokes vector containing intensity and polarization information 
    type initial_stokes_vector: (4,) numpy.ndarray
    param stat_bf_angle: Angle of birefringent axis
    type stat_bf_angle: float
    param stat_bf: Absolute maximal phase lag due to static birefringence
    type stat_bf: float
    param gt_angle: Angle of Glan-Thompson polarizer
    type gt_angle: float
    
    return: 
    I_detect: final stokes vector
    type I_detect: (len(time_array), 4) numpy.ndarray
    I_detect_LP: Modulated stokes vector of linear component after static PEM matrix
    type I_detect_LP: (len(time_array), 4) numpy.ndarray
    I_detect_CPL: Modulated stokes vector of CPL component after static PEM matrix
    type I_detect_CPL: (len(time_array), 4) numpy.ndarray
    '''
    
    # Modulate signal first with static PEM component
    static_PEM = static_PEM_matrix(stat_bf_angle, stat_bf)
    
    I_static_mod = static_PEM @ initial_stokes_vector
    
    # Stokes vector after static PEM can now include fake CPL!
    # Check fourth component of both vectors
    # Problem with Minkowski metrics, if separating LP and CPL component, after multiplying with matrices results won't be the same!
    I_static_mod_CPL = np.array([abs(I_static_mod[3]), 0, 0, I_static_mod[3]])
    I_static_mod_LP = np.array([np.sqrt(I_static_mod[1]**2+I_static_mod[2]**2), I_static_mod[1], I_static_mod[2], 0])
    
    # Create Mueller matrix of dynamic (ideal) PEM 
    
    #Get dynamic PEM matrix
    dynamic_PEM = dynamic_PEM_matrix(time_array, f_mod = f_mod)
    
    #Define Mueller matrix of Glan-Thompson polarizer
    gt_polarizer = linear_polarizer_matrix(gt_angle)
    
    #Compute entire output stokes vector and separated results for linear and CPL component
    I_detect = gt_polarizer @ dynamic_PEM @ I_static_mod 
    I_detect_CPL = gt_polarizer @ dynamic_PEM @ I_static_mod_CPL
    I_detect_LP = gt_polarizer @ dynamic_PEM @ I_static_mod_LP
    
    
    return I_detect, I_detect_LP, I_detect_CPL

def calculate_final_stokes_vector_commute(time_array, initial_stokes_vector, stat_bf_angle, stat_bf, gt_angle = 135):
    '''
    Compute Stokes vector, given a certain initial polarization, of setup including static and dynamic PEM component and linear (Glan-Thompson Polarizer) 
    In comparison too calculate_final_stokes_vector(), though, is static and dynamic component of PEM swapped, to see if model of dividing PEM is valid

    param time_array: Time interval of relevance 
    type time_array: (len(time_array),) numpy.ndarray
    param initial_stokes_vector: Initial stokes vector containing intensity and polarization information 
    type initial_stokes_vector: (4,) numpy.ndarray
    
    return: 
    I_detect: final stokes vector
    type I_detect: (len(time_array), 4) numpy.ndarray
    '''
      
    # Modulate signal first with static PEM component
    static_PEM = static_PEM_matrix(stat_bf_angle, stat_bf)
    
    #Get dynamic PEM matrix
    dynamic_PEM = dynamic_PEM_matrix(time_array)
    
    I_dynamic_mod = dynamic_PEM @ initial_stokes_vector
    
    I_static_mod = I_dynamic_mod @ static_PEM.T 
    
    # Create Mueller matrix of dynamic (ideal) PEM 
    
    #Define Mueller matrix of Glan-Thompson polarizer
    gt_polarizer = linear_polarizer_matrix(gt_angle)
    
    #Compute entire output stokes vector and separated results for linear and CPL component
    I_detect = I_static_mod @ gt_polarizer.T
   
    return I_detect

def resolve_freq(modulated_stokes_vector, time_array, plot = True):
    '''
    Carry out FFT with modulated intensity 
    
    param modulated_intensity: Modulated Stokes vector 
    type modulated_intensity: len(time_array)x4x1 numpy.ndarray 
    param time_array: To modulated intensity associated time array 
    type time_array: (len(time_array),) numpy.ndarray
    param plot: Optionally plot Fourier transform and time series of frequency components
    type plot: boolean
    
    return: 
    I_mod_freq: Dictionary containing:
        key 'Time': Orignally utilized time interval
        type 'Time': len(time_array)x4x1 numpy.ndarray 
        key 'I_total': Modulated intensity
        type 'I_total': (len(time_array), 4) numpy.ndarray
        key 'Freq': Frequency 'x' axis of FFT
        type 'Freq': (len(time_array)/2 + 1,) numpy.ndarray
        key 'Freq_mag': Magnitude of appearing frequencies after FFT ('y' axis of FFT) (what is the scale?)
        type 'Feq_mag': (len(time_array)/2 + 1,) numpy.ndarray
        key 'xx kHz': Frequency contributions back in time space
        type 'xx kHz': len(time_array)x4x1 numpy.ndarray 
    
    '''
    
    #Extract modulated intensity out of Stokes vector
    I_mod = modulated_stokes_vector[:,0]
    
    #Compute Fourier coefficients of signal
    I_modf = np.fft.rfft(I_mod)
    
    #Compute the corresponding frequencies (parameters to plug in: window length n = len(I_mod), d: time spacing between data points)
    freq = np.fft.rfftfreq(len(time_array), d = time_array[1]-time_array[0])
    freq_magnitude = np.abs(I_modf)/len(I_modf)
    
    #Plot intensity modulation 
    if plot == True:
        fig, axes = plt.subplots(2, 1, figsize = (20,20))
        axes[0].plot(time_array*1e6, I_mod, label = 'Modulated I(t)')
        axes[0].set_xlabel(r'Time in $\mu$s')
        axes[0].set_ylabel('Modulated Intensity in a.u.')
        axes[0].grid(True)
        
        #Plot fourier transform
        axes[1].plot(freq[:50]*1e-3, freq_magnitude[:50])
        axes[1].set_xlabel('Frequency in kHz')
        axes[1].set_ylabel('Magnitude of frequencies (Normalized to DC component)')
        axes[1].grid(True)
    
    #Find all contributing frequencies 
    boolean_mask = freq_magnitude > 1e-5 #choose threshold to 1 % of DC component (scaled to 1 in DC)
    filtered_freq = freq[boolean_mask]
    idx_freq = np.where(boolean_mask)[0]
    
    #Prepare sum array to check if following loop is computing the frequency components correctly
    summe =  np.zeros_like(I_mod)
    
    #Initialize dictionary containing all series of frequency data
    I_mod_freq = {'Time': time_array, 'I_total': I_mod, 'Freq': freq, 'Freq_mag': freq_magnitude}
    
    for i in idx_freq:
        
        #Zero out all frequencies except the target frequencies and its conjugate
        I_modf_filtered = np.zeros_like(I_modf)
        I_modf_filtered[i] = I_modf[i]
        
        #Perform inverse FFT on single frequency contributions
        I_mod_freq[f'{int(np.round(freq[i]*1e-3, 0))} kHz'] = np.fft.irfft(I_modf_filtered)
        if plot == True:
            axes[0].plot(time_array*1e6, I_mod_freq[f'{int(np.round(freq[i]*1e-3, 0))} kHz'], label = str(int(np.round(freq[i]*1e-3, 0))) + ' kHz')
        summe += I_mod_freq[f'{int(np.round(freq[i]*1e-3, 0))} kHz']
    
    if plot == True:
        axes[0].legend(loc = 'upper right')
    
    #Optionally control if individual signals sum up to modulated intensity
    # if plot == True:
    #     axes[0].plot(time_array*1e6, summe, color = 'black')
    
    return I_mod_freq
    
def extract_glum(I_mod, f_mod = 50e3):
    '''
    Extract glum from modulated Stokes vector
    
    param I_mod: Modulated intensity with all frequency components as time series (output of resolve_freq) 
    type I_mod: dict with keys type 'xx kHz'
    
    return type glum: float
    
    '''
       
    #Find all local maxima and minima in 1f-modulated data
    try:
        f_mod_str = [f'{int(np.round(f_mod*1e-3, 0))} kHz']
        if I_mod[f_mod_str[0]][0] >= 0:
            I_L_idx, _ = find_peaks(I_mod[f_mod_str[0]]) 
            I_R_idx, _ = find_peaks(-I_mod[f_mod_str[0]]) 
        else:
            I_R_idx, _ = find_peaks(I_mod[f_mod_str[0]])
            I_L_idx, _ = find_peaks(-I_mod[f_mod_str[0]])
        
    except KeyError:
        print('No CPL expected!')
        return 0
    
    #Prepare looping through all appearing odd multiples of f modulations
    f_mod_m = f_mod * 3
    I_L = I_mod[f_mod_str[0]][I_L_idx]
    I_R = I_mod[f_mod_str[0]][I_R_idx]
    
    while f'{int(np.round(f_mod_m*1e-3, 0))} kHz' in I_mod.keys():
        
        f_mod_str.append(f'{int(np.round(f_mod_m*1e-3, 0))} kHz')
        f_mod_m = f_mod_m + f_mod * 2
        
        #Add signal of higher odd orders of modulation
        I_L += I_mod[f_mod_str[-1]][I_L_idx]
        I_R += I_mod[f_mod_str[-1]][I_R_idx]
    
    #Average over all maxima in cycle
    I_L_final = np.average(I_L)
    I_R_final = np.average(I_R)
    
    #Test for theoretical correction factor (should be 0.8821 from Bessel functions formalism (Kitzmann et al))
    # corr_factor_I_L = np.average(I_L/I_mod[f_mod_str[0]][I_L_idx])
    # corr_factor_I_R = np.average(I_R/I_mod[f_mod_str[0]][I_R_idx])
    # print(corr_factor_I_L, corr_factor_I_R)
    
    glum = 2*(I_L_final-I_R_final)/(I_L_final + I_R_final + 2*np.average(I_mod['0 kHz']))
    
    return glum
    
def extract_LinearArtefact(I_mod, f_mod = 50e3):
    '''
    Extract linear artefacts from modulated Stokes vector
    
    param I_mod: Modulated intensity with all frequency components as time series (output of resolve_freq) 
    type I_mod: dict with keys type 'xx kHz'
 
    return type lp: float
    
    '''

    #Find all local maxima in 2f-modulated data
    try:
        f_mod_str = [f'{int(np.round(2*f_mod*1e-3, 0))} kHz']
        I_LP_idx_max, _ = find_peaks(I_mod[f_mod_str[0]]) 
        I_LP_idx_min, _ = find_peaks(-I_mod[f_mod_str[0]])

    except KeyError:
        print('No Linear Artefacts expected!')
        return 0
    
    #Prepare looping through all appearing even multiples of f modulations
    f_mod_m = f_mod * 4
    I_LP_max = I_mod[f_mod_str[0]][I_LP_idx_max]
    I_LP_min = -I_mod[f_mod_str[0]][I_LP_idx_min]
    
    while f'{int(np.round(f_mod_m*1e-3, 0))} kHz' in I_mod.keys():
        
        f_mod_str.append(f'{int(np.round(f_mod_m*1e-3, 0))} kHz')
        f_mod_m = f_mod_m + f_mod * 2
        
        #Add signal of higher even orders of modulation
        I_LP_max += I_mod[f_mod_str[-1]][I_LP_idx_max]
        I_LP_min += -I_mod[f_mod_str[-1]][I_LP_idx_min]
 
    
    #Average over all maxima in cycle
    I_LP_final = np.average(np.concatenate((I_LP_max, I_LP_min)))#/(np.average(I_mod['0 kHz']))
    
    #Handwaving solution for correct sign
    if I_mod[f'{int(np.round(2*f_mod*1e-3, 0))} kHz'][0] < 0:
        I_LP_final = -I_LP_final
    
    return I_LP_final

