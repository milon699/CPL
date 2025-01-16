# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:17:56 2024

@author: Milon
"""

import numpy as np
import matplotlib.pyplot as plt

#%%

plt.rcParams['font.size'] = 15.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 1
plt.close('all')

#%%

#Time array
f_mod = 50 
time = np.linspace(0, 1/f_mod, 100)

#Define linearly polarized input vector (arbitrary direction)

#Angle of polarization (relative to horizontal axis)
theta = 45

#Total intensity of incoming light beam
S_0 = 1

#Actual Stokes vector
I_0 = np.array([S_0, S_0*np.cos(2*theta), S_0*np.sin(2*theta), 0])

#Now define Mueller matrices of static and dynamic component of PEM
#Static birefringence defect
delta = 0.10

#axis of birefringence
alpha = 0

#Define Mueller matrix of static effect of PEM
static_PEM = np.array([[1,0,0,0], 
                       [0, (np.cos(2*alpha))**2 + (np.sin(2*alpha))**2 * np.cos(delta), np.cos(2*alpha) * np.sin(2*alpha) * (1-np.cos(delta)), -np.sin(2*alpha) * np.sin(delta)],
                       [0, np.cos(2*alpha) * np.sin(2*alpha) * (1-np.cos(delta)), (np.sin(2*alpha))**2 + (np.cos(2*alpha))**2 * np.cos(delta), np.cos(2*alpha)*np.sin(delta)],
                       [0, np.sin(2*alpha) * np.sin(delta), -np.cos(2*alpha)*np.sin(delta), np.cos(delta)]])



#Define Mueller matrix of dynamic effect of 
#Try to avoid for loops and create [len(time_steps), 4, 4] matrix 

#Initialize result array
dynamic_PEM = np.zeros((len(time), 4, 4))
phases = phase_ret(time, f_mod = f_mod)

#Create upper identity from downer phase retardation seperately
top_left = np.eye(2)
cos_phases = np.cos(phases)
sin_phases = np.sin(phases)
bottom_right = np.stack([np.stack([cos_phases, -sin_phases], axis = 1),np.stack([sin_phases, cos_phases], axis = 1)], axis = -1)

dynamic_PEM[:, :2, :2] = top_left
dynamic_PEM[:, 2:, 2:] = bottom_right

#Define Mueller matrix of Glan-Thompson polarizer
gt_polarizer = np.array([[0.5, 0, -0.5, 0],
                        [0, 0, 0, 0],
                        [-0.5, 0, 0.5, 0],
                        [0, 0, 0, 0]])

#Now compute end polarization state by multiplying matrices with incoming Stokes vector
I_detect = gt_polarizer @ dynamic_PEM @ static_PEM @ I_0

I_fake = static_PEM @ I_0
I_fake_CPL = np.array([abs(I_fake[3]), 0, 0, I_fake[3]])
I_fake_linear = np.array([np.sqrt(I_fake[1]**2+I_fake[2]**2), I_fake[1], I_fake[2], 0])

I_detect_fake_CPL = gt_polarizer @ dynamic_PEM @ I_fake_CPL
I_detect_fake_linear = gt_polarizer @ dynamic_PEM @ I_fake_linear

#Plot total intensity as a function of time 
plt.figure('I(t)')
plt.plot(time, I_detect[:,0], color = 'black', label = 'Total')
plt.plot(time, I_detect_fake_CPL[:,0], color = 'blue', label = 'CPL')
plt.plot(time, I_detect_fake_linear[:,0], color = 'red', label = 'LP')

plt.legend()
plt.xlabel('Time in s')
plt.ylabel('Intensity in a.u.')
plt.grid()
