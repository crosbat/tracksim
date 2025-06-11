"""
Contains utility function used in the main tracksim module as well as other
useful functions.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def moving_average(a,n):
    ma = np.cumsum(a, dtype=float)
    ma[n:] = ma[n:] - ma[:-n]
    
    return ma[n-1:]/n

def exp_average(a, alpha):
    
    ma = np.zeros(len(a))
    ma[0] = a[0]    
    for i in range(1, len(a)):
        ma[i] = alpha*a[i]+ (1-alpha)*ma[i-1]
    
    return ma

def plot_vehicle_and_battery_data(vehicle):
    
    time = vehicle.simulation_results['Time [s]']
    
    sim_len = len(time)
    
    speed = vehicle.simulation_results['Actual speed [m/s]']
    acceleration = vehicle.simulation_results['Actual acceleration [m/s2]']
    power = vehicle.simulation_results['Battery power demand [W]']
    
    pack_current = vehicle.pack.simulation_results['Pack']['Current [A]']
    pack_voltage = vehicle.pack.simulation_results['Pack']['Voltage [V]']
    pack_avg_soc = vehicle.pack.simulation_results['Pack']['Avg SOC']
    pack_min_soc = vehicle.pack.simulation_results['Pack']['Min SOC']
    pack_max_soc = vehicle.pack.simulation_results['Pack']['Max SOC']
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    if vehicle.pack.cells_are_identical:
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20,10), sharex=True)
    else:
        fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(20,10), sharex=True)
    
    ax[0,1].set_title('Vehicle Data')
    
    ax[0,0].set_ylabel('Speed [km/h]')
    ax[0,0].plot(time, speed*3.6)
    
    ax[0,1].set_ylabel('Acceleration [m/s2]')
    ax[0,1].plot(time, acceleration)
    
    ax[0,2].set_ylabel('Power Demand [kW]')
    ax[0,2].plot(time, power/1000)
    
    ax[1,1].set_title('Pack Data')
    
    ax[1,0].set_ylabel('Current [A]')
    ax[1,0].plot(time, pack_current)
    
    ax[1,1].set_ylabel('Voltage [V]')
    ax[1,1].plot(time, pack_voltage)
    
    ax[1,2].set_ylabel('SOC')
    ax[1,2].plot(time, pack_avg_soc, label='Avg', color='tab:blue')
    ax[1,2].plot(time, pack_max_soc, label='Max/Min', linestyle='--', color='tab:blue')
    ax[1,2].plot(time, pack_min_soc, linestyle='--', color='tab:blue')
    ax[1,2].legend()
    
    if vehicle.pack.cells_are_identical:
        
        ax[2,1].set_title('Cell Data')
        
        cell_current = vehicle.pack.simulation_results['Cell 0-0']['Current [A]']
        cell_voltage = vehicle.pack.simulation_results['Cell 0-0']['Voltage [V]']
        cell_soc = vehicle.pack.simulation_results['Cell 0-0']['SOC']
        
        ax[2,0].set_ylabel('Current [A]')
        ax[2,0].plot(time, cell_current)
        ax[2,0].set_xlabel('Time [s]')
        
        ax[2,1].set_ylabel('Voltage [V]')
        ax[2,1].plot(time, cell_voltage)
        ax[2,1].set_xlabel('Time [s]')
        
        ax[2,2].set_ylabel('SOC')
        ax[2,2].plot(time, cell_soc)
        ax[2,2].set_xlabel('Time [s]')
    
    else:
        
        ax[2,1].set_title('PCM Data')
        
        for j in range(min(vehicle.pack.Np, 16)):
            cell_current = vehicle.pack.simulation_results[f'Cell 0-{j}']['Current [A]']
            cell_voltage = vehicle.pack.simulation_results[f'Cell 0-{j}']['Voltage [V]']
            cell_soc = vehicle.pack.simulation_results[f'Cell 0-{j}']['SOC']
            
            ax[2,0].set_ylabel('Current [A]')
            ax[2,0].plot(time, cell_current, label=f'Cell 0-{j}')
            ax[2,0].set_xlabel('Time [s]')
            ax[2,0].legend()
            
            ax[2,1].set_ylabel('Voltage [V]')
            ax[2,1].plot(time, cell_voltage, label=f'Cell 0-{j}')
            ax[2,1].set_xlabel('Time [s]')
            ax[2,1].legend()
            
            ax[2,2].set_ylabel('SOC')
            ax[2,2].plot(time, cell_soc, label=f'Cell 0-{j}')
            ax[2,2].set_xlabel('Time [s]')
            ax[2,2].legend()
        
        number_of_pcm_to_plot = min(vehicle.pack.Ns, 16)
        
        for i in range(number_of_pcm_to_plot):
            
            PCM_current = np.zeros(shape=(sim_len, vehicle.pack.Np))
            PCM_voltage = np.zeros(shape=(sim_len, vehicle.pack.Np))
            PCM_soc = np.zeros(shape=(sim_len, vehicle.pack.Np))
            
            for j in range(vehicle.pack.Np):
                PCM_current[:,j] = vehicle.pack.simulation_results[f'Cell {i}-{j}']['Current [A]']
                PCM_voltage[:,j] = vehicle.pack.simulation_results[f'Cell {i}-{j}']['Voltage [V]']
                PCM_soc[:,j] = vehicle.pack.simulation_results[f'Cell {i}-{j}']['SOC']
            
            PCM_current = np.sum(PCM_current, axis=1)
            PCM_voltage = np.mean(PCM_voltage, axis=1)
            PCM_soc = np.mean(PCM_soc, axis=1)
            
            ax[3,0].set_ylabel('Current [A]')
            ax[3,0].plot(time, PCM_current, label=f'PCM {i}')
            ax[3,0].set_xlabel('Time [s]')
            
            ax[3,1].set_ylabel('Voltage [V]')
            ax[3,1].plot(time, PCM_voltage, label=f'PCM {i}')
            ax[3,1].set_xlabel('Time [s]')
            
            ax[3,2].set_ylabel('SOC')
            ax[3,2].plot(time, PCM_soc, label=f'PCM {i}')
            ax[3,2].set_xlabel('Time [s]')
            ax[3,1].legend(bbox_to_anchor=(0.5,-0.4), loc='center', ncol=8)
            
    return None