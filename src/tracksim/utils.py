"""
Contains utility function used in the main tracksim module as well as other
useful functions.
"""

import os
import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from scipy.optimize import minimize

# =============================================================================
# Utility functions for tracksim module
# =============================================================================

def make_clean_dir(path: str) -> None:
    """
    Cleans a giving directory. If the directory does not exist, then it will
    be created.

    Parameters
    ----------
    path : str
        Path to the directory.

    Returns
    -------
    None.

    """
    if path not in os.listdir():
        print(f"\nMaking '{path}'")
        os.mkdir(path)

    else:
        print(f"\nPurging '{path}'")
        shutil.rmtree(path)
        os.mkdir(path)
    
    return None

def get_cell_currents_voltages(vf: np.ndarray, 
                               r0: np.ndarray, 
                               desired_power: float,  
                               charge_current_is_positive: bool, 
                               Ns: int, 
                               Np: int,
                               cells_are_identical: bool) -> tuple:
    """
    Calculates the required cell currents and voltages to meet the desired
    power of the battery pack. The battery pack is modeled as an equivalent
    Thevenin circuit i.e.
    
    vT = vf - rT*I
    
    where vf is the voltage of the voltage source, rT is the Thevenin
    equivalent resistance, and I is the battery pack current (negative 
    charge current). The Thevenin equivalent resistance is calcuated based on
    the individual series resistance of each cell. 

    Parameters
    ----------
    vf : numpy.ndarray
        Non-instantaneous voltage of each battery cell.
    r0 : numpy.ndarray
        Series resistance of each cell.
    desired_power : float
        Desired power for the current step.
    cells_are_identical : float
        If true, then the thevenin equivalent voltage source and resistance are
        assumed to be the same for all cells. This simplifies the calculation
        of vT and rT using the number of cells in series and parallel. If False
        then the cells are treated as not being equal which can slow down
        computations.
    charge_current_is_positive : bool
        Indicates the direction of the current.
    Ns : int
        Number of cells in series.
    Np : int
        Number of cells in parallel.

    Returns
    -------
    ik : numpy.ndarray
        Individual cell currents for the current step.
    vk : numpy.ndarray
        Individual cell voltages for the current step.
    I : float
        Battery pack current for the current step.
    V : float
        Battery pack voltage for the current step.

    """
    if cells_are_identical:
        
        rT = r0/Np # Thevenin eq. resistance per module
        vT = vf # Thevenin eq. voltage per module
        rT_pack = Ns*rT # Thevenin eq. resistance for whole pack
        vT_pack = Ns*vT # Thevenin eq. voltage for whole pack
        
        if charge_current_is_positive:
            
            I = (vT_pack-np.sqrt(vT_pack**2-4*rT_pack*desired_power)
                 )/(-2*rT_pack) # Find necessary current for the desired power
            
            V = vT_pack + rT_pack*I # Get pack voltage
            
            vk = V/Ns # PCM terminal voltages
            ik = (vk-vf)/r0 # Individual cell currents
            
            vk = vf + r0*ik # Get individual cell voltages
        
        else:
            
            I = (vT_pack-np.sqrt(vT_pack**2-4*rT_pack*desired_power)
                 )/(2*rT_pack) # Find necessary current for the desired power
            V = vT_pack - rT_pack*I # Get pack voltage
            
            vk = V/Ns # PCM terminal voltages
            ik = (vf-vk)/r0 # Individual cell currents
            
            vk = vf - r0*ik # Get individual cell voltages
    
    else:
        rT = 1/np.sum(1/r0,axis=1) # Thevenin eq. resistance per module
        vT = np.sum(vf/r0,axis=1)*rT # Thevenin eq. voltage per module
        rT_pack = np.sum(rT) # Thevenin eq. resistance for whole pack
        vT_pack = np.sum(vT) # Thevenin eq. voltage for whole pack
        
        if charge_current_is_positive:
            I = (vT_pack-np.sqrt(vT_pack**2-4*rT_pack*desired_power)
                 )/(-2*rT_pack) # Find necessary current for the desired power
            V = vT_pack + rT_pack*I
            
            vk = (np.sum(vf/r0,axis=1)+I)/np.sum(1/r0,axis=1) # PCM terminal voltages
            vk = np.tile(vk, (Np,1)).T
            ik = (vk-vf)/r0 # Individual cell currents
            
            vk = vf + r0*ik # Get individual cell voltages
        
        else:
            I = (vT_pack-np.sqrt(vT_pack**2-4*rT_pack*desired_power)
                 )/(2*rT_pack) # Find necessary current for the desired power
            V = vT_pack - rT_pack*I # Get pack voltage
            
            vk = (np.sum(vf/r0,axis=1)-I)/np.sum(1/r0,axis=1) # PCM terminal voltages
            vk = np.tile(vk, (Np,1)).T
            ik = (vf-vk)/r0 # Individual cell currents
            
            vk = vf - r0*ik # Get individual cell voltages

    return ik, vk, I, V

def obj_func_positive_charge_current(ik : float, 
                                     v_inst : callable,
                                     vf : float,
                                     desired_power : float,
                                     Ns : int,
                                     Np : int) -> float:
    
    return ((vf + v_inst(ik))*ik - desired_power/(Ns*Np))**2

def obj_func_negative_charge_current(ik : float, 
                                     v_inst : callable,
                                     vf : float,
                                     desired_power : float,
                                     Ns : int) -> float:
    
    return ((vf - v_inst(ik))*ik - desired_power/Ns)**2

def get_cell_currents_voltages_optimization(vf: np.ndarray,
                                            v_inst : callable,
                                            ik_init : float,
                                            desired_power : float,
                                            charge_current_is_positive: bool, 
                                            Ns: int, 
                                            Np: int) -> tuple:
        
    if charge_current_is_positive:
        
        partial_func = partial(obj_func_positive_charge_current, 
                               v_inst = v_inst,
                               vf = vf,
                               desired_power = -desired_power,
                               Ns = Ns, Np = Np)
        
        ik = minimize(partial_func, ik_init).x.reshape(1,1)
        
        vk = vf + v_inst(ik)
    
    else:
        
        partial_func = partial(obj_func_negative_charge_current, 
                               v_inst = v_inst,
                               vf = vf,
                               desired_power = desired_power,
                               Ns = Ns)
        
        ik = minimize(partial_func, ik_init).x.reshape(1,1)
        
        vk = vf - v_inst(ik)
    
    I = ik*Np
    V = vk*Ns
    
    return ik, vk, I, V

def check_if_cells_are_identical(cell_model : np.ndarray | dict,
                                 initial_soc : np.ndarray | float | int,
                                 initial_temp : np.ndarray | float | int,
                                 initial_rc_current : np.ndarray | float | int,
                                 coolant_temp : np.ndarray | float | int) -> bool:
    
    initial_socs_are_identical = False
    initial_temps_are_identical = False
    initial_ircs_are_identical = False
    coolant_temps_are_identical = False
    
    cell_model_is_array = isinstance(cell_model, np.ndarray)
    soc_is_array = isinstance(initial_soc, np.ndarray)
    temp_is_array = isinstance(initial_temp, np.ndarray)
    irc_is_array = isinstance(initial_rc_current, np.ndarray)
    coolant_temp_is_array = isinstance(coolant_temp, np.ndarray)
    
    if not soc_is_array:
        initial_socs_are_identical = True
    else:
        initial_socs_are_identical = np.allclose(initial_soc, initial_soc[0,0])
    
    if not temp_is_array:
        initial_temps_are_identical = True
    else:
        initial_temps_are_identical = np.allclose(initial_temp, initial_temp[0,0])
    
    if not irc_is_array:
        initial_ircs_are_identical = True
    else:
        initial_ircs_are_identical = np.allclose(initial_rc_current, initial_rc_current[0,0])
    
    if not coolant_temp_is_array:
        coolant_temps_are_identical = True
    else:
        coolant_temps_are_identical = np.allclose(coolant_temp, coolant_temp[0,0])
    
    if cell_model_is_array:
        warnings.warn('When passing an array of cell models, it is assumed they will be treated as not identical during simulation. This will increase time and storage complexity in the simulation. If the cell models are meant to be identical, then please pass the cell model as a single dictionary.')
    
    return (not cell_model_is_array) & initial_socs_are_identical & initial_temps_are_identical & initial_ircs_are_identical & coolant_temps_are_identical

# =============================================================================
# Other useful functions
# =============================================================================

def moving_average(a : iter, n : int) -> np.array:
    ma = np.cumsum(a, dtype=float)
    ma[n:] = ma[n:] - ma[:-n]
    
    return ma[n-1:]/n

def exp_average(a : iter, alpha : float) -> np.array:
    
    ma = np.zeros(len(a))
    ma[0] = a[0]    
    for i in range(1, len(a)):
        ma[i] = alpha*a[i]+ (1-alpha)*ma[i-1]
    
    return ma

def translate_exp_term(term):
    lambda_coeff = float(term.split('*')[0].split('[')[1])
    return f'np.exp({lambda_coeff}*np.sqrt(np.abs(I)))'

def convert_pybatteryid_model_to_tracksim(pybid_model_path : str) -> dict:
    """
    Loads and converts an LPV model generated from the PyBatteryID package into
    a model structure compatible with TRACKSIM.

    Parameters
    ----------
    pybid_model_path : str
        Path to the PyBatteryID model.

    Raises
    ------
    ImportError
        Raised if the PyBatteryID package is not installed.

    Returns
    -------
    dict
        Converted cell model compatible with TRACKSIM.

    """
    try:
        from pybatteryid.utilities import load_model_from_file
    except ImportError:
        raise ImportError("Running this function requires an installation of PyBatteryID. Please install this package using 'pip install pybatteryid'.")

    basis_function_dict = {'s' : 'SOC',
                           '1/s' : '(1/SOC)',
                           'log[s]' : 'np.log(SOC)',
                           'd[0,1]' : 'np.sign(I)',
                           'd[0,0]' : 'np.sign(I)'}
    
    supported_basis_functions = list(basis_function_dict.keys())
    supported_basis_functions.append('exp[lambda*sqrt[|i|]]')
    
    pybid_model = load_model_from_file(pybid_model_path)

    # Define cell model for TRACKSIM

    tracksim_model = {'Model name' : 'LPV',
                      'Reference' : 'N/A',
                      'Description' : f'Converted from file: {pybid_model_path}',
                      'Cathode' : None,
                      'Anode' : None,
                      'Form factor' : None,
                      'Nominal voltage [V]' : None,
                      'Min voltage [V]' : None,
                      'Max voltage [V]' : None,
                      'Nominal capacity [As]' : None,
                      'Mass [kg]' : None,
                      'Model type' : 'LPV',
                      'Model order' : pybid_model.model_order,
                      'Nonlinearity order' : pybid_model.nonlinearity_order,
                      'Model SOC range [%]' : '0 - 100',
                      'Model temperature range [C]' : '0 - 40',
                      'Positive charging current' : True,
                      'Capacity [As]' : pybid_model.battery_capacity,
                      'Coulombic efficiency' : 1,
                      'OCV [V]': lambda SOC=0.5,T=None : pybid_model.emf_function(SOC, T)} 

    # Make list of ARX coefficients (a1, a2, ..., b0, b1, b2, ...)
    arx_coeffs = []
    for i in range(pybid_model.model_order):
        arx_coeffs.append(f'a{i+1}')

    for i in range(pybid_model.model_order+1):
        arx_coeffs.append(f'b{i}')

    # Group model terms using the same v or i measurement together
    arx_terms = [] # List of nested lists i.e. [[term strings for a1], [term strings for a2], ...]
    term_coeffs = [] # List of nested lists i.e. [[coefficients for a1], [coefficients for a2], ...]
    
    for arx_coeff in arx_coeffs:
        
        coeff_index = int(arx_coeff[1:]) # i.e. 0,1,2,3,...
        
        if 'a' in arx_coeff:
            string_to_search_for = f'v(k-{coeff_index})'
        
        else:
            # If 'b' in arx_coeff
            
            if coeff_index == 0:
                string_to_search_for = 'i(k)'
            
            else:
                string_to_search_for = f'i(k-{coeff_index})'
    
        relevant_indices = [index for index, term in enumerate(pybid_model.model_terms) if string_to_search_for in term]
        
        arx_terms.append(list(pybid_model.model_terms[relevant_indices]))
        term_coeffs.append(list(pybid_model.model_estimate[relevant_indices]))

    # Translate the model estimates and terms into lambda expression for TRACKSIM
    for arx_coeff, terms, coeffs in zip(arx_coeffs, arx_terms, term_coeffs):
        
        lambda_string = 'lambda SOC=0.5, T=25, I=0 : '
    
        for term, coeff in zip(terms, coeffs):
            
            lambda_string = lambda_string + str(coeff)
            
            if '×' in term:
                
                term_parts = term.split('×')[1:]
                
                for term_part in term_parts:
        
                    variable = term_part.split('(')[0]
                    
                    if 'exp' in variable:
                        lambda_string = lambda_string + '*' + translate_exp_term(variable)
                    
                    else:
                        try:
                            lambda_string = lambda_string + '*' + basis_function_dict[variable]
                        except KeyError:
                            raise KeyError(f'The term {variable} is currently not supported. Please use a model consisting only of the following supported basis functions: {supported_basis_functions}')                        
                        
            lambda_string = lambda_string + ' + ' # Add '+' sign for future terms
            
        lambda_string = lambda_string[:-3] # Remove the last '+' sign since we are finished
        # print(lambda_string)
        tracksim_model[arx_coeff] = eval(lambda_string) # Convert the lambda string to a callable and add it to the model dict
    
    return tracksim_model

def plot_vehicle_and_battery_data(vehicle) -> None:
    
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
  
if __name__ == '__main__':
    
    pybid_model_path = 'PyBatteryID_models/model_n,l=3,4.npy'
    
    tracksim_model = convert_pybatteryid_model_to_tracksim(pybid_model_path)
    
    # Add additional cell parameters
    
    tracksim_model['Mass [kg]'] = 0.1
    tracksim_model['Nominal capacity [As]'] = 3600
    tracksim_model['Nominal voltage [V]'] = 3.6
    
    # Test model in a vehicle
    
    import pandas as pd
    
    from vehicle_models import ChevyVoltTuned
    from pack_models import ChevyVoltPack
    from tracksim import Vehicle, Pack
    
    pack_model = ChevyVoltPack.copy()
    
    pack_model['Number of cells in series'] = 32
    pack_model['Number of cells in parallel'] = 16
    
    pack = Pack(pack_model, tracksim_model)
    vehicle = Vehicle(ChevyVoltTuned, pack)
    
    udds = pd.read_csv('example_trip_data/udds.csv')
    
    vehicle.simulate_vehicle(udds['Time [s]'], udds['Speed [m/s]'], 1)
    vehicle.simulate_battery_pack()
    
    plot_vehicle_and_battery_data(vehicle)