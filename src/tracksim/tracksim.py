import pickle
import time
import os
import shutil
import warnings
import numpy as np
import pandas as pd
import libsumo as ls

from tqdm import tqdm
from multiprocessing import Pool
from scipy.optimize import minimize
from functools import partial

# =============================================================================
# Utility functions
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
# Traffic simulation functions
# =============================================================================

class Traffic():
    """
    Class used to define and run the traffic simulation. The main method of 
    this class is 'simulate_traffic'.
    """
    def __init__(self, 
                 config_path: str, 
                 output_dir: str ='simulated_trip_files', 
                 duration: int = 1, 
                 time_step: float = 1,
                 record_position: bool = False,
                 to_geo: bool = True,
                 record_lane: bool = False, 
                 pbar: bool = True,
                 checkpoint_dir: str = 'trip_checkpoints',
                 lite_mode_ratio: bool = None,
                 random_state: int = None,
                 remove_checkpoints_when_finished: bool = True) -> None:
        """
        Initializes the Traffic class used for simulating the vehicle traffic. 

        Parameters
        ----------
        config_path : str
            The path to the SUMO configuration file for the scenario to be 
            simulated.
        output_dir : str, optional
            Directory to store the final simulated trip files. By default, the
            trip files are stored in a directory named 'simulated_trip_files'.
        duration : int, optional
            Duration of simulation in hours. The default is 1 hour.
        time_step : float, optional
            Length of time between samples in the simulation in seconds. The
            default is 1 second between each sample. The minimum value of
            'time_step' is 0.1 seconds.
        record_position : bool, optional
            Records the x and y position in the network of each vehicle in the 
            simulation if True. Enabling this will increase file sizes. 
            The default is False.
        to_geo : bool, optional
            Converts the position of the vehicle into geograpic coordinates 
            (longitude, latitude) if True. Only used if record_position is True.
            The default is True.
        record_lane : bool, optional
            Records the ID of the current lane in the network of each vehicle 
            in the simulation if True. Enabling this will increase file sizes. 
            The default is False.
        pbar : bool, optional
            Displays a progress bar during the simulation if True. The default 
            is True.
        lite_mode_ratio : float, optional
            Can be set as a number between 0 and 1 which gives the ratio of trips 
            to process in the 'process_checkpoints' method. If 'lite_mode_ratio' 
            is 0.1, only 10% of the trips are processed. The trips to process 
            are randomnly selected. If None, then all trips are processed, If 0, 
            then no trips are processed and will have to processed manually by 
            the user by calling the 'process_checkpoints' method. NOTE: all trips 
            still need to be simulated, this variable only affects the processing
            of the simulated trips after simulation. The default is None.
        random_state : int or str, optional
            Sets the seed for the randomizer used to shuffle the order of the 
            trips to process in 'process_checkpoints'. If None, then the 
            randomizer is initialized without setting a seed. If 'off', then 
            the order to process the trips is not randomized. Note: if 
            random_state is 'off', then it will switch to None if lite_mode_ratio 
            is not None. The default is None.
        remove_checkpoints_when_finished : bool, optional
            Removes the intermediate checkpoints generated after simulation
            and before the final processing if True. These files are mainly 
            used to leverage disk storage over memory storage and can easily be 
            removed after final processing. It is a good idea to turn this off
            if you want use 'process_checkpoints' multiple times with
            different values of 'lite_mode_ratio' or 'random_state' and do
            not need to run the SUMO simulation again.

        Returns
        -------
        None.

        """
        
        self.config_path = config_path
        self.output_dir = output_dir
        self.duration = duration
        self.time_step = time_step
        self.record_position = record_position
        self.to_geo = to_geo
        self.record_lane = record_lane
        self.pbar = pbar
        self.lite_mode_ratio = lite_mode_ratio
        
        if self.lite_mode_ratio == 1:
            # Process all trips
            self.lite_mode_ratio = None
        
        if self.lite_mode_ratio is not None:
            if (self.lite_mode_ratio > 1) or (self.lite_mode_ratio < 0):
                raise ValueError("Please provide 'lite_mode_ratio' as a number between 0 (inclusive) and 1 (inclusive)")
        
        if (self.lite_mode_ratio is not None) and (random_state == 'off'):
            # We need randomness to shuffle the trips
            warnings.warn("'random_state' has been switcehd to None since 'lite_mode_ratio' is not None")
            self.random_state = None
        
        self.random_state = random_state
        self.checkpoint_dir = checkpoint_dir
        self.remove_checkpoints_when_finished = remove_checkpoints_when_finished
        
        return None
    
    def update_vehicle_data(self, veh_id: str, data: dict, step: int) -> None:
        
        if veh_id not in data.keys():
            # Initialize entry for vehicle
            data[veh_id] = dict()
            data[veh_id]['Time [s]'] = [] # Current time in simulation [s]
            data[veh_id]['Speed [m/s]'] = [] # Speed [m/s]
            
            if self.record_position and not self.to_geo:
                data[veh_id]['x [m]'] = [] # x position
                data[veh_id]['y [m]'] = [] # y position
            
            elif self.record_position and self.to_geo:
                data[veh_id]['Longitude'] = [] # Longitude
                data[veh_id]['Latitude'] = [] # Latitude
            
            if self.record_lane:
                data[veh_id]['Lane ID'] = [] # Network lane id
        
        data[veh_id]['Time [s]'].append(step*self.time_step)
        
        data[veh_id]['Speed [m/s]'].append(ls.vehicle.getSpeed(veh_id))
          
        if self.record_position:
            pos = ls.vehicle.getPosition(veh_id)
            
            if not self.to_geo:
                data[veh_id]['x [m]'].append(pos[0])
                data[veh_id]['y [m]'].append(pos[1])
            else:
                lon, lat = ls.simulation.convertGeo(*pos)
                data[veh_id]['Longitude'].append(lon)
                data[veh_id]['Latitude'].append(lat)
       
        if self.record_lane:
            data[veh_id]['Lane ID'].append(ls.vehicle.getLaneID(veh_id))
        
        return None

    def process_vehicle_data(self, veh_id: str) -> None:
        """
        Processes the vehicle data for one vehicle so the trip is combined into
        one CSV file.
    
        Parameters
        ----------
        veh_id : str
            ID of the vehicle.
    
        Returns
        -------
        None.
    
        """
        
        veh_files = [file for file in os.listdir(self.checkpoint_dir) if  veh_id == file.split('_')[0]] # Get all checkpoints for this vehicle
        veh_files.sort()
        
        with open(f'{self.checkpoint_dir}/{veh_files[0]}', 'rb') as file:
            veh_dict = pickle.load(file)
        
        for veh_file in veh_files[1:]:
            with open(f'{self.checkpoint_dir}/{veh_file}', 'rb') as file:
                veh_dict_part = pickle.load(file)
            
            for key in veh_dict.keys():
                veh_dict[key] = veh_dict[key] + veh_dict_part[key]
        
        for key in veh_dict.keys():
            # Make all lists to arrays for easier processing
            veh_dict[key] = np.array(veh_dict[key])
        
        # Convert to DataFrame and save as csv file
        
        veh_df = pd.DataFrame(veh_dict)
        
        veh_df.to_csv(f'{self.output_dir}/{veh_id}.csv', index=False)
    
        return None

    def process_checkpoints(self) -> None:
        """
        Processes the vehicle data generated by the method 'simulate_traffic' 
        so data from each vehicle is combined into one CSV file instead of 
        being split between multiple files. Each vehicle is processed in
        parallel. The resulting trip files can be found in the folder
        'simulated_trips'.
    
        Returns
        -------
        None.
    
        """
        
        print('\nProcessing checkpoints')
        
        # Get the ID of every vehicle
        veh_ids = list({file.split('_')[0] for file in os.listdir(self.checkpoint_dir)})
        
        if self.random_state != 'off':
            if self.random_state is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(self.random_state)
        
            rng.shuffle(veh_ids)
        
        if self.lite_mode_ratio is not None:
            veh_ids = veh_ids[:int(np.ceil(len(veh_ids)*self.lite_mode_ratio))]
        
        pool = Pool()
        
        if self.pbar is not None:
            pbar=tqdm(total=len(veh_ids), position=0, leave=True)
            for _ in pool.imap_unordered(self.process_vehicle_data, veh_ids):
                pbar.update()
    
        else:
            pool.imap_unordered(self.process_vehicle_data, veh_ids)
        
        pool.close()
        
        return None

    def save_vehicle_data(self, veh_id: str, data: dict) -> None:
        
        veh_dict = data[veh_id]
        
        for key in veh_dict.keys():
            # Make all lists to arrays for easier processing
            veh_dict[key] = np.array(veh_dict[key])
        
        # Convert to DataFrame and save as csv file
        
        veh_df = pd.DataFrame(veh_dict)
        
        veh_df.to_csv(f'{self.output_dir}/{veh_id}.csv', index=False)
        
        return None

    def simulate_traffic(self, checkpoint_length = 3600) -> None:
        """
        Simulates the traffic from a SUMO config file and tracks the data for 
        each vehicle in the simulation. The simulation saves vehicle data 
        periodically in order to save on memory.
        
        Parameters
        ----------
        checkpoint_length : int
            Number od simulation seconds to process before saving the data as a 
            checkpoint. For instance, if checkpoint_length = 3600, then the
            simulation data is saved to disk every simulation hour. Reducing
            this will save on memory usage but will increase the number of
            checkpoints saved to disk. The default is 3600.
        
        Returns
        -------
        None.
    
        """
        
        make_clean_dir(self.checkpoint_dir)
        
        print('\nStarting simulation')
        
        time_start = time.time()
        
        data = dict() # Initialize simulation data storage
        
        # Initialize simulation
        ls.start(["sumo", "-c", self.config_path, "--step-length", str(self.time_step)])
        print(f'\nStarted simulation with timedelta: {ls.simulation.getDeltaT()}s')
        n_steps = np.floor(self.duration*3600*(1/self.time_step))
        
        if self.pbar:
            pbar = tqdm(total=n_steps, position=0, leave=True) # Define progress bar
        
        # Run simulation
        step=0
        while step < n_steps:
            
            ls.simulationStep()
            veh_list = ls.vehicle.getIDList() # Get vehicles on the road
            # veh_list = ['veh' + veh_id if 'veh' not in veh_id else veh_id for veh_id in veh_list] # Sometimes the 'veh' indicator is omitted (eg. in the Berlin scenario) and needs to be put back in
            
            try:
                for veh_id in veh_list:
                    self.update_vehicle_data(veh_id, data, step)
            except IndexError:
                pass
            
            step += 1
            
            if step > 0 and step%(3600/self.time_step)==0:
                # For every 1 hour
                
                timeslot_index = int(step/(3600/self.time_step))
                
                for veh_id in data.keys():
                    # Save the trip data for this vehicle in this timeslot
                    
                    with open(f'{self.checkpoint_dir}/{veh_id}_{timeslot_index}.pickle', 'wb') as file:
                        pickle.dump(data[veh_id], file)
                
                data = dict() # Reset data storage
            
            if self.pbar:
                pbar.update()
        
        ls.close()
        
        if self.pbar:
            pbar.close()
        
        print(f'\n Finished simulation in {time.time()-time_start:.2f} seconds!')
        
        make_clean_dir(self.output_dir)
        
        if self.lite_mode_ratio != 0:
            # If 'lite_mode_ratio' is zero, then do not process any trips
    
            self.process_checkpoints()
        
            if self.remove_checkpoints_when_finished:
                
                print(f"\nRemoving '{self.checkpoint_dir}'")
                shutil.rmtree(f'{self.checkpoint_dir}')
            
        return None
    
# =============================================================================
# Vehicle definitions
# =============================================================================

class Pack():
    """
    Class used to define a battery pack comprising of one or more modules.
    """

    def __init__(self, pack_model: dict, 
                 cell_model: dict, 
                 temperature_model: dict = None) -> None:
        """
        Initializes the Pack class

        Parameters
        ----------
        pack_model : dict
            Dictionary describing the battery pack. The dictionary has to
            follow the same format as those in tracksim.pack_models.
        cell_model : dict or numpy.ndarray of dicts
            Dict or array of dicts describing the model of the cells in the 
            battery pack. The format of the dict has to follow those in t
            racksim.cell_models. If one dict is given, then each cell is
            assumed to follow the same model. If an array of dicts is given,
            then each each cell is assumed to have a distinct model. WARNING:
            having different cells will significantly increase computation time.
        temperature_model : dict or numpy.ndarray of dicts, optional
            Dict or array of dicts describing the temperature model of the 
            cells in the battery pack. The format of the dict has to follow 
            those in tracksim.temperature_models. If one dict is given, then 
            each cell is assumed to follow the same temperature model. If an 
            array of dicts is given, then each each cell is assumed to have a 
            distinct model. WARNING: having different cells will significantly 
            increase computation time.

        Raises
        ------
        ValueError
            Raised if cell_model or temperature_model are not dicts or array of
            dicts.
        KeyError
            Raised if the 'Model type' value is not supported.

        Returns
        -------
        None.

        """
        
        self.pack_model = pack_model
        
        self.Ns = self.pack_model['Number of cells in series']*self.pack_model['Number of modules']
        self.Np = self.pack_model['Number of cells in parallel']
        self.n_cells = self.Ns*self.Np
        
        self.cell_model = cell_model
        self.cell_model_is_array = isinstance(self.cell_model, np.ndarray)
        
        if (not self.cell_model_is_array) and (not isinstance(self.cell_model, dict)):
            raise ValueError('Please give the cell model as either a dictionary or an array of dictionaries. See tracksim.cell_models for a selection of compatible models.')
        
        # Check if the shape of the model array is the same as the shape of the battery pack
        
        if self.cell_model_is_array:
            if not self.cell_model.shape == (self.Ns, self.Np):
                raise ValueError(f"The shape of the cell model array is inconsistent with the shape of the battery pack. The shape of the cell model array is {self.cell_model.shape} while the shape of the battery pack is {(self.Ns, self.Np)}")
        
        # Check if the model type is supported
        
        supported_model_types = ['ECM', 'ARX', 'LPV']
        
        if self.cell_model_is_array:
            if self.cell_model[0,0]['Model type'] not in supported_model_types:
                raise ValueError(f"The model type '{self.cell_model[0,0]['Model type']}' is currently not supported. Supported model types: {supported_model_types}.")
        
        else:
            if self.cell_model['Model type'] not in supported_model_types:
                raise ValueError(f"The model type '{self.cell_model['Model type']}' is currently not supported. Supported model types: {supported_model_types}.")
        
        # Check if the models are of the same type and order
        
        if self.cell_model_is_array:
            
            if self.cell_model[0,0]['Model Type'] == 'ECM':
                
                for i in range(self.Ns):
                    for j in range(self.Np):
                        
                        # Check if the model types are the same
                        if not self.cell_model[i,j]['Model type'] == self.cell_model[0,0]['Model type']:
                            raise ValueError(f"The types of the cell models are not the same. The model type of cell (0,0) is {self.cell_model[0,0]['Model type']} while model type of cell ({i},{j}) is {self.cell_model[i,j]['Model type']}.")
                
                        # Check if the number of RC pairs are the same
                        if not self.cell_model[i,j]['Number of RC pairs'] == self.cell_model[0,0]['Number of RC pairs']:
                            raise ValueError(f"The number of RC pairs in the cell models are not the same. The number of RC pairs in cell (0,0) is {self.cell_model[0,0]['Number of RC pairs']} while the number of RC pairs in cell ({i},{j}) is {self.cell_model[i,j]['Number of RC pairs']}.")
            
            elif self.cell_model[0,0]['Model Type'] in ['LPV', 'ARX']:
                
                for i in range(self.Ns):
                    for j in range(self.Np):
                        
                        # Check if the model types are the same
                        if not self.cell_model[i,j]['Model type'] == self.cell_model[0,0]['Model type']:
                            raise ValueError(f"The types of the cell models are not the same. The model type of cell (0,0) is {self.cell_model[0,0]['Model type']} while model type of cell ({i},{j}) is {self.cell_model[i,j]['Model type']}.")
                
                        # Check if the model orders are the same
                        if not self.cell_model[i,j]['Model order'] == self.cell_model[0,0]['Model order']:
                            raise ValueError(f"The model order in the cell models are not the same. The model order in cell (0,0) is {self.cell_model[0,0]['Model order']} while the model order in cell ({i},{j}) is {self.cell_model[i,j]['Model order']}.")
            
        # Check if the model(s) are dynamic
        
        if self.cell_model_is_array:
            self.cell_model_is_dynamic = np.zeros(shape=(self.Ns, self.Np), dtype=bool)
        
            if self.cell_model[0,0]['Model type'] == 'ECM':
                for i in range(self.Ns):
                    for j in range(self.Np):
                        if callable(self.cell_model[i,j]['R0 [Ohm]']):
                            self.cell_model_is_dynamic[i,j] = True
        
            elif self.cell_model[0,0]['Model type'] in ['LPV', 'ARX']:
                for i in range(self.Ns):
                    for j in range(self.Np):
                        if callable(self.cell_model[i,j]['b0']):
                            self.cell_model_is_dynamic[i,j] = True
        else:
            if self.cell_model['Model type'] == 'ECM':
                self.cell_model_is_dynamic = callable(self.cell_model['R0 [Ohm]'])
            
            elif self.cell_model['Model type'] in ['LPV', 'ARX']:
                self.cell_model_is_dynamic = callable(self.cell_model['b0'])
        
        self.temperature_model = temperature_model
        self.temperature_model_is_array = isinstance(self.temperature_model, np.ndarray)
        
        if (temperature_model is not None) and (not self.temperature_model_is_array) and (not isinstance(self.temperature_model, dict)):
            raise ValueError('Please give the temperature model as either a dictionary or an array of dictionaries. See tracksim.temperature_models for a selection of compatible models.')
        
        # Check if the shape of the model array is the same as the shape of the battery pack
        
        if self.temperature_model_is_array:
            if not self.temperature_model.shape == (self.Ns, self.Np):
                raise ValueError(f"The shape of the temperature model array is inconsistent with the shape of the battery pack. The shape of the temperature model array is {self.cell_model.shape} while the shape of the battery pack is {(self.Ns, self.Np)}")
        
        # Get pack mass, charge capacity, and energy capacity
        
        if self.cell_model_is_array:
            
            cell_masses = np.zeros(shape=(self.Ns,self.Np))
            cell_nominal_charge_capacities = np.zeros(shape=(self.Ns,self.Np))
            cell_nominal_voltages = np.zeros(shape=(self.Ns,self.Np))
            
            for i in range(self.Ns):
                for j in range(self.Np):
                    cell_masses[i,j] = self.cell_model[i,j]['Mass [kg]']
                    cell_nominal_charge_capacities[i,j] = self.cell_model[i,j]['Nominal capacity [As]']
                    cell_nominal_voltages[i,j] = self.cell_model[i,j]['Nominal voltage [V]']
            
            self.mass = np.sum(cell_masses)*1/(1-self.pack_model['Battery module overhead'])*1/(1-self.pack_model['Battery pack overhead'])
            self.nominal_charge_capacity = np.min(np.sum(cell_nominal_charge_capacities, axis=1))
            self.nominal_energy_capacity = np.sum(np.mean(cell_nominal_voltages, axis=1))*self.nominal_charge_capacity
            
        else:
            self.mass = self.n_cells*self.cell_model['Mass [kg]']*1/(1-self.pack_model['Battery module overhead'])*1/(1-self.pack_model['Battery pack overhead'])
            self.nominal_charge_capacity = self.Np * self.cell_model['Nominal capacity [As]']
            self.nominal_energy_capacity = self.nominal_charge_capacity*self.Ns*self.cell_model['Nominal voltage [V]']/3600 # Wh
        
        if self.nominal_energy_capacity < 20000:
            warnings.warn(f"The nominal energy capacity for this pack is {self.nominal_energy_capacity/1000:.2f} kWh, which is considered low for typical EVs. This may have adverse effects in the simulation if the voltage of the cells goes outside the normal bounds. To increase the nominal capacity, please consider increasing the number of cells by changing the 'Number of cells in series' or 'Number of cells in parallel' values in the pack model.")
        
        self.efficiency = self.pack_model['Battery pack efficiency']
        
        self.initial_conditions = dict()
        self.simulation_results = None
        self.charge_current_is_positive = self.cell_model['Positive charging current']
        self.cells_are_identical = None
        
        return None
    
    def simulate_pack(self, 
                      desired_power : iter, 
                      sample_period : int | float, 
                      initial_soc : np.ndarray | int |float = 0.8,
                      initial_temp : np.ndarray | int |float = 25,
                      initial_rc_current : np.ndarray | int |float = 0,
                      coolant_temp : np.ndarray | int |float | None = None):
        """
        Simulates the battery pack with the given input power profile. The
        battery pack is assumed to be initially at rest.

        Parameters
        ----------
        desired_power : iter
            Input power profile in Watts.
        sample_period : int | float
            Time between samples in seconds.
        initial_soc : np.ndarray | int | float, optional
            Initial SOC of the cells in the pack, either as a Ns x Np numpy 
            array or as a single number (same across all cells). The default 
            is 0.8.
        inital_temp : np.ndarray | int | float, optional
            Initial temperature of the cells in the pack in Celsius, either as 
            a Ns x Np numpy array or as a single number (same across all cells). 
            The default is 25 C.
        initial_rc_current : np.ndarray | int | float, optional
            Initial diffusion current of the cells in the pack in Ampere, either 
            as a Ns x Np numpy array or as a single number (same across all 
            cells). Only used if the cell model is an Equivalent Circuit Model
            The default is 0 A (cells are at rest).

        Raises
        ------
        KeyError
            Raised if the cell model is of an unrecognized type.

        Returns
        -------
        None.

        """
        
        if self.cell_model['Model type'] == 'ECM':
            self.cells_are_identical = check_if_cells_are_identical(self.cell_model, 
                                                                    initial_soc, 
                                                                    initial_temp, 
                                                                    initial_rc_current, 
                                                                    coolant_temp)
            
            self._simulate_pack_ECM(desired_power, 
                                    initial_soc, 
                                    initial_temp,
                                    coolant_temp,
                                    sample_period)
            
        elif self.cell_model['Model type'] in ['LPV', 'ARX']:
            self.cells_are_identical = check_if_cells_are_identical(self.cell_model, 
                                                                    initial_soc, 
                                                                    initial_temp, 
                                                                    initial_rc_current=0, 
                                                                    coolant_temp=0)
            
            self._simulate_pack_LPV(desired_power, 
                                    initial_soc,
                                    initial_temp,
                                    sample_period)
            
        else:
            raise KeyError('The model type is currently not supported.')
        
        return None

    def _format_initial_condition(self,
                                  input_ : np.ndarray | int | float, 
                                  condition_name : str, 
                                  variable_name : str) -> np.ndarray:
        
        if isinstance(input_, np.ndarray):    
            if not input_.shape == (self.Ns, self.Np):
                raise ValueError(f"The shape of '{variable_name}' is inconsistent with the shape of the battery pack. The shape of '{variable_name}' is {input_.shape} while the shape of the battery pack is {(self.Ns, self.Np)}")
            
            formatted_input = input_
        
        elif isinstance(input_, (int, float)):
            
            if self.cells_are_identical:
                formatted_input = np.ones(shape=(1, 1))*input_
            
            else:
                formatted_input = np.ones(shape=(self.Ns, self.Np))*input_
                
        else:
            raise ValueError(f"'{variable_name}' should be either np.ndarray, int, or float")
        
        self.initial_conditions[condition_name] = formatted_input
        
        return formatted_input

    def _initialize_simulation_ECM(self, 
                                  sim_len : int, 
                                  sample_period : int | float, 
                                  num_rc_pairs : int) -> None:
        """
        Initializes the storage for the simulation results. This method is only
        intended to be used by the simulate_pack_ECM method and not on its own.

        Parameters
        ----------
        sim_len : int
            Length of simulation in samples.
        sample_period : float
            Time between samples in seconds.
        num_rc_pairs : int
            Number of RC pairs in the ECM. Must be at least 1.

        Returns
        -------
        None.

        """
        
        if self.cells_are_identical:
            Ns = 1
            Np = 1
        else:
            Ns = self.Ns
            Np = self.Np
        
        self.simulation_results = dict()
        
        # Set up storage
        self.simulation_results['Time [s]'] = np.arange(sim_len)*sample_period # s
        self.simulation_results['Pack'] = dict()
        self.simulation_results['Pack']['Current [A]'] = np.zeros(sim_len) # A
        self.simulation_results['Pack']['Voltage [V]'] = np.zeros(sim_len) # V
        self.simulation_results['Pack']['Min SOC'] = np.zeros(sim_len) # Minimum SOC of all cells
        self.simulation_results['Pack']['Max SOC'] = np.zeros(sim_len) # Maximum SOC of all cells
        self.simulation_results['Pack']['Avg SOC'] = np.zeros(sim_len) # Average SOC of all cells
        self.simulation_results['Pack']['Min temperature [C]'] = np.zeros(sim_len) # Minimum temperature of all cells
        self.simulation_results['Pack']['Max temperature [C]'] = np.zeros(sim_len) # Maximum temperature of all cells
        self.simulation_results['Pack']['Avg temperature [C]'] = np.zeros(sim_len) # Average temperature of all cells
        for i in range(Ns):
            for j in range(Np):
                self.simulation_results[f'Cell {i}-{j}'] = dict()
                self.simulation_results[f'Cell {i}-{j}']['R0 [Ohm]'] = np.zeros(sim_len) # Ohm
                self.simulation_results[f'Cell {i}-{j}']['Current [A]'] = np.zeros(sim_len) # A
                self.simulation_results[f'Cell {i}-{j}']['Voltage [V]'] = np.zeros(sim_len) # V
                self.simulation_results[f'Cell {i}-{j}']['SOC'] = np.zeros(sim_len)
                self.simulation_results[f'Cell {i}-{j}']['OCV [V]'] = np.zeros(sim_len) # V
                self.simulation_results[f'Cell {i}-{j}']['Temperature [C]'] = np.zeros(sim_len) # Deg C
                for l in range(num_rc_pairs):
                    self.simulation_results[f'Cell {i}-{j}'][f'RC{l+1} current [A]'] = np.zeros(sim_len) # A
                    self.simulation_results[f'Cell {i}-{j}'][f'R{l+1} [Ohm]'] = np.zeros(sim_len) # Ohm
                    self.simulation_results[f'Cell {i}-{j}'][f'C{l+1} [F]'] = np.zeros(sim_len) # F
        
        return None
    
    def _simulate_pack_ECM(self, 
                           desired_power : iter, 
                           initial_soc : np.ndarray | int | float,
                           initial_temp : np.ndarray | int | float,
                           coolant_temp : np.ndarray | int | float,
                           sample_period : int | float) -> None:
        """
        Simulates the battery pack under the given battery power profile.

        Parameters
        ----------
        desired_power : iterable
            Profile of the demanded power in W.
        initial_soc : np.ndarray | int | float, optional
            Initial SOC of the cells in the pack, either as a Ns x Np numpy 
            array or as a single number (same across all cells). The default 
            is 0.8.
        inital_temp : np.ndarray | int | float, optional
            Initial temperature of the cells in the pack in Celsius, either as 
            a Ns x Np numpy array or as a single number (same across all cells). 
            The default is 25 C.
        initial_rc_current : np.ndarray | int | float, optional
            Initial diffusion current of the cells in the pack in Ampere, either 
            as a Ns x Np numpy array or as a single number (same across all 
            cells). Only used if the cell model is an Equivalent Circuit Model
            The default is 0 A (cells are at rest).
        sample_period : float
            Time between samples in seconds.

        Returns
        -------
        None.

        """
        
        # Initialize temporary variables
        
        if self.cells_are_identical:
            Ns = 1
            Np = 1
            
        else:    
            Ns = self.Ns
            Np = self.Np
        
        if self.cell_model_is_array:
            num_rc_pairs = self.cell_model[0,0]['Number of RC pairs']
        else:
            num_rc_pairs = self.cell_model['Number of RC pairs']
        
        sim_len = len(desired_power)
        
        # Initialize cell states
        z = self._format_initial_condition(initial_soc, 'SOC', 'initial_soc')
        T = self._format_initial_condition(initial_temp, 'Cell temperature [C]', 'initial_temp')
        irc = np.zeros(shape=(num_rc_pairs, Ns, Np))
        
        if not coolant_temp is None:
            Tf = self._format_initial_condition(coolant_temp, 'Coolant temperature [C]', 'coolant_temp')
        else:
            Tf = None
        
        # Initialize cell parameters
        
        q = np.zeros(shape=(Ns,Np)) # Ns x Np
        eta = np.zeros(shape=(Ns,Np)) # Ns x Np
        r0 = np.zeros(shape=(Ns,Np)) # Ns x Np
        r = np.zeros(shape=(num_rc_pairs, Ns, Np)) # num_rc_pairs x Ns x Np
        c = np.zeros(shape=(num_rc_pairs, Ns, Np)) # num_rc_pairs x Ns x Np
        rc = np.zeros(shape=(num_rc_pairs, Ns, Np)) # num_rc_pairs x Ns x Np
        
        if self.cell_model_is_array:
            for i in range(Ns):
                for j in range(Np):
                    q[i,j] = self.cell_model[i,j]['Capacity [As]']
                    eta[i,j] = self.cell_model[i,j]['Coulombic efficiency']
                    
                    if self.cell_model_is_dynamic[i,j]:
                        r0[i,j] = self.cell_model[i,j]['R0 [Ohm]'](z[i,j], T[i,j])
                        r0[i,j] += 2*self.cell_model[i,j]['Tab resistance [Ohm]']
                        
                        for l in range(num_rc_pairs):
                            r[l,i,j] = self.cell_model[i,j][f'R{l+1} [Ohm]'](z[i,j], T[i,j])
                            c[l,i,j] = self.cell_model[i,j][f'C{l+1} [F]'](z[i,j], T[i,j])
                            rc[l,i,j] = np.exp(-sample_period/np.abs(r[l,i,j]*c[l,i,j]))
                    else:
                        r0[i,j] = self.cell_model[i,j]['R0 [Ohm]']
                        r0[i,j] += 2*self.cell_model[i,j]['Tab resistance [Ohm]']
                        
                        for l in range(num_rc_pairs):
                            r[l,i,j] = self.cell_model[i,j][f'R{l+1} [Ohm]']
                            c[l,i,j] = self.cell_model[i,j][f'C{l+1} [F]']
                            rc[l,i,j] = np.exp(-sample_period/np.abs(r[l,i,j]*c[l,i,j]))
                
        else:
            q = np.ones(shape=(Ns,Np))*self.cell_model['Capacity [As]']
            eta = np.ones(shape=(Ns,Np))*self.cell_model['Coulombic efficiency']
            
            if self.cell_model_is_dynamic:
                r0 = self.cell_model['R0 [Ohm]'](z, T)
                r0 += 2*np.ones(shape=(Ns,Np))*self.cell_model['Tab resistance [Ohm]'] # Add tab resistance for each cell
                
                r = np.zeros(shape=(num_rc_pairs, Ns, Np))
                c = np.zeros(shape=(num_rc_pairs, Ns, Np))
                rc = np.zeros(shape=(num_rc_pairs, Ns, Np))
                
                for l in range(num_rc_pairs):
                    r[l,:,:] = self.cell_model[f'R{l+1} [Ohm]'](z, T)
                    c[l,:,:] = self.cell_model[f'C{l+1} [F]'](z, T)
                    rc[l,:,:] = np.exp(-sample_period/np.abs(r[l,:,:]*c[l,:,:]))
            
            else:
                r0 = np.ones(shape=(Ns,Np))*self.cell_model['R0 [Ohm]']
                r0 += 2*np.ones(shape=(Ns,Np))*self.cell_model['Tab resistance [Ohm]'] # Add tab resistance for each cell
                
                r = np.zeros(shape=(num_rc_pairs, Ns, Np))
                c = np.zeros(shape=(num_rc_pairs, Ns, Np))
                rc = np.zeros(shape=(num_rc_pairs, Ns, Np))
                
                for l in range(num_rc_pairs):
                    r[l,:,:] = np.ones(shape=(Ns,Np))*self.cell_model[f'R{l+1} [Ohm]']
                    c[l,:,:] = np.ones(shape=(Ns,Np))*self.cell_model[f'C{l+1} [F]']
                    rc[l,:,:] = np.exp(-sample_period/np.abs(r[l,:,:]*c[l,:,:]))
        
        if self.temperature_model is not None:
            
            h = np.zeros(shape=(Ns,Np))
            A = np.zeros(shape=(Ns,Np))
            m = np.zeros(shape=(Ns,Np))
            Cp = np.zeros(shape=(Ns,Np))
            
            if self.temperature_model_is_array:
                if self.cell_model_is_array:
                    for i in range(Ns):
                        for j in range(Np):
                            h[i,j] = self.temperature_model[i,j]['Equivalent convective heat transfer coefficient [W/(m2K)]']
                            A[i,j] = self.cell_model[i,j]['Surface area [m2]']
                            m[i,j] = self.cell_model[i,j]['Mass [kg]']*1000 # g
                            Cp[i,j] = self.temperature_model[i,j]['Specific heat capacity [J/(kgK)]']*1000 # J/gK
            
                else:
                    for i in range(Ns):
                        for j in range(Np):
                            h[i,j] = self.temperature_model[i,j]['Equivalent convective heat transfer coefficient [W/(m2K)]']
                            A[i,j] = self.cell_model['Surface area [m2]']
                            m[i,j] = self.cell_model['Mass [kg]']*1000 # g
                            Cp[i,j] = self.temperature_model[i,j]['Specific heat capacity [J/(kgK)]']*1000 # J/gK       
            
            else:
                if self.cell_model_is_array:
                    for i in range(Ns):
                        for j in range(Np):
                            h[i,j] = self.temperature_model['Equivalent convective heat transfer coefficient [W/(m2K)]']
                            A[i,j] = self.cell_model[i,j]['Surface area [m2]']
                            m[i,j] = self.cell_model[i,j]['Mass [kg]']*1000 # g
                            Cp[i,j] = self.temperature_model['Specific heat capacity [J/(kgK)]']*1000 # J/gK
                
                else:
                    # If both cell model and temperature model are dictionaries
                    
                    h = np.ones(shape=(Ns,Np))*self.temperature_model['Equivalent convective heat transfer coefficient [W/(m2K)]']
                    A = np.ones(shape=(Ns,Np))*self.cell_model['Surface area [m2]']
                    m = np.ones(shape=(Ns,Np))*self.cell_model['Mass [kg]']*1000 # g
                    Cp = np.ones(shape=(Ns,Np))*self.temperature_model['Specific heat capacity [J/(kgK)]']*1000 # J/gK
        
            e = np.exp((-h*A*sample_period)/m*Cp)
        
        self._initialize_simulation_ECM(len(desired_power), sample_period, num_rc_pairs) # Initialize storage
        
        for k in range(sim_len):
            
            if self.cell_model_is_array:
                ocv = np.zeros(shape=(Ns,Np))
                
                for i in range(Ns):
                    for j in range(Np):
                        ocv[i,j] = self.cell_model[i,j]['OCV [V]'](z[i,j], T[i,j])
            
            else:
                ocv = self.cell_model['OCV [V]'](z,T) # Get OCV for each cell
            
            if self.charge_current_is_positive:
                v_cells = ocv + np.sum(r*irc, axis=0) # Add diffusion voltages
                ik, vk, I, V = get_cell_currents_voltages(v_cells, r0, 
                                                          desired_power[k],
                                                          self.charge_current_is_positive,
                                                          self.Ns, self.Np,
                                                          self.cells_are_identical)
                ik[ik>0] = ik[ik>0]*eta[ik>0] # Multiply by eta for cells where we are charging
                z += (sample_period/q)*ik # Update SOC
                
            else:
                v_cells = ocv - np.sum(r*irc, axis=0) # Add diffusion voltages
                ik, vk, I, V = get_cell_currents_voltages(v_cells, r0, 
                                                          desired_power[k],
                                                          self.charge_current_is_positive,
                                                          self.Ns, self.Np,
                                                          self.cells_are_identical)
                ik[ik<0] = ik[ik<0]*eta[ik<0] # Multiply by eta for cells where we are charging
                z -= (sample_period/q)*ik # Update SOC
            
            irc = rc*irc + (1-rc)*np.tile(ik, (num_rc_pairs,1,1)) # Update RC resistor currents
            
            # Update temperature
            if self.temperature_model is not None:
                
                if self.temperature_model_is_array:
                    
                    dOCV_dT = np.zeros(shape=(Ns,Np))

                    for i in range(Ns):
                        for j in range(Np):
                            dOCV_dT[i,j] = self.temperature_model[i,j]['Entropic heat coefficient'](z[i,j])
                
                else:
                    dOCV_dT = self.temperature_model['Entropic heat coefficient'](z)
                
                if self.charge_current_is_positive:
                    Qk = ik**2*r0 - ik*np.sum(irc*r, axis=0) - ik*(T+273.15)*dOCV_dT
                
                else:
                    Qk = ik**2*r0 + ik*np.sum(irc*r, axis=0) + ik*(T+273.15)*dOCV_dT
                
                if Tf:
                    T = e*(T+273.15) + (1-e)*(Qk/(h*A) + (Tf+273.15)) - 273.15
                else:
                    # No cooling
                    T = e*(T+273.15) + (1-e)*(Qk/(h*A) + (T+273.15)) - 273.15
            
            # Update ECM parameters
            
            if self.cell_model_is_array:
                if self.cell_model_is_dynamic.any():
                    for i in range(Ns):
                        for j in range(Np):
                            
                            if self.cell_model_is_dynamic[i,j]:
                                r0[i,j] = self.cell_model[i,j]['R0 [Ohm]'](z[i,j], T[i,j])
                                
                                for l in range(num_rc_pairs):
                                    r[l,i,j] = self.cell_model[i,j][f'R{l+1} [Ohm]'](z[i,j], T[i,j])
                                    c[l,i,j] = self.cell_model[i,j][f'C{l+1} [F]'](z[i,j], T[i,j])
                                    rc[l,:,:] = np.exp(-sample_period/np.abs(r[l,:,:]*c[l,:,:]))

            else:
                # if cell model is not array (dict)
                if self.cell_model_is_dynamic:
                    r0 = self.cell_model['R0 [Ohm]'](z, T)
                    for l in range(num_rc_pairs):
                        r[l,:,:] = self.cell_model[f'R{l+1} [Ohm]'](z, T)
                        c[l,:,:] = self.cell_model[f'C{l+1} [F]'](z, T)
                        rc[l,:,:] = np.exp(-sample_period/np.abs(r[l,:,:]*c[l,:,:]))
            
            # Store measurements
            self.simulation_results['Pack']['Current [A]'][k] = I
            self.simulation_results['Pack']['Voltage [V]'][k] = V 
            self.simulation_results['Pack']['Min SOC'][k] = np.min(z)
            self.simulation_results['Pack']['Max SOC'][k] = np.max(z)
            self.simulation_results['Pack']['Avg SOC'][k] = np.mean(z)
            self.simulation_results['Pack']['Min temperature [C]'][k] = np.min(T)
            self.simulation_results['Pack']['Max temperature [C]'][k] = np.max(T)
            self.simulation_results['Pack']['Avg temperature [C]'][k] = np.mean(T)
            
            for i in range(Ns):
                for j in range(Np):
                    self.simulation_results[f'Cell {i}-{j}']['R0 [Ohm]'][k] = r0[i,j]
                    self.simulation_results[f'Cell {i}-{j}']['Current [A]'][k] = ik[i,j]
                    self.simulation_results[f'Cell {i}-{j}']['Voltage [V]'][k] = vk[i, j]
                    self.simulation_results[f'Cell {i}-{j}']['SOC'][k] = z[i,j]
                    self.simulation_results[f'Cell {i}-{j}']['OCV [V]'][k] = ocv[i,j]
                    self.simulation_results[f'Cell {i}-{j}']['Temperature [C]'][k] = T[i,j]
                    for l in range(num_rc_pairs):
                        self.simulation_results[f'Cell {i}-{j}'][f'RC{l+1} current [A]'][k] = irc[l,i,j]
                        self.simulation_results[f'Cell {i}-{j}'][f'R{l+1} [Ohm]'][k] = r[l,i,j]
                        self.simulation_results[f'Cell {i}-{j}'][f'C{l+1} [F]'][k] = c[l,i,j]
            
        return None

    def _initialize_simulation_LPV(self, 
                                   sim_len : int, 
                                   sample_period : int | float, 
                                   model_order : int) -> None:
        """
        Initializes the storage for the simulation results. This method is only
        intended to be used by the simulate_pack_LPV method and not on its own.

        Parameters
        ----------
        sim_len : int
            Length of simulation in samples.
        sample_period : int | float
            Time between samples in seconds.
        model_order : int
            Order of the LPV model.

        Returns
        -------
        None.

        """
        
        if self.cells_are_identical:
            Ns = 1
            Np = 1
        else:
            Ns = self.Ns
            Np = self.Np
        
        self.simulation_results = dict()
        
        # Set up storage
        self.simulation_results['time'] = np.arange(sim_len)*sample_period # s
        self.simulation_results['Pack'] = dict()
        self.simulation_results['Pack']['Current [A]'] = np.zeros(sim_len) # A
        self.simulation_results['Pack']['Voltage [V]'] = np.zeros(sim_len) # V
        self.simulation_results['Pack']['Min SOC'] = np.zeros(sim_len) # Minimum SOC of all cells
        self.simulation_results['Pack']['Max SOC'] = np.zeros(sim_len) # Maximum SOC of all cells
        self.simulation_results['Pack']['Avg SOC'] = np.zeros(sim_len) # Average SOC of all cells
        self.simulation_results['Pack']['Min temperature [C]'] = np.zeros(sim_len) # Minimum temperature of all cells
        self.simulation_results['Pack']['Max temperature [C]'] = np.zeros(sim_len) # Maximum temperature of all cells
        self.simulation_results['Pack']['Avg temperature [C]'] = np.zeros(sim_len) # Average temperature of all cells
        for i in range(Ns):
            for j in range(Np):
                self.simulation_results[f'Cell {i}-{j}'] = dict()
                self.simulation_results[f'Cell {i}-{j}']['Current [A]'] = np.zeros(sim_len) # A
                self.simulation_results[f'Cell {i}-{j}']['Voltage [V]'] = np.zeros(sim_len) # V
                self.simulation_results[f'Cell {i}-{j}']['SOC'] = np.zeros(sim_len)
                self.simulation_results[f'Cell {i}-{j}']['OCV [V]'] = np.zeros(sim_len)
                self.simulation_results[f'Cell {i}-{j}']['Temperature [C]'] = np.zeros(sim_len) # Deg C
                self.simulation_results[f'Cell {i}-{j}']['b0'] = np.zeros(sim_len)
                for l in range(model_order):
                    self.simulation_results[f'Cell {i}-{j}'][f'a{l+1}'] = np.zeros(sim_len)
                    self.simulation_results[f'Cell {i}-{j}'][f'b{l+1}'] = np.zeros(sim_len)

    def _simulate_pack_LPV(self, 
                          desired_power : iter,
                          initial_soc : np.ndarray | int | float,
                          initial_temp : np.ndarray | int | float,
                          sample_period : int | float):
        """
        Simulates the battery pack under the given battery power profile.

        Parameters
        ----------
        desired_power : iterable
            Profile of the demanded power in W.
        initial_soc : np.ndarray | int | float, optional
            Initial SOC of the cells in the pack, either as a Ns x Np numpy 
            array or as a single number (same across all cells). The default 
            is 0.8.
        inital_temp : np.ndarray | int | float, optional
            Initial temperature of the cells in the pack in Celsius, either as 
            a Ns x Np numpy array or as a single number (same across all cells). 
            The default is 25 C.
        sample_period : int or float
            Time between samples in seconds.

        Returns
        -------
        None.

        """
        
        if self.initial_conditions is None:
            print('No initial conditions set. Initializing with default values.')
            self.set_initial_conditions_LPV()
        
        # Initialize temporary variables
        
        if self.cells_are_identical:
            Ns = 1
            Np = 1
        else:
            Ns = self.Ns
            Np = self.Np
        
        if self.cell_model_is_array:
            model_order = self.cell_model[0,0]['Model order']
        else:
            model_order = self.cell_model['Model order']
        
        sim_len = len(desired_power)
        
        # Initialize cell states
        z = self._format_initial_condition(initial_soc, 'SOC', 'initial_soc')
        T = self._format_initial_condition(initial_temp, 'Cell temperature [C]', 'initial_temp')
        
        # Check if the b0 parameter is depends on current (amplitude or direction)
        b0_depends_on_current = False
        test_currents = np.linspace(-10, 10, 5)
        if self.cell_model_is_array:
            for i in range(Ns):
                for j in range(Np):
                    if self.cell_model_is_dynamic[i,j]:
                        b0_values = self.cell_model['b0'](0.5, 25, test_currents)
                        
                        if not isinstance(b0_values, (int, float)):
                            b0_depends_on_current = True
        
        else:
            if self.cell_model_is_dynamic:
                b0_values = self.cell_model['b0'](0.5, 25, test_currents)
                
                if not isinstance(b0_values, (int, float)):
                    b0_depends_on_current = True
        
        # Initialize cell parameters
        
        q = np.zeros(shape=(Ns,Np)) # Ns x Np
        eta = np.zeros(shape=(Ns,Np)) # Ns x Np
        b0 = np.zeros(shape=(Ns,Np)) # Ns x Np
        a = np.zeros(shape=(model_order, Ns, Np)) # model_order x Ns x Np
        b = np.zeros(shape=(model_order, Ns, Np)) # model_order x Ns x Np
        
        # Initialize matrices for storing history values
        
        v_hist = np.ones(shape=(model_order, Ns, Np))*self.cell_model['OCV [V]'](z, T)
        ocv_hist = v_hist.copy()
        i_hist = np.zeros(shape=(model_order, Ns, Np)) # Cells are at rest
        T_hist = np.ones(shape=(model_order, Ns, Np))*T
        z_hist = np.ones(shape=(model_order, Ns, Np))*z
        
        if self.cell_model_is_array:
            for i in range(Ns):
                for j in range(Np):
                    q[i,j] = self.cell_model[i,j]['Capacity [As]']
                    eta[i,j] = self.cell_model[i,j]['Coulombic efficiency']
                    
                    if self.cell_model_is_dynamic[i,j]:
                        b0[i,j] = self.cell_model[i,j]['b0'](z[i,j], T[i,j])
                        
                        for l in range(model_order):
                            a[l,i,j] = self.cell_model[i,j][f'a{l+1}'](z_hist[l,i,j], T_hist[l,i,j], i_hist[l,i,j])
                            b[l,i,j] = self.cell_model[i,j][f'b{l+1}'](z_hist[l,i,j], T_hist[l,i,j], i_hist[l,i,j])
                    else:
                        b0[i,j] = self.cell_model[i,j]['b0']
                        
                        for l in range(model_order):
                            a[l,i,j] = self.cell_model[i,j][f'a{l+1}']
                            b[l,i,j] = self.cell_model[i,j][f'b{l+1}']
                
        else:
            q = np.ones(shape=(Ns,Np))*self.cell_model['Capacity [As]']
            eta = np.ones(shape=(Ns,Np))*self.cell_model['Coulombic efficiency']
            
            if self.cell_model_is_dynamic:
                b0 = self.cell_model['b0'](z, T)
                
                for l in range(model_order):
                    a[l,:,:] = self.cell_model[f'a{l+1}'](z_hist[l,:,:], T_hist[l,:,:], i_hist[l,:,:])
                    b[l,:,:] = self.cell_model[f'b{l+1}'](z_hist[l,:,:], T_hist[l,:,:], i_hist[l,:,:])
                  
            else:
                b0 = np.ones(shape=(Ns,Np))*self.cell_model['b0']
                
                for l in range(model_order):
                    a[l,:,:] = np.ones(shape=(Ns,Np))*self.cell_model[f'a{l+1}']
                    b[l,:,:] = np.ones(shape=(Ns,Np))*self.cell_model[f'b{l+1}']
        
        self._initialize_simulation_LPV(sim_len, sample_period, model_order) # Initialize storage
        
        for k in range(sim_len):
            
            # Get the current OCV
            
            if self.cell_model_is_array:
                ocv = np.zeros(shape=(Ns,Np))
                for i in range(Ns):
                    for j in range(Np):
                        ocv[i,j] = self.cell_model[i,j]['OCV [V]'](z[i,j], T[i,j])
                        
            else:
                ocv = self.cell_model['OCV [V]'](z,T) # Get OCV for each cell
            
            z_old = z.copy() # Save soc_k for later storing
            
            v_cells = np.sum(a*(v_hist-ocv_hist) + b*i_hist, axis=0) + ocv # Get Vf #TODO: Check if there should be a - sign on the a coefficients
            
            if not b0_depends_on_current:
                # Calculate cell currents and voltages using Thevenin modeling
                
                ik, vk, I, V = get_cell_currents_voltages(v_cells, b0,
                                                          desired_power[k],
                                                          self.charge_current_is_positive, 
                                                          self.Ns, self.Np,
                                                          self.cells_are_identical,)
            
            else:
                # Calculate cell currents and voltages using optimization
                
                if not self.cells_are_identical:
                    raise ValueError('Current-dependent b0 is currently not supported in packs with non-identical cells and initial conditions.')
                
                v_inst = lambda x : self.cell_model['b0'](SOC=z, T=T, I=x)*x
                
                ik, vk, I, V = get_cell_currents_voltages_optimization(v_cells, 
                                                                       v_inst,
                                                                       i_hist[0,0,0],
                                                                       desired_power[k],
                                                                       self.charge_current_is_positive, 
                                                                       self.Ns, self.Np)
                
            # Update dynamics    
            
            if self.charge_current_is_positive:
                ik_eta_corrected = ik.copy()
                ik_eta_corrected[ik_eta_corrected>0] = ik_eta_corrected[ik_eta_corrected>0]*eta[ik_eta_corrected>0] # Multiply by eta for cells where we are charging
                z += (sample_period/q)*ik_eta_corrected # Update SOC
            
            else:
                ik_eta_corrected = ik.copy()
                ik_eta_corrected[ik_eta_corrected<0] = ik_eta_corrected[ik_eta_corrected<0]*eta[ik_eta_corrected<0] # Multiply by eta for cells where we are charging
                z -= (sample_period/q)*ik_eta_corrected # Update SOC
            
            # Update history matrices
            
            for l in range(model_order-1):
                v_hist[-(l+1),:,:] = v_hist[-(l+2),:,:]
                ocv_hist[-(l+1),:,:] = ocv_hist[-(l+2),:,:]
                i_hist[-(l+1),:,:] = i_hist[-(l+2),:,:]
                T_hist[-(l+1),:,:] = T_hist[-(l+2),:,:]
                z_hist[-(l+1),:,:] = z_hist[-(l+2),:,:]
            
            v_hist[0,:,:] = vk
            ocv_hist[0,:,:] = ocv
            i_hist[0,:,:] = ik
            T_hist[0,:,:] = T
            z_hist[0,:,:] = z_old
            
            # Update LPV parameters
            
            if self.cell_model_is_array:
                for i in range(Ns):
                    for j in range(Np):
                        if self.cell_model_is_dynamic[i,j]:
                            b0[i,j] = self.cell_model[i,j]['b0'](z[i,j], T[i,j], ik[i,j])
                            
                            for l in range(model_order):
                                a[l,i,j] = self.cell_model[i,j][f'a{l+1}'](z_hist[l,i,j], T_hist[l,i,j], i_hist[l,i,j])
                                b[l,i,j] = self.cell_model[i,j][f'b{l+1}'](z_hist[l,i,j], T_hist[l,i,j], i_hist[l,i,j])
                
            else:
                if self.cell_model_is_dynamic:
                    b0 = self.cell_model['b0'](z, T, ik)
                    
                    for l in range(model_order):
                        a[l,:,:] = self.cell_model[f'a{l+1}'](z_hist[l,:,:], T_hist[l,:,:], i_hist[l,:,:])
                        b[l,:,:] = self.cell_model[f'b{l+1}'](z_hist[l,:,:], T_hist[l,:,:], i_hist[l,:,:])
                        
            # Store measurements
            self.simulation_results['Pack']['Current [A]'][k] = I
            self.simulation_results['Pack']['Voltage [V]'][k] = V
            self.simulation_results['Pack']['Min SOC'][k] = np.min(z_old)
            self.simulation_results['Pack']['Max SOC'][k] = np.max(z_old)
            self.simulation_results['Pack']['Avg SOC'][k] = np.mean(z_old)
            self.simulation_results['Pack']['Min temperature [C]'][k] = np.min(T)
            self.simulation_results['Pack']['Max temperature [C]'][k] = np.max(T)
            self.simulation_results['Pack']['Avg temperature [C]'][k] = np.mean(T)
            
            for i in range(Ns):
                for j in range(Np):
                    self.simulation_results[f'Cell {i}-{j}']['b0'][k] = b0[i,j]
                    self.simulation_results[f'Cell {i}-{j}']['Current [A]'][k] = ik[i,j]
                    self.simulation_results[f'Cell {i}-{j}']['Voltage [V]'][k] = vk[i,j]
                    self.simulation_results[f'Cell {i}-{j}']['SOC'][k] = z_old[i,j]
                    self.simulation_results[f'Cell {i}-{j}']['OCV [V]'][k] = ocv[i,j]
                    self.simulation_results[f'Cell {i}-{j}']['Temperature [C]'][k] = T[i,j]
                    for l in range(model_order):
                        self.simulation_results[f'Cell {i}-{j}'][f'a{l+1}'][k] = a[l,i,j]
                        self.simulation_results[f'Cell {i}-{j}'][f'b{l+1}'][k] = b[l,i,j]
        
        return None

class Vehicle():
    
    def __init__(self, vehicle_model, pack):
   
        
        self.vehicle_model = vehicle_model
        self.pack = pack
        
        self.mass_curb = self.vehicle_model['Mass [kg]'] + self.pack.mass
        self.mass_max = self.mass_curb + self.vehicle_model['Payload [kg]']
        self.mass_rotating = (
            (self.vehicle_model['Motor inertia [kg/m2]'] + self.vehicle_model['Gear inertia [kg/m2]']
             ) * self.vehicle_model['Gear ratio']**2 + self.vehicle_model['Wheel inertia [kg/m2]']*self.vehicle_model['Number of wheels']
            ) / self.vehicle_model['Wheel radius [m]']**2
        self.mass_equivalent = self.vehicle_model['Mass [kg]'] + self.mass_rotating
        self.max_speed = 2*np.pi*self.vehicle_model['Wheel radius [m]']*self.vehicle_model['Max RPM [RPM]']/(60*self.vehicle_model['Gear ratio']) # m/s
        self.max_power = 2*np.pi* self.vehicle_model['Max motor torque [Nm]'] * self.vehicle_model['Rated RPM [RPM]'] / 60 # W
        self.drivetrain_efficiency = self.pack.efficiency * self.vehicle_model['Inverter efficiency'] * self.vehicle_model['Motor efficiency'] * self.vehicle_model['Gear efficiency']
        
        self.initial_conditions = None
        self.simulation_results = None

    def initialize_simulation(self, 
                              time, 
                              speed_desired, 
                              sample_period):
        """
        Initializes the storage for the simulation results. This method is only
        intended to be used by the simulate_vehicle method and not on its own.

        Parameters
        ----------
        time : iterable
            Contains the time data for the trip of the vehicle.
        speed_desired : iterable
            Contains the speed data for the trip of the vehicle.
        sample_period : float
            Time between samples in seconds.

        Returns
        -------
        None.

        """
        self.simulation_results = dict()
        
        sim_len = len(time)
        
        self.simulation_results['Time [s]'] = np.array(time) # s
        self.simulation_results['Sample period [s]'] = sample_period # s
        self.simulation_results['Desired speed [m/s]'] = np.clip(np.array(speed_desired), 0, self.max_speed) # m/s
        self.simulation_results['Desired acceleration [m/s2]'] = np.zeros(sim_len) # m/s2
        self.simulation_results['Desired acceleration force [N]'] = np.zeros(sim_len) # N
        self.simulation_results['Aerodynamic force [N]'] = np.zeros(sim_len) # N
        self.simulation_results['Rolling grade force [N]'] = np.zeros(sim_len) # N
        self.simulation_results['Torque demand [Nm]'] = np.zeros(sim_len) # Nm
        self.simulation_results['Max torque [Nm]'] = np.zeros(sim_len) # Nm
        self.simulation_results['Limited regenerative torque [Nm]'] = np.zeros(sim_len) # Nm
        self.simulation_results['Limited torque [Nm]'] = np.zeros(sim_len) # Nm
        self.simulation_results['Motor torque [Nm]'] = np.zeros(sim_len) # Nm
        self.simulation_results['Limited power [W]'] = np.zeros(sim_len) # W
        self.simulation_results['Battery power demand [W]'] = np.zeros(sim_len) # W
        self.simulation_results['Actual acceleration force [N]l'] = np.zeros(sim_len) # N
        self.simulation_results['Actual acceleration [m/s2]'] = np.zeros(sim_len) # m/s2
        self.simulation_results['Motor speed [RPM]'] = np.zeros(sim_len) # RPM
        self.simulation_results['Motor power [W]'] = np.zeros(sim_len) # W
        self.simulation_results['Actual speed [m/s]'] = np.zeros(sim_len) # m/s

    def set_initial_conditions(self, speed=0, motor_speed=0):
        """
        Sets the initial conditions of the vehicle.

        Parameters
        ----------
        speed : float, optional
            Initial speed of the vehicle in m/s. The default is 0 m/s.
        motor_speed : float, optional
            Initial motor speed of the vehicle in RPM. The default is 0 RPM.

        Returns
        -------
        None.

        """
        self.initial_conditions = dict()
        
        self.initial_conditions['Speed [m/s]'] = speed # m/s
        self.initial_conditions['Motor speed [RPM]'] = motor_speed # RPM
    
    def simulate_vehicle(self, time, speed_desired, sample_period, off_times=None):
        """
        Simulates the vehicle given the desired speed profile.

        Parameters
        ----------
        time : iterable
            Contains the time data for the trip of the vehicle.
        speed_desired : iterable
            Contains the speed data for the trip of the vehicle.
        sample_period : float
            Time between samples in seconds.

        Returns
        -------
        None.

        """
        
        # Initialize temporary variables
        
        drag_coef = self.vehicle_model['Drag coefficient']
        front_area = self.vehicle_model['Frontal area [m2]']
        roll_coef = self.vehicle_model['Rolling coefficient']
        road_force = self.vehicle_model['Brake drag [N]']
        wheel_radius = self.vehicle_model['Wheel radius [m]']
        gear_ratio = self.vehicle_model['Gear ratio']
        L_max = self.vehicle_model['Max motor torque [Nm]']
        RPM_rated = self.vehicle_model['Rated RPM [RPM]']
        RPM_max = self.vehicle_model['Max RPM [RPM]']
        regen_torque = self.vehicle_model['Fractional regen torque limit']
        overhead_power = self.vehicle_model['Overhead power [W]']
        
        air_density = 1.225 # kg/m3
        G = 9.81 # m/s2
        
        self.initialize_simulation(time, speed_desired, sample_period)
        
        if not self.initial_conditions:
            # If initial conditions have not been set by the user
            self.set_initial_conditions(speed=speed_desired[0])
        
        sim_len = len(self.simulation_results['Time [s]'])
        
        self.simulation_results['Actual speed [m/s]'][-1] = self.initial_conditions['Speed [m/s]']
        self.simulation_results['Motor speed [RPM]'][-1] = self.initial_conditions['Motor speed [RPM]']
        
        for i in range(sim_len):
            
            self.simulation_results['Desired acceleration [m/s2]'][i] = (
                self.simulation_results['Desired speed [m/s]'][i] - self.simulation_results['Actual speed [m/s]'][i-1]
                )/sample_period
            self.simulation_results['Desired acceleration force [N]'][i] = self.mass_equivalent * self.simulation_results['Desired acceleration [m/s2]'][i]
            self.simulation_results['Aerodynamic force [N]'][i] = 0.5*air_density*drag_coef*front_area*(self.simulation_results['Actual speed [m/s]'][i-1])**2
            
            self.simulation_results['Rolling grade force [N]'][i] = 0
            
            if self.simulation_results['Actual speed [m/s]'][i-1] > 0:
                self.simulation_results['Rolling grade force [N]'][i] += roll_coef*self.mass_max*G
            
            self.simulation_results['Torque demand [Nm]'][i] = (self.simulation_results['Desired acceleration force [N]'][i] + 
                                                           self.simulation_results['Aerodynamic force [N]'][i] + 
                                                           self.simulation_results['Rolling grade force [N]'][i] +
                                                           road_force)*wheel_radius/gear_ratio
            
            self.simulation_results['Max torque [Nm]'][i] = L_max
            if self.simulation_results['Motor speed [RPM]'][i-1] >= RPM_rated:
                self.simulation_results['Max torque [Nm]'][i] = self.simulation_results['Max torque [Nm]'][i]*RPM_rated/self.simulation_results['Motor speed [RPM]'][i-1]
            
            self.simulation_results['Limited regenerative torque [Nm]'][i] = min(self.simulation_results['Max torque [Nm]'][i], regen_torque*L_max)
            self.simulation_results['Limited torque [Nm]'][i] = min(self.simulation_results['Torque demand [Nm]'][i], self.simulation_results['Max torque [Nm]'][i])
            
            if self.simulation_results['Limited torque [Nm]'][i] > 0:
                self.simulation_results['Motor torque [Nm]'][i] = self.simulation_results['Limited torque [Nm]'][i]
            else:
                self.simulation_results['Motor torque [Nm]'][i] = max(-self.simulation_results['Limited regenerative torque [Nm]'][i], self.simulation_results['Limited torque [Nm]'][i])
            
            self.simulation_results['Actual acceleration force [N]l'][i] = (
                self.simulation_results['Limited torque [Nm]'][i] * gear_ratio / 
                wheel_radius - self.simulation_results['Aerodynamic force [N]'][i] - 
                self.simulation_results['Rolling grade force [N]'][i] - road_force)
            
            self.simulation_results['Actual acceleration [m/s2]'][i] = self.simulation_results['Actual acceleration force [N]l'][i]/self.mass_equivalent
            self.simulation_results['Motor speed [RPM]'][i] = min(RPM_max,
                                                            gear_ratio*(
                                                            self.simulation_results['Actual speed [m/s]'][i-1] + 
                                                            self.simulation_results['Actual acceleration [m/s2]'][i]*sample_period)*60/(
                                                                2*np.pi*wheel_radius))
                                                                       
            self.simulation_results['Actual speed [m/s]'][i] = self.simulation_results['Motor speed [RPM]'][i]*2*np.pi*wheel_radius/(60*gear_ratio)
            
            if self.simulation_results['Limited torque [Nm]'][i] > 0:
                self.simulation_results['Motor power [W]'][i] = self.simulation_results['Limited torque [Nm]'][i]
            else:
                self.simulation_results['Motor power [W]'][i] = max(self.simulation_results['Limited torque [Nm]'][i],
                                                             -self.simulation_results['Limited regenerative torque [Nm]'][i])
            
            self.simulation_results['Motor power [W]'][i] = self.simulation_results['Motor power [W]'][i]*2*np.pi/60*(
                self.simulation_results['Motor speed [RPM]'][i-1] + self.simulation_results['Motor speed [RPM]'][i])/2
            
            self.simulation_results['Limited power [W]'][i] = max(-self.max_power, 
                                                         min(self.max_power, 
                                                             self.simulation_results['Motor power [W]'][i]))
            
            self.simulation_results['Battery power demand [W]'][i] = overhead_power
            
            if self.simulation_results['Limited power [W]'][i] > 0:
                self.simulation_results['Battery power demand [W]'][i] += self.simulation_results['Limited power [W]'][i]/self.drivetrain_efficiency
            else:
                self.simulation_results['Battery power demand [W]'][i] += self.simulation_results['Limited power [W]'][i]*self.drivetrain_efficiency

    def simulate_battery_pack(self,
                              initial_soc : np.ndarray | float | int = 0.8,
                              inital_temp : np.ndarray | float | int = 25,
                              initial_rc_current : np.ndarray | float | int = 0):
        """
        Wrapper which simulates the battery pack using the generated power 
        demand from the vehicle simulation.

        Parameters
        ----------
        initial_soc : np.array | float | int, optional
            Initial SOC of the cells in the pack, either as a Ns x Np numpy 
            array or as a single number (same across all cells). The default 
            is 0.8.
        inital_temp : np.array | float | int, optional
            Initial temperature of the cells in the pack in Celsius, either as 
            a Ns x Np numpy array or as a single number (same across all cells). 
            The default is 25 C.
        initial_rc_current : np.array | float | int, optional
            Initial diffusion current of the cells in the pack in Ampere, either 
            as a Ns x Np numpy array or as a single number (same across all 
            cells). Only used if the cell model is an Equivalent Circuit Model
            The default is 0 A (cells are at rest).

        Returns
        -------
        None.

        """
        
        if self.simulation_results is None:
            raise RuntimeError("Please run the 'simulate_vehicle' method before simulating the battery pack.")
        
        self.pack.simulate_pack(self.simulation_results['Battery power demand [W]'], 
                                self.simulation_results['Sample period [s]'],
                                initial_soc,
                                inital_temp,
                                initial_rc_current)
    
if __name__ == '__main__':
    pass
    
    