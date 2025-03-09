import pickle
import time
import os
import numpy as np
import pandas as pd
import libsumo as ls

from tqdm import tqdm
from multiprocessing import Pool

# =============================================================================
# Traffic simulation functions
# =============================================================================

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

def get_cell_currents_voltages(vf, r0, desired_power, cells_are_identical, Ns, Np):
    
    if cells_are_identical:
        rT = r0/Np # Thevenin eq. resistance per module
        vT = vf # Thevenin eq. voltage per module
        rT_pack = Ns*rT # Thevenin eq. resistance for whole pack
        vT_pack = Ns*vT # Thevenin eq. voltage for whole pack
        
        I = (vT_pack-np.sqrt(vT_pack**2-4*rT_pack*desired_power*1000))/(2*rT_pack) # Find necessary current for the desired power
        V = vT_pack - rT_pack*I
        
        vk = V/Ns # PCM terminal voltages
        ik = (vf-vk)/r0 # Individual cell currents
        
        vk = vf - r0*ik # Get individual cell voltages
    
    else:
        rT = 1/np.sum(1/r0,axis=1) # Thevenin eq. resistance per module
        vT = np.sum(vf/r0,axis=1)*rT # Thevenin eq. voltage per module
        rT_pack = np.sum(rT) # Thevenin eq. resistance for whole pack
        vT_pack = np.sum(vT) # Thevenin eq. voltage for whole pack
        
        I = (vT_pack-np.sqrt(vT_pack**2-4*rT_pack*desired_power*1000))/(2*rT_pack) # Find necessary current for the desired power
        V = vT_pack - rT_pack*I
        
        vk = (np.sum(vf/r0,axis=1)-I)/np.sum(1/r0,axis=1) # PCM terminal voltages
        vk = np.tile(vk, (Np,1)).T
        ik = (vf-vk)/r0 # Individual cell currents
        
        vk = vf - r0*ik # Get individual cell voltages

    return ik, vk, I, V

class Traffic():
    """
    Class used to define and run the traffic simulation. The main method of 
    this class is 'simulate_traffic'.
    """
    
    def __init__(self, config_path, output_dir='simulated_trip_files', duration=24, 
                 time_step=1, record_position=False, record_lane=False, 
                 pbar=True, lite_mode_ratio=None, random_state=None, 
                 remove_split_trip_files=True):
        """
        Initializes the Traffic class used for simulating the vehicle traffic. 

        Parameters
        ----------
        config_path : str
            The path to the SUMO configuration file for the scenario to be 
            simulated.
        output_dir : str, optional
            directory to store the final simulated trip files. By default, the
            trip files are stored in a directory named 'simulated_trip_files'.
        duration : int, optional
            Duration of simulation in hours. The default is 24 hours.
        time_step : float, optional
            Length of time between samples in the simulation in seconds. The
            default is 1 second between each sample. The minimum value of
            'time_step' is 0.1 seconds.
        record_position : bool, optional
            Records the x and y position in the network of each vehicle in the 
            simulation if True. Enabling this will increase file sizes. 
            The default is False.
        record_lane : bool, optional
            Records the ID of the current lane in the network of each vehicle 
            in the simulation if True. Enabling this will increase file sizes. 
            The default is False.
        pbar : bool, optional
            Displays a progress bar during the simulation if True. The default 
            is True.
        lite_mode_ratio : float, optional
            Can be set as a number between 0 and 1 which gives the ratio of trips 
            to process in the end, e.g. if 'lite_mode_ratio' is 0.1, only 10%
            of the trips are processed. The trips to process are randomnly 
            selected. If None, then all trips are processed, If 0, then no trips
            are processed and will have to processed manually by the user by 
            calling the 'process_simulation_data' method. NOTE: all trips 
            still need to be simulated, this variable only affects the processing
            of the simulated trips after simulation. The default is None.
        random_state : int or str, optional
            Sets the seed for the randomizer used to shuffle the order of the 
            trips to process in 'process_trips'. If None, then the randomizer is 
            initialized without setting a seed. If 'off', then the order to
            process the trips is not randomized. Note: if random_state is 'off',
            then it will switch to None if lite_mode_ratio is not None. The 
            default is None.
        remove_split_trip_files : bool, optional
            Removes the intermediate trip files generated after simulation
            and before the final processing if True. These files are mainly 
            used to leverage disk storage over memory storage and can easily be 
            removed after final processing. It is a good idea to turn this off
            if you want use 'process simulation data' multiple times with
            different values of 'lite_mode_ratio' or 'lite_mode_seed' and do
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
        self.record_lane = record_lane
        self.pbar = pbar
        self.lite_mode_ratio = lite_mode_ratio
        
        if self.lite_mode_ratio == 1:
            self.lite_mode_ratio = None
        
        if self.lite_mode_ratio is not None:
            if (self.lite_mode_ratio > 1) or (self.lite_mode_ratio < 0):
                raise ValueError("Please provide 'lite_mode_ratio' as a number between 0 (inclusive) and 1 (inclusive)")
        
        if (self.lite_mode_ratio is None) and (random_state == 'off'):
            # We need randomness to shuffle the trips
            self.random_state = None
        
        self.random_state = random_state
        self.remove_split_trip_files = remove_split_trip_files
    
    def update_vehicle_data(self, veh_id, data, step):
        
        if veh_id not in data.keys():
            # Initialize entry for vehicle
            data[veh_id] = dict()
            data[veh_id]['Time [s]'] = [] # Current time in simulation [s]
            data[veh_id]['Speed [m/s]'] = [] # Speed [m/s]
            
            if self.record_position:
                data[veh_id]['x [m]'] = [] # x position
                data[veh_id]['y [m]'] = [] # y position
            
            if self.record_lane:
                data[veh_id]['Lane ID'] = [] # Lane id
        
        data[veh_id]['Time [s]'].append(step*self.time_step)
        
        try:
            data[veh_id]['Speed [m/s]'].append(ls.vehicle.getSpeed(veh_id))
           
            if self.record_position:
                pos = ls.vehicle.getPosition(veh_id)
                data[veh_id]['x [m]'].append(pos[0])
                data[veh_id]['y [m]'].append(pos[1])
            
            if self.record_lane:
                data[veh_id]['Lane ID'].append(ls.vehicle.getLaneID(veh_id))
            
        except ls.TraCIException as error:
            
            if 'veh' in veh_id:
                data[veh_id]['Speed [m/s]'].append(ls.vehicle.getSpeed(veh_id[3:])) # Remove 'veh' indicator since SUMO does not expect that
               
                if self.record_position: 
                    pos = ls.vehicle.getPosition(veh_id[3:])
                    data[veh_id]['x [m]'].append(pos[0])
                    data[veh_id]['y [m]'].append(pos[1])
        
                if self.record_lane:
                    data[veh_id]['Lane ID'].append(ls.vehicle.getLaneID(veh_id[3:]))
            else:
                raise error
                
    
    def process_vehicle_data(self, veh_id):
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
        
        veh_files = [file for file in os.listdir('simulated_trips_split') if veh_id+'_' in file]
        veh_files.sort()
        
        with open(f'simulated_trips_split/{veh_files[0]}', 'rb') as file:
            veh_dict = pickle.load(file)
        
        for veh_file in veh_files[1:]:
            with open(f'simulated_trips_split/{veh_file}', 'rb') as file:
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
    
    def process_simulation_data(self):
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
        
        if self.output_dir not in os.listdir():
            print(f"\nMaking '{self.output_dir}'")
            os.mkdir(self.output_dir)
        else:
            print(f"\nPurging '{self.output_dir}'")
            
            dir_length = len((folder:=os.listdir(self.output_dir)))
            pbar = tqdm(total=dir_length, position=0, leave=True)
            if dir_length != 0:
                for file in folder:
                    os.remove(f'{self.output_dir}/{file}')
                    pbar.update()
        
        print('\nProcessing simulation data')
        
        veh_ids = list({file.split('_')[0] for file in os.listdir('simulated_trips_split')})
        
        if self.random_state != 'off':
            if self.random_state is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(self.random_state)
        
            rng.shuffle(veh_ids)
        
        if self.lite_mode_ratio is not None:
            veh_ids = veh_ids[:int(np.ceil(len(veh_ids)*0.3))]
        
        pool = Pool()
        
        if self.pbar is not None:
            pbar=tqdm(total=len(veh_ids), position=0, leave=True)
            for _ in pool.imap_unordered(self.process_vehicle_data, veh_ids):
                pbar.update()
    
        else:
            pool.imap_unordered(self.process_vehicle_data, veh_ids)
        
        pool.close()
        
        return None
    
    def simulate_traffic(self):
        """
        Simulates the traffic from a SUMO config file and tracks the time,
        position, speed, and ID of the current lane for each vehicle. The 
        simulation saves vehicle data every hour in order to save on memory 
        bandwidth.
    
        Returns
        -------
        None.
    
        """
        
        if 'simulated_trips_split' not in os.listdir():
            print("\nMaking 'simulated_trips_split'")
            os.mkdir('simulated_trips_split')
    
        else:
            print("\nPurging 'simulated_trips_split'")
            
            dir_length = len((folder:=os.listdir('simulated_trips_split')))
            pbar = tqdm(total=dir_length, position=0, leave=True)
            if dir_length != 0:
                for file in folder:
                    os.remove(f'simulated_trips_split/{file}')
                    pbar.update()
        
        print('\nStarting simulation')
        
        time_start = time.time()
        
        data = dict() # Initialize simulation data storage
        
        # Initialize simulation
        ls.start(["sumo", "-c", self.config_path, "--step-length", str(self.time_step)])
        print(f'\nStarted simulation with timedelta: {ls.simulation.getDeltaT()}s')
        n_steps = self.duration*3600*(1/self.time_step)
        
        if self.pbar:
            pbar = tqdm(total=n_steps, position=0, leave=True) # Define progress bar
        
        # Run simulation
        step=0
        while step < n_steps:
                
            ls.simulationStep()
            veh_list = ls.vehicle.getIDList() # Get vehicles on the road
            veh_list = ['veh' + veh_id if 'veh' not in veh_id else veh_id for veh_id in veh_list] # Sometimes the 'veh' indicator is omitted (eg. in the Berlin scenario) and needs to be put back in
            
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
                    
                    with open(f'simulated_trips_split/{veh_id}_{timeslot_index}.pickle', 'wb') as file:
                        pickle.dump(data[veh_id], file)
                
                data = dict() # Reset data storage
            
            if self.pbar:
                pbar.update()
        
        ls.close()
        
        if self.pbar:
            pbar.close()
        
        print(f'\n Finished simulation in {time.time()-time_start:.2f} seconds!')
        
        if self.lite_mode_ratio != 0: 
            self.process_simulation_data()
        
            if self.remove_split_trip_files:
                
                print("\nRemoving 'simulated_trips_split'")
                
                dir_length = len((folder:=os.listdir('simulated_trips_split')))
                pbar = tqdm(total=dir_length, position=0, leave=True)
                if dir_length != 0:
                    for file in folder:
                        os.remove(f'simulated_trips_split/{file}')
                        pbar.update()
                
                os.rmdir('simulated_trips_split')
        
        return None

# =============================================================================
# Vehicle definitions
# =============================================================================

class Pack():
    """
    Class used to define a battery pack comprising of one or more modules.
    

    Attributes
    ----------
    n_series : int
        Number of cells in series per module
    n_parallel : int
        Number of cells in parallel per module.
    n_modules : int
        Number of modules in the pack.
    n_cells : int
        Total number of cells in the pack.
    cell_mass : float
        Mass of each cell in kilograms.
    module_overhead : float
        Additional weight in the module for packaging, electronics etc. 
        expressed as a percentage (between 0 and 1) of total mass of the cells.
    pack_overhead : float
        Additional weight in the pack for packaging, electronics etc.
        expressed as a percentage (between 0 and 1) of total mass of the 
        modules.
    mass : float
        Total mass of the pack in kilograms.
    efficiency : float
        Total energy efficiency of the pack expressed as a percentage between
        0 and 1.
    initial_conditions : dict
        Initial conditions used in the simulation of the battery pack and set 
        by the set_initial_conditions method. If the initial conditions have 
        not been set yet, then the initial_conditions attribute is a NoneType 
        object.
    simulation_results : dict
        Contains relevant results after simualting the battery pack using the
        simulate_pack method. If the simulation has not been performed yet,
        then the simulation_results attribute is a NoneType object.
    
    """
    def __init__(self, pack_model, cell_model, temperature_model=None):
        """
        Initializes the Pack class.

        Parameters
        ----------
        n_series : int
            Number of cells in series per module
        n_parallel : int
            Number of cells in parallel per module. Every module is assumed to have
            the same number of cells in parallel.
        n_modules : int
            Number of modules in the pack.
        cell_mass : float
            Mass of each cell in kilograms.
        module_overhead : float
            Additional weight in the module for packaging, electronics etc. 
            expressed as a percentage (between 0 and 1) of total mass of the cells.
        pack_overhead : float
            Additional weight in the pack for packaging, electronics etc.
            expressed as a percentage (between 0 and 1) of total mass of the 
            modules.
        efficiency : float
            Total energy efficiency of the pack expressed as a percentage between
            0 and 1.

        Returns
        -------
        None.

        """
        
        self.pack_model = pack_model
        
        self.cell_model = cell_model
        self.cell_model_is_array = isinstance(self.cell_model, np.ndarray)
        
        if (not self.cell_model_is_array) and (not isinstance(self.cell_model, dict)):
            raise ValueError('Please give the cell model as either a dictionary or an array of dictionaries. See tracksim.cell_models for a selection of compatible models.')
        
        self.cell_model_is_dynamic = False
        
        
        if 'ECM' in self.cell_model['Model Type']:
            if self.cell_model_is_array:
                self.cell_model_is_dynamic = callable(self.cell_model[0,0]['R0'])
            
            else:
                self.cell_model_is_dynamic = callable(self.cell_model['R0'])
        
        elif 'ARX' in self.cell_model['Model Type']:
            if self.cell_model_is_array:
                self.cell_model_is_dynamic = callable(self.cell_model[0,0]['b0'])
            
            else:
                self.cell_model_is_dynamic = callable(self.cell_model['b0'])
        else:
            raise KeyError('Please check model type')

        self.temperature_model = temperature_model
        self.temperature_model_is_array = isinstance(self.temperature_model, np.ndarray)
        
        if (temperature_model is not None) and (not self.temperature_model_is_array) and (not isinstance(self.temperature_model, dict)):
            raise ValueError('Please give the temperature model as either a dictionary or an array of dictionaries. See tracksim.temperature_models for a selection of compatible models.')
        
        self.Ns = self.pack_model['No. Cells Series']*self.pack_model['No. Modules']
        self.Np = self.pack_model['No. Cells Parallel']
        
        self.n_cells = self.Ns*self.Np
        
        if self.cell_model_is_array:
            
            cell_masses = np.zeros(shape=(self.Ns,self.Np))
            cell_nominal_charge_capacities = np.zeros(shape=(self.Ns,self.Np))
            cell_nominal_voltages = np.zeros(shape=(self.Ns,self.Np))
            
            for i in range(self.Ns):
                for j in range(self.Np):
                    cell_masses[i,j] = self.cell_model[i,j]['Mass [kg]']
                    cell_nominal_charge_capacities[i,j] = self.cell_model[i,j]['Nominal Capacity [Ah]']
                    cell_nominal_voltages[i,j] = self.cell_model[i,j]['Nominal Voltage [V]']
            
            self.mass = np.sum(cell_masses)*1/(1-self.pack_model['Battery Module Overhead'])*1/(1-self.pack_model['Battery Pack Overhead'])
            self.nominal_charge_capacity = np.min(np.sum(cell_nominal_charge_capacities, axis=1))
            self.nominal_energy_capacity = np.sum(np.mean(cell_nominal_voltages, axis=1))*self.nominal_charge_capacity
            
        else:
            self.mass = self.n_cells*self.cell_model['Mass [kg]']*1/(1-self.pack_model['Battery Module Overhead'])*1/(1-self.pack_model['Battery Pack Overhead'])
            self.nominal_charge_capacity = self.Np * self.cell_model['Nominal Capacity [Ah]']
            self.nominal_energy_capacity = self.nominal_charge_capacity*self.Ns*self.cell_model['Nominal Voltage [V]'] # Wh
        self.efficiency = self.pack_model['Battery Pack Efficiency']
        
        self.initial_conditions = None
        self.simulation_results = None
        self.identical_cells = False
        self.charge_current_is_positive = self.cell_model['Positive Charge Current']
    
    def set_initial_conditions_ECM(self, soc=0.8, irc=0, cell_temp=25, coolant_temp=25):
        """
        Sets the initial conditions for the battery pack before the simulation.

        Parameters
        ----------
        soc : float or ndarray of floats, optional
            Contains the initial SOC for each cell expressed as a percentage 
            (between 0 and 1) stored in an (n_series*n_modules) X n_parallel 
            ndarray. If a float is given instead, each cell is initialized with 
            the same SOC. By default, all cells are initialized with an SOC 
            of 0.8.

        irc : int, float or ndarray of floats, optional
            Contains the initial diffusion current for each cell in A  stored 
            in an n_rc_pairs X (n_series*n_modules) X n_parallel ndarray. If an 
            int float is given instead, each cell is initialized with the same 
            diffusion current. By default, all cells are initialized with a 
            diffusion current of 0 (relaxed cell).
        
        cell_temp : int, float or ndarray of floats, optional
            Contains the initial temperature for each cell expressed in Celsius 
            stored in an (n_series*n_modules) X n_parallel ndarray. If an int 
            or float is given instead, each cell is initialized with the same 
            SOC. By  default, all cells are initialized with atemperature of 
            25 deg C.
        
        coolant_temp : int or float, optional
            Temperature of the coolant to be applied to each cell in Celsius. 
            By default, the coolant is set at 25 deg Celsius.
        
        Returns
        -------
        None.

        """
        
        Ns = self.Ns
        Np = self.Np
        
        self.initial_conditions = dict()
        
        if isinstance(soc, (int, float)):
            self.initial_conditions['SOC'] = np.ones(shape=(Ns,Np))*soc
        elif isinstance(soc, np.ndarray):
            self.initial_conditions['SOC'] = soc
        else:
            raise ValueError("Please provide 'soc' as either an int, float, or ndarray.")
        
        if isinstance(irc, (int, float)):
            self.initial_conditions['rc_current'] = np.ones(shape=(Ns,Np))*irc
        elif isinstance(soc, np.ndarray):
            self.initial_conditions['rc_current'] = irc
        else:
            raise ValueError("Please provide 'irc' as either an int, float, or ndarray.")
        
        if isinstance(cell_temp, (int, float)):
            self.initial_conditions['cell_temp'] = np.ones(shape=(Ns,Np))*cell_temp
        elif isinstance(cell_temp, np.ndarray):
            self.initial_conditions['cell_temp'] = cell_temp
        else:
            raise ValueError("Please provide 'cell_temp' as either an int, float or ndarray.")
        
        if isinstance(coolant_temp, (int, float)):
            self.initial_conditions['coolant_temp'] = np.ones(shape=(Ns,Np))*coolant_temp
        else:
            raise ValueError("Please provide 'coolant_temp' as either an int or a float.")
    
    def simulate_pack(self, desired_power, time_delta):
        
        if 'ECM' in self.cell_model['Model Type']:
            self.simulate_pack_ECM(desired_power, time_delta)
            
        elif 'ARX' in self.cell_model['Model Type']:
            self.simulate_pack_ARX(desired_power, time_delta)
            
        else:
            raise KeyError('Please check model type')
    
    def initialize_simulation_ECM(self, sim_len, time_delta, num_rc_pairs, cells_are_identical):
        """
        Initializes the storage for the simulation results. This method is only
        intended to be used by the simulate_pack_ECM method and not on its own.

        Parameters
        ----------
        sim_len : int
            Length of simulation in samples.
        time_delta : float
            Length of time between samples in seconds.

        Returns
        -------
        None.

        """
        
        if cells_are_identical:
            Ns = 1
            Np = 1
        else:
            Ns = self.Ns
            Np = self.Np
        
        self.simulation_results = dict()
        
        # Set up storage
        self.simulation_results['time'] = np.arange(sim_len)*time_delta # s
        self.simulation_results['pack'] = dict()
        self.simulation_results['pack']['current'] = np.zeros(sim_len) # A
        self.simulation_results['pack']['voltage'] = np.zeros(sim_len) # V
        self.simulation_results['pack']['min_soc'] = np.zeros(sim_len) # Minimum SOC of all cells
        self.simulation_results['pack']['max_soc'] = np.zeros(sim_len) # Maximum SOC of all cells
        self.simulation_results['pack']['avg_soc'] = np.zeros(sim_len) # Average SOC of all cells
        self.simulation_results['pack']['min_temp'] = np.zeros(sim_len) # Minimum temperature of all cells
        self.simulation_results['pack']['max_temp'] = np.zeros(sim_len) # Maximum temperature of all cells
        self.simulation_results['pack']['avg_temp'] = np.zeros(sim_len) # Average temperature of all cells
        for i in range(Ns):
            for j in range(Np):
                self.simulation_results[f'cell_{i}-{j}'] = dict()
                self.simulation_results[f'cell_{i}-{j}']['R0'] = np.zeros(sim_len) # Ohm
                self.simulation_results[f'cell_{i}-{j}']['current'] = np.zeros(sim_len) # A
                self.simulation_results[f'cell_{i}-{j}']['voltage'] = np.zeros(sim_len) # V
                self.simulation_results[f'cell_{i}-{j}']['soc'] = np.zeros(sim_len)
                self.simulation_results[f'cell_{i}-{j}']['temp'] = np.zeros(sim_len) # Deg C
                for l in range(num_rc_pairs):
                    self.simulation_results[f'cell_{i}-{j}'][f'current_rc{l+1}'] = np.zeros(sim_len) # A
                    self.simulation_results[f'cell_{i}-{j}'][f'R{l+1}'] = np.zeros(sim_len) # Ohm
                    self.simulation_results[f'cell_{i}-{j}'][f'C{l+1}'] = np.zeros(sim_len) # F
            
    def simulate_pack_ECM(self, desired_power, time_delta):
        """
        Simulates the battery pack under the given battery power profile.

        Parameters
        ----------
        desired_power : iterable
            Profile of the demanded power in kW.
        time_delta : float
            Length of time between samples in seconds.

        Raises
        ------
        AttributeError
            Raised if the set_initial_conditions method has not been run before
            starting the simulation.

        Returns
        -------
        None.

        """
        
        if self.initial_conditions is None:
            print('No initial conditions set. Initializing with default values.')
            self.set_initial_conditions_ECM()
        
        # Initialize temporary variables
        
        Ns = self.Ns
        Np = self.Np
        
        if self.cell_model_is_array:
            num_rc_pairs = self.cell_model[0,0]['No. RC Pairs']
        else:
            num_rc_pairs = self.cell_model['No. RC Pairs']
            
        sim_len = len(desired_power)
        
        # Initialize cell states
        z = self.initial_conditions['SOC']
        irc = self.initial_conditions['rc_current']
        T = self.initial_conditions['cell_temp']
        Tf = self.initial_conditions['coolant_temp']
        
        cells_are_identical = False
        if not self.cell_model_is_array:
            
            if np.allclose(z, z[0,0]) and np.allclose(irc, irc[0,0]) and np.allclose(T, T[0,0]) and np.allclose(Tf, Tf[0,0]):
                # If the cells have the same parameters and the same initial 
                # conditions, then only simulate one cell
                cells_are_identical = True
                Ns = 1
                Np = 1
                z = z[0,0].reshape(1,1)
                irc = irc[0,0].reshape(1,1)
                T = T[0,0].reshape(1,1)
                Tf = Tf[0,0].reshape(1,1)
        
        # Initialize cell parameters
        
        q = np.zeros(shape=(Ns,Np)) # Ns x Np
        eta = np.zeros(shape=(Ns,Np)) # Ns x Np
        r0 = np.zeros(shape=(Ns,Np)) # Ns x Np
        r = np.zeros(shape=(num_rc_pairs, Ns, Np)) # num_rc_pairs x Ns x Np
        c = np.zeros(shape=(num_rc_pairs, Ns, Np)) # num_rc_pairs x Ns x Np
        rc = np.zeros(shape=(num_rc_pairs, Ns, Np)) # num_rc_pairs x Ns x Np
        
        if self.cell_model_is_array:
            
            if self.cell_model_is_dynamic:
                
                for i in range(Ns):
                    for j in range(Np):
                        
                        q[i,j] = self.cell_model[i,j]['Capacity [Ah]']
                        eta[i,j] = self.cell_model[i,j]['Coulombic Efficiency']
                        r0[i,j] = self.cell_model[i,j]['R0'](z[i,j], T[i,j])
                        r0[i,j] += 2*self.cell_model[i,j]['Tab Resistance [Ohm]']
                        
                        for l in range(num_rc_pairs):
                            r[l,i,j] = self.cell_model[i,j][f'R{l+1}'](z[i,j], T[i,j])
                            c[l,i,j] = self.cell_model[i,j][f'C{l+1}'](z[i,j], T[i,j])
                            rc[l,i,j] = np.exp(-time_delta/np.abs(r[l,i,j]*c[l,i,j]))
            
            else:
                
                for i in range(Ns):
                    for j in range(Np):
                        
                        q[i,j] = self.cell_model[i,j]['Capacity [Ah]']
                        eta[i,j] = self.cell_model[i,j]['Coulombic Efficiency']
                        r0[i,j] = self.cell_model[i,j]['R0']
                        r0[i,j] += 2*self.cell_model['Tab Resistance [Ohm]']
                        
                        for l in range(num_rc_pairs):
                            r[l,i,j] = self.cell_model[i,j][f'R{l+1}']
                            c[l,i,j] = self.cell_model[i,j][f'C{l+1}']
                            rc[l,i,j] = np.exp(-time_delta/np.abs(r[l,i,j]*c[l,i,j]))
                
        else:
            
            q = np.ones(shape=(Ns,Np))*self.cell_model['Capacity [Ah]']
            eta = np.ones(shape=(Ns,Np))*self.cell_model['Coulombic Efficiency']
            
            if self.cell_model_is_dynamic:
                
                r0 = self.cell_model['R0'](z, T)
                r0 += 2*np.ones(shape=(Ns,Np))*self.cell_model['Tab Resistance [Ohm]'] # Add tab resistance for each cell
                
                r = np.zeros(shape=(num_rc_pairs, Ns, Np))
                c = np.zeros(shape=(num_rc_pairs, Ns, Np))
                rc = np.zeros(shape=(num_rc_pairs, Ns, Np))
                
                for l in range(num_rc_pairs):
                  r[l,:,:] = self.cell_model[f'R{l+1}'](z, T)
                  c[l,:,:] = self.cell_model[f'C{l+1}'](z, T)
                  rc[l,:,:] = np.exp(-time_delta/np.abs(r[l,:,:]*c[l,:,:]))
            
            else:
                
                r0 = np.ones(shape=(Ns,Np))*self.cell_model['R0']
                r0 += 2*np.ones(shape=(Ns,Np))*self.cell_model['Tab Resistance [Ohm]'] # Add tab resistance for each cell
                
                r = np.zeros(shape=(num_rc_pairs, Ns, Np))
                c = np.zeros(shape=(num_rc_pairs, Ns, Np))
                rc = np.zeros(shape=(num_rc_pairs, Ns, Np))
                
                for l in range(num_rc_pairs):
                  r[l,:,:] = np.ones(shape=(Ns,Np))*self.cell_model[f'R{l+1}']
                  c[l,:,:] = np.ones(shape=(Ns,Np))*self.cell_model[f'C{l+1}']
                  rc[l,:,:] = np.exp(-time_delta/np.abs(r[l,:,:]*c[l,:,:]))
        
        if self.temperature_model is not None:
            
            h = np.zeros(shape=(Ns,Np))
            A = np.zeros(shape=(Ns,Np))
            m = np.zeros(shape=(Ns,Np))
            Cp = np.zeros(shape=(Ns,Np))
            
            if self.temperature_model_is_array:
                
                if self.cell_model_is_array:
                
                    for i in range(Ns):
                        for j in range(Np):
                            h[i,j] = self.temperature_model[i,j]['Equivalent Convective Heat Transfer Coefficient [W/(m2K)]']
                            A[i,j] = self.cell_model[i,j]['Surface Area [m2]']
                            m[i,j] = self.cell_model[i,j]['Mass [kg]']*1000 # g
                            Cp[i,j] = self.temperature_model[i,j]['Specific Heat Capacity [J/(kgK)]']*1000 # J/gK
                else:
                    for i in range(Ns):
                        for j in range(Np):
                            h[i,j] = self.temperature_model[i,j]['Equivalent Convective Heat Transfer Coefficient [W/(m2K)]']
                            A[i,j] = self.cell_model['Surface Area [m2]']
                            m[i,j] = self.cell_model['Mass [kg]']*1000 # g
                            Cp[i,j] = self.temperature_model[i,j]['Specific Heat Capacity [J/(kgK)]']*1000 # J/gK
                        
            else:
                
                if self.cell_model_is_array:
                    for i in range(Ns):
                        for j in range(Np):
                            h[i,j] = self.temperature_model['Equivalent Convective Heat Transfer Coefficient [W/(m2K)]']
                            A[i,j] = self.cell_model[i,j]['Surface Area [m2]']
                            m[i,j] = self.cell_model[i,j]['Mass [kg]']*1000 # g
                            Cp[i,j] = self.temperature_model['Specific Heat Capacity [J/(kgK)]']*1000 # J/gK
                
                else:
                    h = np.ones(shape=(Ns,Np))*self.temperature_model['Equivalent Convective Heat Transfer Coefficient [W/(m2K)]']
                    A = np.ones(shape=(Ns,Np))*self.cell_model['Surface Area [m2]']
                    m = np.ones(shape=(Ns,Np))*self.cell_model['Mass [kg]']*1000 # g
                    Cp = np.ones(shape=(Ns,Np))*self.temperature_model['Specific Heat Capacity [J/(kgK)]']*1000 # J/gK
        
            e = np.exp((-h*A*time_delta)/m*Cp)
        
        self.initialize_simulation_ECM(len(desired_power), time_delta, num_rc_pairs, cells_are_identical) # Initialize storage
        
        for k in range(sim_len):
            
            if self.cell_model_is_array:
                v_cells = np.zeros(shape=(Ns,Np))
                for i in range(Ns):
                    for j in range(Np):
                        v_cells[i,j] = self.cell_model[i,j]['OCV'](z[i,j], T[i,j])
            else:
                v_cells = self.cell_model['OCV'](z,T) # Get OCV for each cell
            
            v_cells -= np.sum(r*irc, axis=0) # Add diffusion voltages
            
            if self.charge_current_is_positive:
                
                ik, vk, I, V = get_cell_currents_voltages(v_cells, -r0, -desired_power[k], cells_are_identical, self.Ns, self.Np)
                ik[ik>0] = ik[ik>0]*eta[ik>0] # Multiply by eta for cells where we are charging
                z += (time_delta/(3600*q))*ik # Update SOC
                irc = rc*irc + (1-rc)*np.tile(ik, (num_rc_pairs,1,1)) # Update RC resistor currents
                
            else:
                
                ik, vk, I, V = get_cell_currents_voltages(v_cells, r0, desired_power[k], cells_are_identical, self.Ns, self.Np)
                ik[ik<0] = ik[ik<0]*eta[ik<0] # Multiply by eta for cells where we are charging
                z -= (time_delta/(3600*q))*ik # Update SOC
                irc = rc*irc - (1-rc)*np.tile(ik, (num_rc_pairs,1,1)) # Update RC resistor currents
            
            # Update temperature
            if self.temperature_model is not None:
                
                if self.temperature_model_is_array:
                    
                    dOCV_dT = np.zeros(shape=(Ns,Np))

                    for i in range(Ns):
                        for j in range(Np):
                            dOCV_dT[i,j] = self.temperature_model[i,j]['Entropic Heat Coefficient'](z[i,j])
                
                else:
                    dOCV_dT = self.temperature_model['Entropic Heat Coefficient'](z)
                
                if self.charge_current_is_positive:
                    Qk = ik**2*r0 + ik*np.sum(irc*r, axis=0) + ik*(T+273.15)*dOCV_dT
                else:
                    Qk = ik**2*r0 - ik*np.sum(irc*r, axis=0) - ik*(T+273.15)*dOCV_dT
                
                T = e*(T+273.15) + (1-e)*(Qk/(h*A) + (Tf+273.15)) - 273.15
            
            # Update ECM parameters
            if self.cell_model_is_dynamic:
                
                if self.cell_model_is_array:

                    for i in range(Ns):
                        for j in range(Np):
                            r0[i,j] = self.cell_model[i,j]['R0'](z[i,j], T[i,j])
                            
                            for l in range(num_rc_pairs):
                                r[l,i,j] = self.cell_model[i,j][f'R{l+1}'](z[i,j], T[i,j])
                                c[l,i,j] = self.cell_model[i,j][f'C{l+1}'](z[i,j], T[i,j])
                    
                    for l in range(num_rc_pairs):
                        rc[l,:,:] = np.exp(-time_delta/np.abs(r[l,:,:]*c[l,:,:]))
                
                else:
                    r0 = self.cell_model['R0'](z, T)
                    for l in range(num_rc_pairs):
                        r[l,:,:] = self.cell_model[f'R{l+1}'](z, T)
                        c[l,:,:] = self.cell_model[f'C{l+1}'](z, T)
                        rc[l,:,:] = np.exp(-time_delta/np.abs(r[l,:,:]*c[l,:,:]))
            
            # Store measurements
            self.simulation_results['pack']['current'][k] = self.Np*ik # I
            self.simulation_results['pack']['voltage'][k] = self.Ns*vk #V 
            self.simulation_results['pack']['min_soc'][k] = np.min(z)
            self.simulation_results['pack']['max_soc'][k] = np.max(z)
            self.simulation_results['pack']['avg_soc'][k] = np.mean(z)
            self.simulation_results['pack']['min_temp'][k] = np.min(T)
            self.simulation_results['pack']['max_temp'][k] = np.max(T)
            self.simulation_results['pack']['avg_temp'][k] = np.mean(T)
            
            for i in range(Ns):
                for j in range(Np):
                    self.simulation_results[f'cell_{i}-{j}']['R0'][k] = r0[i,j]
                    self.simulation_results[f'cell_{i}-{j}']['current'][k] = ik[i,j]
                    self.simulation_results[f'cell_{i}-{j}']['voltage'][k] = vk[i, j]
                    self.simulation_results[f'cell_{i}-{j}']['soc'][k] = z[i,j]
                    self.simulation_results[f'cell_{i}-{j}']['temp'][k] = T[i,j]
                    for l in range(num_rc_pairs):
                        self.simulation_results[f'cell_{i}-{j}'][f'current_rc{l+1}'][k] = irc[l,i,j]
                        self.simulation_results[f'cell_{i}-{j}'][f'R{l+1}'][k] = r[l,i,j]
                        self.simulation_results[f'cell_{i}-{j}'][f'C{l+1}'][k] = c[l,i,j]
    
    def set_initial_conditions_ARX(self, soc=0.8, cell_temp=25, coolant_temp=25):
        """
        Sets the initial conditions for the battery pack before the simulation.

        Parameters
        ----------
        soc : float or ndarray of floats, optional
            Contains the initial SOC for each cell expressed as a percentage 
            (between 0 and 1) stored in an (n_series*n_modules) X n_parallel 
            ndarray. If a float is given instead, each cell is initialized with 
            the same SOC. By default, all cells are initialized with an SOC 
            of 0.8.
        
        cell_temp : int, float or ndarray of floats, optional
            Contains the initial temperature for each cell expressed in Celsius 
            stored in an (n_series*n_modules) X n_parallel ndarray. If an int 
            or float is given instead, each cell is initialized with the same 
            SOC. By  default, all cells are initialized with atemperature of 
            25 deg C.
        
        coolant_temp : int or float, optional
            Temperature of the coolant to be applied to each cell in Celsius. 
            By default, the coolant is set at 25 deg Celsius.
        
        Returns
        -------
        None.

        """
        
        Ns = self.Ns
        Np = self.Np
        
        self.initial_conditions = dict()
        
        if isinstance(soc, (int, float)):
            self.initial_conditions['SOC'] = np.ones(shape=(Ns,Np))*soc
        elif isinstance(soc, np.ndarray):
            self.initial_conditions['SOC'] = soc
        else:
            raise ValueError("Please provide 'soc' as either an int, float, or ndarray.")
        
        if isinstance(cell_temp, (int, float)):
            self.initial_conditions['cell_temp'] = np.ones(shape=(Ns,Np))*cell_temp
        elif isinstance(cell_temp, np.ndarray):
            self.initial_conditions['cell_temp'] = cell_temp
        else:
            raise ValueError("Please provide 'cell_temp' as either an int, float or ndarray.")
        
        if isinstance(coolant_temp, (int, float)):
            self.initial_conditions['coolant_temp'] = np.ones(shape=(Ns,Np))*coolant_temp
        else:
            raise ValueError("Please provide 'coolant_temp' as either an int or a float.")
    
    def initialize_simulation_ARX(self, sim_len, time_delta, model_order, cells_are_identical):
        """
        Initializes the storage for the simulation results. This method is only
        intended to be used by the simulate_pack_ARX method and not on its own.

        Parameters
        ----------
        sim_len : int
            Length of simulation in samples.
        time_delta : float
            Length of time between samples in seconds.

        Returns
        -------
        None.

        """
        
        if cells_are_identical:
            Ns = 1
            Np = 1
        else:
            Ns = self.Ns
            Np = self.Np
        
        self.simulation_results = dict()
        
        # Set up storage
        self.simulation_results['time'] = np.arange(sim_len)*time_delta # s
        self.simulation_results['pack'] = dict()
        self.simulation_results['pack']['current'] = np.zeros(sim_len) # A
        self.simulation_results['pack']['voltage'] = np.zeros(sim_len) # V
        self.simulation_results['pack']['min_soc'] = np.zeros(sim_len) # Minimum SOC of all cells
        self.simulation_results['pack']['max_soc'] = np.zeros(sim_len) # Maximum SOC of all cells
        self.simulation_results['pack']['avg_soc'] = np.zeros(sim_len) # Average SOC of all cells
        self.simulation_results['pack']['min_temp'] = np.zeros(sim_len) # Minimum temperature of all cells
        self.simulation_results['pack']['max_temp'] = np.zeros(sim_len) # Maximum temperature of all cells
        self.simulation_results['pack']['avg_temp'] = np.zeros(sim_len) # Average temperature of all cells
        for i in range(Ns):
            for j in range(Np):
                self.simulation_results[f'cell_{i}-{j}'] = dict()
                self.simulation_results[f'cell_{i}-{j}']['current'] = np.zeros(sim_len) # A
                self.simulation_results[f'cell_{i}-{j}']['voltage'] = np.zeros(sim_len) # V
                self.simulation_results[f'cell_{i}-{j}']['soc'] = np.zeros(sim_len)
                self.simulation_results[f'cell_{i}-{j}']['temp'] = np.zeros(sim_len) # Deg C
                self.simulation_results[f'cell_{i}-{j}']['b0'] = np.zeros(sim_len)
                for l in range(model_order):
                    self.simulation_results[f'cell_{i}-{j}'][f'a{l+1}'] = np.zeros(sim_len)
                    self.simulation_results[f'cell_{i}-{j}'][f'b{l+1}'] = np.zeros(sim_len)
            
    def simulate_pack_ARX(self, desired_power, time_delta):
        """
        Simulates the battery pack under the given battery power profile.

        Parameters
        ----------
        desired_power : iterable
            Profile of the demanded power in kW.
        time_delta : float
            Length of time between samples in seconds.

        Raises
        ------
        AttributeError
            Raised if the set_initial_conditions method has not been run before
            starting the simulation.

        Returns
        -------
        None.

        """
        
        if self.initial_conditions is None:
            print('No initial conditions set. Initializing with default values.')
            self.set_initial_conditions_ARX()
        
        # Initialize temporary variables
        
        Ns = self.Ns
        Np = self.Np
        
        if self.cell_model_is_array:
            model_order = self.cell_model[0,0]['Model Order']
        else:
            model_order = self.cell_model['Model Order']
        
        sim_len = len(desired_power)
        
        # Initialize cell states
        z = self.initial_conditions['SOC']
        T = self.initial_conditions['cell_temp'] # Celsius
        
        cells_are_identical = False
        if not self.cell_model_is_array:
            
            if np.allclose(z, z[0,0]) and np.allclose(T, T[0,0]):
                # If the cells have the same parameters and the same initial 
                # conditions, then only simulate one cell
                cells_are_identical = True
                Ns = 1
                Np = 1
                z = z[0,0].reshape(1,1)
                T = T[0,0].reshape(1,1)
        
        # Initialize cell parameters
        
        q = np.zeros(shape=(Ns,Np)) # Ns x Np
        eta = np.zeros(shape=(Ns,Np)) # Ns x Np
        b0 = np.zeros(shape=(Ns,Np)) # Ns x Np
        a = np.zeros(shape=(model_order, Ns, Np)) # model_order x Ns x Np
        b = np.zeros(shape=(model_order, Ns, Np)) # model_order x Ns x Np
        
        # Initialize matrices for storing history values
        
        v_hist = np.ones(shape=(model_order, Ns, Np))*self.cell_model['OCV'](z, T)
        ocv_hist = v_hist.copy()
        i_hist = np.zeros(shape=(model_order, Ns, Np))
        T_hist = np.ones(shape=(model_order, Ns, Np))*T
        z_hist = np.ones(shape=(model_order, Ns, Np))*z
        
        if self.cell_model_is_array:
            
            if self.cell_model_is_dynamic:
                
                for i in range(Ns):
                    for j in range(Np):
                        
                        q[i,j] = self.cell_model[i,j]['Capacity [Ah]']
                        eta[i,j] = self.cell_model[i,j]['Coulombic Efficiency']
                        b0[i,j] = self.cell_model[i,j]['b0'](z[i,j], T[i,j])
                        
                        for l in range(model_order):
                            a[l,i,j] = self.cell_model[i,j][f'a{l+1}'](z_hist[l,i,j], T_hist[l,i,j])
                            b[l,i,j] = self.cell_model[i,j][f'b{l+1}'](z_hist[l,i,j], T_hist[l,i,j])
            
            else:
                
                for i in range(Ns):
                    for j in range(Np):
                        
                        q[i,j] = self.cell_model[i,j]['Capacity [Ah]']
                        eta[i,j] = self.cell_model[i,j]['Coulombic Efficiency']
                        b0[i,j] = self.cell_model[i,j]['b0']
                        
                        for l in range(model_order):
                            a[l,i,j] = self.cell_model[i,j][f'a{l+1}']
                            b[l,i,j] = self.cell_model[i,j][f'b{l+1}']
                
        else:
            
            q = np.ones(shape=(Ns,Np))*self.cell_model['Capacity [Ah]']
            eta = np.ones(shape=(Ns,Np))*self.cell_model['Coulombic Efficiency']
            
            if self.cell_model_is_dynamic:
                
                b0 = self.cell_model['b0'](z, T)
                
                for l in range(model_order):
                    a[l,:,:] = self.cell_model[f'a{l+1}'](z_hist[l,:,:], T_hist[l,:,:])
                    b[l,:,:] = self.cell_model[f'b{l+1}'](z_hist[l,:,:], T_hist[l,:,:])
                  
            else:
                
                b0 = np.ones(shape=(Ns,Np))*self.cell_model['b0']
                
                for l in range(model_order):
                    a[l,:,:] = np.ones(shape=(Ns,Np))*self.cell_model[f'a{l+1}']
                    b[l,:,:] = np.ones(shape=(Ns,Np))*self.cell_model[f'b{l+1}']
        
        
        
        self.initialize_simulation_ARX(sim_len, time_delta, model_order, cells_are_identical) # Initialize storage
        
        for k in range(sim_len):
            
            # Get the current OCV
            
            if self.cell_model_is_array:
                ocv = np.zeros(shape=(Ns,Np))
                for i in range(Ns):
                    for j in range(Np):
                        ocv[i,j] = self.cell_model[i,j]['OCV'](z[i,j], T[i,j])
                        
            else:
                ocv = self.cell_model['OCV'](z,T) # Get OCV for each cell
            
            v_cells = np.sum(a*(v_hist-ocv_hist) + b*i_hist, axis=0) + ocv # Get Vf
            
            if self.charge_current_is_positive:
                
                ik, vk, I, V = get_cell_currents_voltages(v_cells, -b0, -desired_power[k], cells_are_identical, self.Ns, self.Np)
                ik[ik>0] = ik[ik>0]*eta[ik>0] # Multiply by eta for cells where we are charging
                z += (time_delta/(3600*q))*ik # Update SOC
            
            else:
                
                ik, vk, I, V = get_cell_currents_voltages(v_cells, -b0, desired_power[k], cells_are_identical, self.Ns, self.Np)
                ik[ik<0] = ik[ik<0]*eta[ik<0] # Multiply by eta for cells where we are charging
                z -= (time_delta/(3600*q))*ik # Update SOC
            
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
            z_hist[0,:,:] = z
            
            # Update ARX parameters
            if self.cell_model_is_dynamic:
                
                if self.cell_model_is_array:

                    for i in range(Ns):
                        for j in range(Np):
                            b0[i,j] = self.cell_model[i,j]['b0'](z[i,j], T[i,j])
                            
                            for l in range(model_order):
                                a[l,i,j] = self.cell_model[i,j][f'a{l+1}'](z_hist[l,i,j], T_hist[l,i,j])
                                b[l,i,j] = self.cell_model[i,j][f'b{l+1}'](z_hist[l,i,j], T_hist[l,i,j])
                
                else:
                    b0 = self.cell_model['b0'](z, T)
                    for l in range(model_order):
                        a[l,:,:] = self.cell_model[f'a{l+1}'](z_hist, T_hist)
                        b[l,:,:] = self.cell_model[f'b{l+1}'](z_hist, T_hist)
                        
            # Store measurements
            self.simulation_results['pack']['current'][k] = I
            self.simulation_results['pack']['voltage'][k] = V
            self.simulation_results['pack']['min_soc'][k] = np.min(z)
            self.simulation_results['pack']['max_soc'][k] = np.max(z)
            self.simulation_results['pack']['avg_soc'][k] = np.mean(z)
            self.simulation_results['pack']['min_temp'][k] = np.min(T)
            self.simulation_results['pack']['max_temp'][k] = np.max(T)
            self.simulation_results['pack']['avg_temp'][k] = np.mean(T)
            
            for i in range(Ns):
                for j in range(Np):
                    self.simulation_results[f'cell_{i}-{j}']['b0'][k] = b0[i,j]
                    self.simulation_results[f'cell_{i}-{j}']['current'][k] = ik[i,j]
                    self.simulation_results[f'cell_{i}-{j}']['voltage'][k] = vk[i,j]
                    self.simulation_results[f'cell_{i}-{j}']['soc'][k] = z[i,j]
                    self.simulation_results[f'cell_{i}-{j}']['temp'][k] = T[i,j]
                    for l in range(model_order):
                        self.simulation_results[f'cell_{i}-{j}'][f'a{l+1}'][k] = a[l,i,j]
                        self.simulation_results[f'cell_{i}-{j}'][f'b{l+1}'][k] = b[l,i,j]
            
class Vehicle():
    
    def __init__(self, vehicle_model, pack):
   
        
        self.vehicle_model = vehicle_model
        self.pack = pack
        
        self.mass_curb = self.vehicle_model['Mass [kg]'] + self.pack.mass
        self.mass_max = self.mass_curb + self.vehicle_model['Payload [kg]']
        self.mass_rotating = (
            (self.vehicle_model['Motor Inertia [kg/m2]'] + self.vehicle_model['Gear Inertia [kg/m2]']
             ) * self.vehicle_model['Gear Ratio']**2 + self.vehicle_model['Wheel Inertia [kg/m2]']*self.vehicle_model['No. Wheels']
            ) / self.vehicle_model['Wheel Radius [m]']**2
        self.mass_equivalent = self.vehicle_model['Mass [kg]'] + self.mass_rotating
        self.max_speed = 2*np.pi*self.vehicle_model['Wheel Radius [m]']*self.vehicle_model['Max RPM [RPM]']/(60*self.vehicle_model['Gear Ratio']) # m/s
        self.max_power = 2*np.pi* self.vehicle_model['Max Motor Torque [Nm]'] * self.vehicle_model['Rated RPM [RPM]'] / 60000 # kW
        self.drivetrain_efficiency = self.pack.efficiency * self.vehicle_model['Inverter Efficiency'] * self.vehicle_model['Motor Efficiency'] * self.vehicle_model['Gear Efficiency']
        
        self.initial_conditions = None
        self.simulation_results = None

    def initialize_simulation(self, time, speed_desired, time_delta):
        """
        Initializes the storage for the simulation results. This method is only
        intended to be used by the simulate_vehicle method and not on its own.

        Parameters
        ----------
        time : iterable
            Contains the time data for the trip of the vehicle.
        speed_desired : iterable
            Contains the speed data for the trip of the vehicle.
        time_delta : float
            Time between samples in seconds.

        Returns
        -------
        None.

        """
        self.simulation_results = dict()
        
        sim_len = len(time)
        
        self.simulation_results['time'] = np.array(time) # s
        self.simulation_results['time_delta'] = time_delta # s
        self.simulation_results['speed_desired'] = np.clip(np.array(speed_desired), 0, self.max_speed) # m/s
        self.simulation_results['acceleration_desired'] = np.zeros(sim_len) # m/s2
        self.simulation_results['acceleration_force_desired'] = np.zeros(sim_len) # N
        self.simulation_results['aero_force'] = np.zeros(sim_len) # N
        self.simulation_results['rolling_grade_force'] = np.zeros(sim_len) # N
        self.simulation_results['torque_demand'] = np.zeros(sim_len) # N-m
        self.simulation_results['torque_max'] = np.zeros(sim_len) # N-m
        self.simulation_results['regen_limit'] = np.zeros(sim_len) # N-m
        self.simulation_results['torque_limit'] = np.zeros(sim_len) # N-m
        self.simulation_results['torque_motor'] = np.zeros(sim_len) # N-m
        # self.simulation_results['power_demand'] = np.zeros(sim_len) # kW
        self.simulation_results['power_limit'] = np.zeros(sim_len) # kW
        self.simulation_results['battery_demand'] = np.zeros(sim_len) # kW
        # self.simulation_results['current'] = np.zeros(sim_len)
        # self.simulation_results['battery_soc'] = np.zeros(sim_len)
        self.simulation_results['acceleration_force_actual'] = np.zeros(sim_len) # N
        self.simulation_results['acceleration_actual'] = np.zeros(sim_len) # m/s2
        self.simulation_results['motor_speed'] = np.zeros(sim_len) # RPM
        self.simulation_results['motor_power'] = np.zeros(sim_len) # kW
        self.simulation_results['speed_actual'] = np.zeros(sim_len) # m/s
    
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
        
        self.initial_conditions['speed'] = speed # m/s
        self.initial_conditions['motor_speed'] = motor_speed # RPM
        
    def simulate_vehicle(self, time, speed_desired, time_delta, off_times=None):
        """
        Simulates the vehicle given the desired speed profile.

        Parameters
        ----------
        time : iterable
            Contains the time data for the trip of the vehicle.
        speed_desired : iterable
            Contains the speed data for the trip of the vehicle.
        time_delta : float
            Time between samples in seconds.

        Returns
        -------
        None.

        """
        
        # Initialize temporary variables
        
        drag_coef = self.vehicle_model['Drag Coefficient']
        front_area = self.vehicle_model['Frontal Area [m2]']
        roll_coef = self.vehicle_model['Rolling Coefficient']
        road_force = self.vehicle_model['Brake Drag [N]']
        wheel_radius = self.vehicle_model['Wheel Radius [m]']
        gear_ratio = self.vehicle_model['Gear Ratio']
        L_max = self.vehicle_model['Max Motor Torque [Nm]']
        RPM_rated = self.vehicle_model['Rated RPM [RPM]']
        RPM_max = self.vehicle_model['Max RPM [RPM]']
        regen_torque = self.vehicle_model['Fractional Regen Torque Limit']
        overhead_power = self.vehicle_model['Overhead Power [W]']
        
        air_density = 1.225 # kg/m3
        G = 9.81 # m/s2
        
        self.initialize_simulation(time, speed_desired, time_delta)
        
        if not self.initial_conditions:
            # If initial conditions have not been set by the user
            self.set_initial_conditions(speed=speed_desired[0])
        
        sim_len = len(self.simulation_results['time'])
        
        self.simulation_results['speed_actual'][-1] = self.initial_conditions['speed']
        self.simulation_results['motor_speed'][-1] = self.initial_conditions['motor_speed']
        
        for i in range(sim_len):
            
            
            
            self.simulation_results['acceleration_desired'][i] = (
                self.simulation_results['speed_desired'][i] - self.simulation_results['speed_actual'][i-1]
                )/time_delta
            self.simulation_results['acceleration_force_desired'][i] = self.mass_equivalent * self.simulation_results['acceleration_desired'][i]
            self.simulation_results['aero_force'][i] = 0.5*air_density*drag_coef*front_area*(self.simulation_results['speed_actual'][i-1])**2
            
            self.simulation_results['rolling_grade_force'][i] = 0
            
            if self.simulation_results['speed_actual'][i-1] > 0:
                self.simulation_results['rolling_grade_force'][i] += roll_coef*self.mass_max*G
            
            self.simulation_results['torque_demand'][i] = (self.simulation_results['acceleration_force_desired'][i] + 
                                                           self.simulation_results['aero_force'][i] + 
                                                           self.simulation_results['rolling_grade_force'][i] +
                                                           road_force)*wheel_radius/gear_ratio
            
            self.simulation_results['torque_max'][i] = L_max
            if self.simulation_results['motor_speed'][i-1] >= RPM_rated:
                self.simulation_results['torque_max'][i] = self.simulation_results['torque_max'][i]*RPM_rated/self.simulation_results['motor_speed'][i-1]
            
            self.simulation_results['regen_limit'][i] = min(self.simulation_results['torque_max'][i], regen_torque*L_max)
            self.simulation_results['torque_limit'][i] = min(self.simulation_results['torque_demand'][i], self.simulation_results['torque_max'][i])
            
            if self.simulation_results['torque_limit'][i] > 0:
                self.simulation_results['torque_motor'][i] = self.simulation_results['torque_limit'][i]
            else:
                self.simulation_results['torque_motor'][i] = max(-self.simulation_results['regen_limit'][i], self.simulation_results['torque_limit'][i])
            
            self.simulation_results['acceleration_force_actual'][i] = (
                self.simulation_results['torque_limit'][i] * gear_ratio / 
                wheel_radius - self.simulation_results['aero_force'][i] - 
                self.simulation_results['rolling_grade_force'][i] - road_force)
            
            self.simulation_results['acceleration_actual'][i] = self.simulation_results['acceleration_force_actual'][i]/self.mass_equivalent
            self.simulation_results['motor_speed'][i] = min(RPM_max,
                                                            gear_ratio*(
                                                            self.simulation_results['speed_actual'][i-1] + 
                                                            self.simulation_results['acceleration_actual'][i]*time_delta)*60/(
                                                                2*np.pi*wheel_radius))
                                                                       
            self.simulation_results['speed_actual'][i] = self.simulation_results['motor_speed'][i]*2*np.pi*wheel_radius/(60*gear_ratio)
            
            if self.simulation_results['torque_limit'][i] > 0:
                self.simulation_results['motor_power'][i] = self.simulation_results['torque_limit'][i]
            else:
                self.simulation_results['motor_power'][i] = max(self.simulation_results['torque_limit'][i],
                                                             -self.simulation_results['regen_limit'][i])
            
            self.simulation_results['motor_power'][i] = self.simulation_results['motor_power'][i]*2*np.pi/60000*(
                self.simulation_results['motor_speed'][i-1] + self.simulation_results['motor_speed'][i])/2
            
            self.simulation_results['power_limit'][i] = max(-self.max_power, 
                                                         min(self.max_power, 
                                                             self.simulation_results['motor_power'][i]))
            
            self.simulation_results['battery_demand'][i] = overhead_power/1000
            
            if self.simulation_results['power_limit'][i] > 0:
                self.simulation_results['battery_demand'][i] += self.simulation_results['power_limit'][i]/self.drivetrain_efficiency
            else:
                self.simulation_results['battery_demand'][i] += self.simulation_results['power_limit'][i]*self.drivetrain_efficiency

    def simulate_battery_pack(self):
        """
        Wrapper which simulates the battery pack using the generated power 
        demand from the vehicle simulation.

        Returns
        -------
        None.

        """
        
        self.pack.simulate_pack(self.simulation_results['battery_demand'], self.simulation_results['time_delta'])

class ECM():
    
    def __init__(self, q, r0, r1, time_constant, time_delta, ocv_curve, soc_curve):
    
        # Define model parameters
        
        self.q = q
        self.r0 = r0
        self. r1 = r1
        self.time_constant = time_constant
        self.time_delta = time_delta
        self.ocv_curve = ocv_curve
        self.soc_curve = soc_curve
        
        # Set initial states
        
        self.irc = 0
        self.soc = 0.5
        self.ik = 0
    
    def step(self, current):
        
        # Update state
        self.soc = self.soc - self.time_delta/self.q*self.ik
        self.irc = np.exp(-self.time_delta/self.time_constant)*self.irc + (1-np.exp(-self.time_delta/self.time_constant))*self.ik
        # Update input
        self.ik = current
        # Update output
        OCV = np.interp(self.soc, self.soc_curve, self.ocv_curve)
        voltage = OCV - self.r1*self.irc - self.r0*self.ik
        
        return voltage, self.soc
    
    def look(self, current):
        soc = self.soc - self.time_delta/self.q*self.ik
        irc = np.exp(-self.time_delta/self.time_constant)*self.irc + (1-np.exp(-self.time_delta/self.time_constant))*self.ik
        ik = current
        OCV = np.interp(soc, self.soc_curve, self.ocv_curve)
        voltage = OCV - self.r1*irc - self.r0*ik
        
        return voltage, soc
    
if __name__ == '__main__':
    pass
    
    