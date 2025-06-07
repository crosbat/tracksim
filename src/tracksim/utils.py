"""
Contains utility function used in the main tracksim module as well as other
useful functions.
"""

import numpy as np

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
                           'd[0,1]' : 'np.sign(I)'}
    
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
        
        lambda_string = 'lambda SOC=0.5,T=25, I=0 : '
    
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
    
    