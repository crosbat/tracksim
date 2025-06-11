import numpy as np
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SOC = 0.5    # 50%
DEFAULT_T   = 25 # deg C
DEFAULT_I = 0

MODEL_LIST = ['GenericECM', 'Zheng2024']

def load_GenericECM() -> dict:

    GenericECM = {'Model name' : None,
                  'Description' : 'Generic structure for a 2RC Equivalent Circuit Model (EM). The model can easily be extended with extra RC pairs by adding "Ri [Ohm]" and "Ci [Ohm]" with i being the index of the RC pair.',
                  'Reference' : None,
                  'Cell model number' : None,
                  'Cathode' : None,
                  'Anode' : None,
                  'Form factor' : None,
                  'Nominal voltage [V]' : None,
                  'Min voltage [V]' : None,
                  'Max voltage [V]' : None,
                  'Nominal capacity [As]' : None,
                  'Mass [kg]' : None,
                  'Surface area [m2]' : None,
                  'Model type' : 'ECM',
                  'Number of RC pairs' : 1,
                  'Model SOC range [%]' : None,
                  'Model temperature range [C]' : None,
                  'Capacity [As]' : None,
                  'Coulombic efficiency' : None,
                  'R0 [Ohm]' : None,
                  'R1 [Ohm]' : None,
                  'R2 [Ohm]' : None,
                  'C1 [F]' : None,
                  'C2 [F]' : None,
                  'OCV [V]' : None,
                  'Tab resistance [Ohm]' : None}
    
    return GenericECM

def load_Zheng2024() -> dict:
    
    Zheng2024_OCV = pd.read_csv(f'{current_dir}/battery_data/Zheng2024_OCV.csv') # SOC, OCV
    Zheng2024Cell = {'Model name' : 'Zheng2024',
                     'Reference' : 'Y. Zheng, Y. Che, X. Hu, X. Sui, and R. Teodorescu, “Online Sensorless Temperature Estimation of Lithium-Ion Batteries Through Electro-Thermal Coupling,” IEEE/ASME Transactions on Mechatronics, vol. 29, no. 6, pp. 4156–4167, Dec. 2024, doi: 10.1109/TMECH.2024.3367291.',
                     'Description' : '1RC Equivalent Circuit Model (ECM) obtained from experimental data. The ECM is part of an electro-thermal model. The corresponding thermal model is accessed by temperature_models.Zheng2024Temp.',
                     'Cell model number' : 'CALB L148N50B',
                     'Cathode' : 'NMC',
                     'Anode' : 'Graphite',
                     'Form factor' : 'Prismatic',
                     'Nominal voltage [V]' : 3.66,
                     'Min voltage [V]' : 2.75,
                     'Max voltage [V]' : 4.3,
                     'Nominal capacity [As]' : 50*3600,
                     'Mass [kg]' : 0.865,
                     'Surface area [m2]' : 0.04364,
                     'Model type' : 'ECM',
                     'Number of RC pairs' : 1,
                     'Model SOC range [%]' : '10 - 90',
                     'Model temperature range [C]' : '25 - 50',
                     'Positive charging current' : True,
                     'Capacity [As]' : 50*3600,
                     'Coulombic efficiency' : 0.99,
                     'R0 [Ohm]' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T : 0.003232 - 0.003615*SOC - 7.782e-05*T + 0.004242*SOC**2 + 6.309e-05*SOC*T + 6.866e-07*T**2 - 0.001827*SOC**3 - 2.442e-05*SOC**2*T - 3.971e-07*SOC*T**2,
                     'R1 [Ohm]' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T : 0.003629 - 0.01388*SOC - 2.321e-05*T + 0.03267*SOC**2 - 1.802e-05*SOC*T + 3.847e-07*T**2 - 0.0214*SOC**3 + 2.067e-05*SOC**2*T - 2.994e-07*SOC*T**2,
                     'C1 [F]' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T : -4.159e+04 + 2.625e+05*SOC + 2767*T - 4.673e+05*SOC**2 - 3183*SOC*T - 25.71*T**2 + 2.727e+05*SOC**3 + 807.7*SOC**2*T + 27.83*SOC*T**2,
                     'OCV [V]': lambda SOC=DEFAULT_SOC,T=DEFAULT_T : np.interp(SOC, Zheng2024_OCV['SOC'], Zheng2024_OCV['OCV [V]']),
                     'Tab resistance [Ohm]' : 0}
    
    return Zheng2024Cell

def load_LPV1() -> dict:

    Sheikh2025_OCV = pd.read_csv(f'{current_dir}/battery_data/Sheikh2025_OCV.csv') # SOC, OCV, dOCVdT, reference temp
    ARX1 = {'Model name' : 'LPV1',
            'Reference' : 'A. M. A. Sheikh, M. C. F. Donkers, and H. J. Bergveld, “Towards Temperature-Dependent Linear Parameter-Varying Models for Lithium-Ion Batteries Using Novel Experimental Design"',
            'Description' : '',
            'Cathode' : 'NMC',
            'Anode' : 'Graphite',
            'Form factor' : 'Cylindrical',
            'Nominal voltage [V]' : 3.66,
            'Min voltage [V]' : 2.75,
            'Max voltage [V]' : 4.3,
            'Nominal capacity [As]' : 2.85*3600,
            'Mass [kg]' : 0.1,
            'Model type' : 'LPV',
            'Model order' : 1,
            'Model SOC range [%]' : '0 - 100',
            'Model temperature range [C]' : '0 - 40',
            'Positive charging current' : True,
            'Capacity [As]' : 2.85*3600,
            'Coulombic efficiency' : 0.99,
            'OCV [V]': lambda SOC=DEFAULT_SOC,T=DEFAULT_T : np.interp(SOC, Sheikh2025_OCV['SOC'], Sheikh2025_OCV['OCV [V]']),
            'a1' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T,I=DEFAULT_I : 0.9933440215515096 - 0.026935937927911158*SOC + 0.0015363842773228609*(1/SOC) + 0.008538228338747142*np.log(SOC) - 8.575836366354313e-05*T,
            'b0' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T,I=DEFAULT_I : 0.03537836953990937 + 0.0347816001624795*SOC + 0.0002619271937535544*(1/SOC) - 0.017933384633424906*np.log(SOC) - 0.0011966599355291887*T,
            'b1' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T,I=DEFAULT_I : -0.032216431735397476 - 0.034904510512622604*SOC + 0.00021704828547704054*(1/SOC) + 0.01957181819840551*np.log(SOC) + 0.0011452590232353857*T} 

    return ARX1

if __name__ == '__main__':
    pass
