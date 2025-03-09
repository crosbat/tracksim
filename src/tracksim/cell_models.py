import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SOC = 0.5    # 50%
DEFAULT_T   = 25 # deg C

GenericECM = {'Model Name' : None,
               'Reference' : None,
               'Cell Model No.' : None,
               'Cathode' : None,
                'Anode' : None,
                'Form Factor' : None,
                'Nominal Voltage [V]' : None,
                'Minimum Voltage [V]' : None,
                'Maximum Voltage [V]' : None,
                'Nominal Capacity [Ah]' : None,
                'Mass [kg]' : None,
                'Surface Area [m2]' : None,
                'Model Type' : 'ECM',
                'No. RC Pairs' : 1,
                'Model SOC Range [%]' : None,
                'Model Temperature Range [C]' : None,
                'Capacity [Ah]' : None,
                'Coulombic Efficiency' : None,
                'R0' : None,
                'R1' : None,
                'C1' : None,
                'OCV' : None,
                'Tab Resistance [Ohm]' : None}

Zheng2024_OCV = np.load(f'{current_dir}/battery_data/Zheng2024_OCV.npy') # SOC, OCV
Zheng2024Cell = {'Model Name' : 'Zheng2024Cell',
                 'Reference' : 'Y. Zheng, Y. Che, X. Hu, X. Sui, and R. Teodorescu, “Online Sensorless Temperature Estimation of Lithium-Ion Batteries Through Electro-Thermal Coupling,” IEEE/ASME Transactions on Mechatronics, vol. 29, no. 6, pp. 4156–4167, Dec. 2024, doi: 10.1109/TMECH.2024.3367291.',
                 'Cell Model No.' : 'CALB L148N50B',
                 'Cathode' : 'NMC',
                 'Anode' : 'Graphite',
                 'Form Factor' : 'Prismatic',
                 'Nominal Voltage [V]' : 3.66,
                 'Minimum Voltage [V]' : 2.75,
                 'Maximum Voltage [V]' : 4.3,
                 'Nominal Capacity [Ah]' : 50,
                 'Mass [kg]' : 0.865,
                 'Surface Area [m2]' : 0.04364,
                 'Model Type' : 'ECM',
                 'No. RC Pairs' : 1,
                 'Model SOC Range [%]' : '10 - 90',
                 'Model Temperature Range [C]' : '25 - 50',
                 'Positive Charging Current' : True,
                 'Capacity [Ah]' : 50,
                 'Coulombic Efficiency' : 0.99,
                 'R0' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T : 0.003232 - 0.003615*SOC - 7.782e-05*T + 0.004242*SOC**2 + 6.309e-05*SOC*T + 6.866e-07*T**2 - 0.001827*SOC**3 - 2.442e-05*SOC**2*T - 3.971e-07*SOC*T**2,
                 'R1' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T : 0.003629 - 0.01388*SOC - 2.321e-05*T + 0.03267*SOC**2 - 1.802e-05*SOC*T + 3.847e-07*T**2 - 0.0214*SOC**3 + 2.067e-05*SOC**2*T - 2.994e-07*SOC*T**2,
                 'C1' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T : -4.159e+04 + 2.625e+05*SOC + 2767*T - 4.673e+05*SOC**2 - 3183*SOC*T - 25.71*T**2 + 2.727e+05*SOC**3 + 807.7*SOC**2*T + 27.83*SOC*T**2,
                 'OCV': lambda SOC=DEFAULT_SOC,T=DEFAULT_T : np.interp(SOC, Zheng2024_OCV[:,0], Zheng2024_OCV[:,1]),
                 'Tab Resistance [Ohm]' : 0}

Sheikh2025_OCV = np.load(f'{current_dir}/battery_data/Sheikh2025_OCV.npy') # SOC, OCV, dOCVdT, reference temp
ARX1 = {'Model Name' : 'ARX1',
        'Reference' : 'A. M. A. Sheikh, M. C. F. Donkers, and H. J. Bergveld, “Towards Temperature-Dependent Linear Parameter-Varying Models for Lithium-Ion Batteries Using Novel Experimental Design"',
        'Cathode' : 'NMC',
        'Anode' : 'Graphite',
        'Form Factor' : 'Cylindrical',
        'Nominal Voltage [V]' : 3.66,
        'Minimum Voltage [V]' : 2.75,
        'Maximum Voltage [V]' : 4.3,
        'Nominal Capacity [Ah]' : 2.85,
        'Mass [kg]' : 0.1,
        'Model Type' : 'ARX',
        'Model Order' : 1,
        'Model SOC Range [%]' : '0 - 100',
        'Model Temperature Range [C]' : '0 - 40',
        'Positive Charging Current' : True,
        'Capacity [Ah]' : 2.85,
        'Coulombic Efficiency' : 0.99,
        'OCV': lambda SOC=DEFAULT_SOC,T=DEFAULT_T : np.interp(SOC, Sheikh2025_OCV[:,0], Sheikh2025_OCV[:,1]),
        'a1' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T : 0.9933440215515096 - 0.026935937927911158*SOC + 0.0015363842773228609*(1/SOC) + 0.008538228338747142*np.log(SOC) - 8.575836366354313e-05*T,
        'b0' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T : 0.03537836953990937 + 0.0347816001624795*SOC + 0.0002619271937535544*(1/SOC) - 0.017933384633424906*np.log(SOC) - 0.0011966599355291887*T,
        'b1' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T : -0.032216431735397476 - 0.034904510512622604*SOC + 0.00021704828547704054*(1/SOC) + 0.01957181819840551*np.log(SOC) + 0.0011452590232353857*T} 

if __name__ == '__main__':
    pass
