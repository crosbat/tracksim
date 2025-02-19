import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

GenericCell = {'Model Name' : None,
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
                'Model Type' : '1RC ECM',
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

Zheng2024_OCV = np.load(f'{current_dir}/battery_data/Zheng2024_OCV.npy')
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
                 'Model Type' : '1RC ECM',
                 'No. RC Pairs' : 1,
                 'Model SOC Range [%]' : '10 - 90',
                 'Model Temperature Range [C]' : '25 - 50',
                 'Capacity [Ah]' : 50,
                 'Coulombic Efficiency' : 0.99,
                 'R0' : lambda SOC,T : 0.003232 - 0.003615*SOC - 7.782e-05*T + 0.004242*SOC**2 + 6.309e-05*SOC*T + 6.866e-07*T**2 - 0.001827*SOC**3 - 2.442e-05*SOC**2*T - 3.971e-07*SOC*T**2,
                 'R1' : lambda SOC,T : 0.003629 - 0.01388*SOC - 2.321e-05*T + 0.03267*SOC**2 - 1.802e-05*SOC*T + 3.847e-07*T**2 - 0.0214*SOC**3 + 2.067e-05*SOC**2*T - 2.994e-07*SOC*T**2,
                 'C1' : lambda SOC,T : -4.159e+04 + 2.625e+05*SOC + 2767*T - 4.673e+05*SOC**2 - 3183*SOC*T - 25.71*T**2 + 2.727e+05*SOC**3 + 807.7*SOC**2*T + 27.83*SOC*T**2,
                 'OCV' : lambda SOC,T : np.interp(SOC, Zheng2024_OCV[:,0], Zheng2024_OCV[:,1]),
                 'Tab Resistance [Ohm]' : 0}

if __name__ == '__main__':
    pass
