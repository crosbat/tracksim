import numpy as np
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SOC = 0.5    # 50%
DEFAULT_T   = 25 # deg C
DEFAULT_I = 0

MODEL_LIST = ['load_GenericECM', 'load_Zheng2024', 'load_Weinreich2026_15deg', 'load_Weinreich2026_25deg', 'load_Weinreich2026_35deg', 'load_LPV_2_1', 'load_LPV_1']

def load_GenericECM() -> dict:

    GenericECM = {'Model name' : None,
                  'Reference' : None,
                  'Description' : 'Generic structure for a 2RC Equivalent Circuit Model (EM). The model can easily be extended with extra RC pairs by adding "Ri [Ohm]" and "Ci [Ohm]" with i being the index of the RC pair.',
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

def load_Weinreich2026_15deg():
    ocv_curve = pd.read_csv(f'{current_dir}/battery_data/Weinreich2026_OCV_15deg.csv', delimiter=';')
    cell_model = {'Model name' : 'Weinreich2026_15deg',
                  'Reference' : 'N. A. Weinreich & R. Teodorescu, “A Data-Selective Strategy for Online Lithium-Ion Battery Parameter Estimation based on Vehicle Trip Data,” Journal of Energy Storage, 2026, (under revision)',
                  'Description' : '1RC Equivalent Circuit Model (ECM) obtained from experimental data at 15 degrees Celsius',
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
                  'Model temperature range [C]' : '15',
                  'Positive charging current': True,
                  'Capacity [As]' : 48.0005*3600,
                  'Coulombic efficiency' : 1,
                  'OCV data' : {'SOC' : np.sort(ocv_curve['SOC']), # Pre-load the OCV data
                                'OCV [V]' : np.sort(ocv_curve['OCV [V]'])}, 
                  'R0 [Ohm]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(0.004646975021658231)*SOC**5+(-0.005810187967459969)*SOC**4+(-0.0022177146042359082)*SOC**3+(0.0062551322241146115)*SOC**2+(-0.0033813937954153732)*SOC**1+(0.003299023611613875),
                  'R1 [Ohm]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(-2.196952927931575)*SOC**7+(7.621628640520184)*SOC**6+(-10.465270588245096)*SOC**5+(7.222611235305998)*SOC**4+(-2.6528134529102125)*SOC**3+(0.5209396160710493)*SOC**2+(-0.05503480259363774)*SOC**1+(0.004432976875819642),
                  'C1 [F]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(6213764.761164937)*SOC**7+(-21569157.622171577)*SOC**6+(29353487.916299466)*SOC**5+(-19820626.038853996)*SOC**4+(7084525.216647513)*SOC**3+(-1437849.736582748)*SOC**2+(191962.31374090747)*SOC**1+(11493.724821858488),
                  'OCV [V]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(-54.505573396416246)*SOC**7+(219.6438581535074)*SOC**6+(-347.87966078882783)*SOC**5+(272.3229491033813)*SOC**4+(-107.86904459999879)*SOC**3+(19.938552657007154)*SOC**2+(-0.8331992401083151)*SOC**1+(3.489836274509828),
                  'dOCV_dz [V]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(-381.5390137749137)*SOC**6+(1317.8631489210445)*SOC**5+(-1739.3983039441391)*SOC**4+(1089.2917964135252)*SOC**3+(-323.60713379999635)*SOC**2+(39.87710531401431)*SOC**1+(-0.8331992401083151),
                  'SOC' : lambda OCV=3.66, T=25 : np.interp(OCV, cell_model['OCV data']['OCV [V]'], cell_model['OCV data']['SOC']),
                  'Tab resistance [Ohm]' : 0}
    
    return cell_model

def load_Weinreich2026_25deg():
    ocv_curve = pd.read_csv(f'{current_dir}/battery_data/Weinreich2026_OCV_25deg.csv', delimiter=';')
    cell_model = {'Model name' : 'Weinreich2026_25deg',
                  'Reference' : 'N. A. Weinreich & R. Teodorescu, “A Data-Selective Strategy for Online Lithium-Ion Battery Parameter Estimation based on Vehicle Trip Data,” Journal of Energy Storage, 2026, (under revision)',
                  'Description' : '1RC Equivalent Circuit Model (ECM) obtained from experimental data at 25 degrees Celsius',
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
                  'Model temperature range [C]' : '25',
                  'Positive charging current': True,
                  'Capacity [As]' : 50.0678*3600,
                  'Coulombic efficiency' : 1,
                  'OCV data' : {'SOC' : np.sort(ocv_curve['SOC']), # Pre-load the OCV data
                                'OCV [V]' : np.sort(ocv_curve['OCV [V]'])}, 
                  'R0 [Ohm]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(0.0007474883944073354)*SOC**5+(0.0015963343521464913)*SOC**4+(-0.007019304104653645)*SOC**3+(0.007355800874863159)*SOC**2+(-0.0032387229893758624)*SOC**1+(0.0026573998951734523),
                  'R1 [Ohm]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(-2.6372269087050837)*SOC**7+(9.526734297127405)*SOC**6+(-13.808486011580968)*SOC**5+(10.256893209363112)*SOC**4+(-4.15610445032266)*SOC**3+(0.914632338835608)*SOC**2+(-0.1035908623381607)*SOC**1+(0.006248354852943021),
                  'C1 [F]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(10338890.705025505)*SOC**7+(-37323123.096180916)*SOC**6+(53585079.493285365)*SOC**5+(-38958444.135351695)*SOC**4+(15304800.272209348)*SOC**3+(-3327727.1957252175)*SOC**2+(410797.64094816596)*SOC**1+(4248.692810452451),
                  'OCV [V]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(-74.68799876967488)*SOC**7+(312.72939381835835)*SOC**6+(-521.1910979514379)*SOC**5+(439.1460613480804)*SOC**4+(-195.73455998907093)*SOC**3+(44.450070738194256)*SOC**2+(-3.9990397826863235)*SOC**1+(3.606312443439931),
                  'dOCV_dz [V]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(-522.8159913877241)*SOC**6+(1876.37636291015)*SOC**5+(-2605.9554897571893)*SOC**4+(1756.5842453923217)*SOC**3+(-587.2036799672128)*SOC**2+(88.90014147638851)*SOC**1+(-3.9990397826863235),
                  'SOC' : lambda OCV=3.66, T=25 : np.interp(OCV, cell_model['OCV data']['OCV [V]'], cell_model['OCV data']['SOC']),
                  'Tab resistance [Ohm]' : 0}
    
    return cell_model


def load_Weinreich2026_35deg():
    ocv_curve = pd.read_csv(f'{current_dir}/battery_data/Weinreich2026_OCV_35deg.csv', delimiter=';')
    cell_model = {'Model name' : 'Weinreich2026_35deg',
                  'Reference' : 'N. A. Weinreich & R. Teodorescu, “A Data-Selective Strategy for Online Lithium-Ion Battery Parameter Estimation based on Vehicle Trip Data,” Journal of Energy Storage, 2026, (under revision)',
                  'Description' : '1RC Equivalent Circuit Model (ECM) obtained from experimental data at 35 degrees Celsius',
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
                  'Model temperature range [C]' : '35',
                  'Positive charging current': True,
                  'Capacity [As]' : 51.4639*3600,
                  'Coulombic efficiency' : 1,
                  'OCV data' : {'SOC' : np.sort(ocv_curve['SOC']), # Pre-load the OCV data
                                'OCV [V]' : np.sort(ocv_curve['OCV [V]'])}, 
                  'R0 [Ohm]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(0.005453081168393732)*SOC**5+(-0.00993846812450695)*SOC**4+(0.003167765828631284)*SOC**3+(0.0032439181664203104)*SOC**2+(-0.002346014906843447)*SOC**1+(0.0022120116893704907),
                  'R1 [Ohm]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(-2.897086672345828)*SOC**7+(10.638591654432002)*SOC**6+(-15.672123951818437)*SOC**5+(11.842351840207616)*SOC**4+(-4.887340542936391)*SOC**3+(1.0918098694414549)*SOC**2+(-0.12305633723272888)*SOC**1+(0.006732252182657636),
                  'C1 [F]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(5498001.321112491)*SOC**7+(-20910050.591272548)*SOC**6+(31622214.250546142)*SOC**5+(-24298878.332099482)*SOC**4+(10253477.156840026)*SOC**3+(-2505161.938768506)*SOC**2+(367469.25904114527)*SOC**1+(6911.005702362503),
                  'OCV [V]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(-43.07272871003196)*SOC**7+(186.11198755604113)*SOC**6+(-318.3639311403839)*SOC**5+(272.2361935954115)*SOC**4+(-120.39055832238411)*SOC**3+(25.99483648970895)*SOC**2+(-1.7234240813645745)*SOC**1+(3.4897741486067084),
                  'dOCV_dz [V]' : lambda SOC=DEFAULT_SOC, T=DEFAULT_T: +(-301.5091009702237)*SOC**6+(1116.6719253362467)*SOC**5+(-1591.8196557019196)*SOC**4+(1088.944774381646)*SOC**3+(-361.17167496715234)*SOC**2+(51.9896729794179)*SOC**1+(-1.7234240813645745),
                  'SOC' : lambda OCV=3.66, T=DEFAULT_T : np.interp(OCV, cell_model['OCV data']['OCV [V]'], cell_model['OCV data']['SOC']),
                  'Tab resistance [Ohm]' : 0}
    
    return cell_model


def load_LPV_2_1():
    Sheikh2025_OCV = pd.read_csv(f'{current_dir}/battery_data/Sheikh2025_OCV.csv') # SOC, OCV, dOCVdT, reference temp
    model = {'Model name' : 'LPV_2_1',
            'Reference' : 'A. M. A. Sheikh, M. C. F. Donkers, and H. J. Bergveld, “A comprehensive approach to sparse identification of linear parameter-varying models for lithium-ion batteries using improved experimental design,” Journal of Energy Storage, vol. 95, p. 112581, Aug. 2024, doi: 10.1016/j.est.2024.112581.',
            'Description' : 'Linear Parameter-Varying (LPV) model with model order 2 and nonlinearity order 1.',
            'Cathode' : 'NMC',
            'Anode' : 'Graphite',
            'Form factor' : 'Cylindrical',
            'Nominal voltage [V]' : 3.66,
            'Min voltage [V]' : 2.75,
            'Max voltage [V]' : 4.3,
            'Nominal capacity [As]' : 3600,
            'Mass [kg]' : 0.1,
            'Model type' : 'LPV',
            'Model order' : 2,
            'Nonlinearity order' : 1,
            'Model SOC range [%]' : '0 - 100',
            'Model temperature range [C]' : '0 - 40',
            'Positive charging current' : True,
            'Capacity [As]' : 3440.05372,
            'Coulombic efficiency' : 0.99,
            'OCV [V]': lambda SOC=DEFAULT_SOC,T=DEFAULT_T : np.interp(SOC, Sheikh2025_OCV['SOC'], Sheikh2025_OCV['OCV [V]']),
            'a1' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T,I=DEFAULT_I : 0.9829323374906082 + 0.0029357490083367637*np.sign(I),
            'a2' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T,I=DEFAULT_I : 0.0001675509416212704*np.sign(I) + 0.0007150281819278089*(1/SOC),
            'b0' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T,I=DEFAULT_I : 0.06533459793664415 + 0.002896300625985309*np.sign(I) + 0.008988174124499539*SOC - 0.0002442796046266317*(1/SOC) - 0.03600794119221361*np.log(SOC) + 0.07140289583237497*np.exp(0.05*np.sqrt(np.abs(I))),
            'b1' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T,I=DEFAULT_I : 0.5980716373164474 - 0.014736520881731318*np.sign(I) - 0.013705950915958595*SOC + 0.0077557073940416575*(1/SOC) + 0.04259382776821052*np.log(SOC) - 0.6881498407470136*np.exp(0.05*np.sqrt(np.abs(I))),
            'b2' : lambda SOC=DEFAULT_SOC,T=DEFAULT_T,I=DEFAULT_I : -0.49461251590396527 + 0.006828469741895115*np.sign(I) - 0.005603817279896843*(1/SOC) + 0.4605303567371089*np.exp(0.05*np.sqrt(np.abs(I)))
            }
            
    return model

def load_LPV1() -> dict:

    Sheikh2025_OCV = pd.read_csv(f'{current_dir}/battery_data/Sheikh2025_OCV.csv') # SOC, OCV, dOCVdT, reference temp
    LPV1 = {'Model name' : 'LPV1',
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

    return LPV1

if __name__ == '__main__':
    pass
