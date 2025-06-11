import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

trip_folder = f'{current_dir}/example_trip_data'

def load_udds() -> pd.DataFrame:
    udds = pd.read_csv(f'{trip_folder}/udds.csv')
    return udds

def load_us06() -> pd.DataFrame:
    us06 = pd.read_csv(f'{trip_folder}/us06.csv')
    return us06

def load_weinreich2025_E45_1() -> pd.DataFrame:    
    Weinreich2025_E45_1 = pd.read_csv(f'{trip_folder}/Weinreich2025_E45_1.csv')
    return Weinreich2025_E45_1

def load_weinreich2025_E45_2() -> pd.DataFrame:    
    Weinreich2025_E45_2 = pd.read_csv(f'{trip_folder}/Weinreich2025_E45_2.csv')
    return Weinreich2025_E45_2
