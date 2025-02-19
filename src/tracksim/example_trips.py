import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

trip_folder = f'{current_dir}/example_trip_data'

udds = pd.read_csv(f'{trip_folder}/udds.csv')
us06 = pd.read_csv(f'{trip_folder}/us06.csv')
Weinreich2025_E45_1 = pd.read_csv(f'{trip_folder}/Weinreich2025_E45_1.csv')
Weinreich2025_E45_2 = pd.read_csv(f'{trip_folder}/Weinreich2025_E45_2.csv')