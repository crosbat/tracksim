# TRACKSIM

TRACKSIM (**TR**affic, vehicle, and battery p**ACK** **SIM**ulator) is an open source python library for generating near-life dynamic battery data. The goal of the project is to provide a tool for generating realistic synthetic battery data with a high amount of user-customizability at every simuation level. 
The three levels are:

1. Traffic Level: A collection of vehicles are simulated in a defined traffic network using the simulation tool [SUMO](https://sumo.dlr.de/docs/index.html).
2. Vehicle Level: The power demand of the battery pack is calculated based on the given speed profile of the vehicle.
3. Battery Level: The current and voltage of the battery pack and every cell in the pack are calculated based on the power demand.

# Installing TRACKSIM

TRACKSIM requires SUMO to be installed on the system in order to perform the traffic simulation. A gude for installing SUMO can be found [here](https://sumo.dlr.de/docs/Installing/index.html). The latest version of SUMO which TRACKSIM has been tested with is 1.22.0.

TRACKSIM is indexed in the Python Package Index (PyPI) under the package name [pytracksim](https://pypi.org/project/pytracksim/). To install the latest version of the package, we recommend using [pip](https://pip.pypa.io/en/stable/).
The command for installing TRACKSIM is:

`pip install pytracksim`

# Using TRACKSIM

There are multiple examples available in the repository to get you started using TRACKSIM. In the example below, we simulate a vehicle and battery pack given in [1] with the cells in the battery pack being modeled as in [2].

Setting up the battery pack and the vehicle can be done using two lines of code:
```python
from tracksim.tracksim import Vehicle, Pack

from tracksim.vehicle_models import ChevyVoltTuned # Modified version of the model in [1]
from tracksim.pack_models import ChevyVoltPack # See [1]
from tracksim.cell_models import load_Zheng2024 # See [2]
from tracksim.temperature_models import Zheng2024Temp # See [2]

Zheng2024Cell = load_Zheng2024()
pack = Pack(ChevyVoltPack, Zheng2024Cell, Zheng2024Temp)
vehicle = Vehicle(ChevyVoltTuned, pack)
```

The vehicle and the battery pack can then be simulated using a given trip profile:
```python

from tracksim.example_trips import load_udds

trip_data = load_udds()

time = trip_data['Time [s]']
sample_period = time[1] - time[0]
speed = trip_data['Speed [m/s]']

vehicle.simulate_vehicle(time, speed, sample_period)
vehicle.simulate_battery_pack()
```

Plotting the results can then be done using one line of code:
```python
from tracksim.utils import plot_vehicle_and_battery_data

fig, ax = plot_vehicle_and_battery_data(vehicle)
```

# Citing TRACKSIM
If you use TRACKSIM in you work, please cite the original paper:

> N. A. Weinreich, X. Sui, R. Teodorescu, and K. G. Larsen, “TRACKSIM: A multi-level simulation framework for near-life battery data generation,” in 2025 26th European Conference on Power Electronics and Applications (EPE’25 ECCE Europe), Aalborg, Denmark, Apr. 2025.

You can also use the BibTex:

```
@inproceedings{weinreich_tracksim_2025,
	address = {Aalborg, Denmark},
	title = {{TRACKSIM}: A Multi-Level Simulation Framework for Near-Life Battery Data Generation},
	language = {en},
	booktitle = {2025 26th {European} {Conference} on {Power} {Electronics} and {Applications} ({EPE}'25 {ECCE} {Europe})},
	author = {Weinreich, Nicolai Andre and Sui, Xin and Teodorescu, Remus and Larsen, Kim Guldstrand},
	month = apr,
	year = {2025}
}
```
# Relevant Publications

[1] G. L. Plett, Battery Management Systems, Volume 2: Equivalent Circuit Methods. in Artech House Power engineering series. Boston: Artech house, 2016.

[2] Y. Zheng, Y. Che, X. Hu, X. Sui, and R. Teodorescu, “Online Sensorless Temperature Estimation of Lithium-Ion Batteries Through Electro-Thermal Coupling,” IEEE/ASME Trans. Mechatron., vol. 29, no. 6, pp. 4156–4167, Dec. 2024, doi: 10.1109/TMECH.2024.3367291.

