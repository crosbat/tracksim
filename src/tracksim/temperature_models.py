
Zheng2024Temp = {'Model name' : 'Zheng2024Temp',
                 'Reference' : 'Y. Zheng, Y. Che, X. Hu, X. Sui, and R. Teodorescu, “Online Sensorless Temperature Estimation of Lithium-Ion Batteries Through Electro-Thermal Coupling,” IEEE/ASME Transactions on Mechatronics, vol. 29, no. 6, pp. 4156–4167, Dec. 2024, doi: 10.1109/TMECH.2024.3367291.',
                 'Description' : 'The temperature part of the thermo-electrical model in Zheng et al., 2024. The corresponding electrical model is accessed by cell_models.load_Zheng2024.',
                 'Cell model number' : 'CALB L148N50B',
                 'Cathode' : 'NMC',
                 'Anode' : 'Graphite',
                 'Form factor' : 'Prismatic',
                 'Model SOC range [%]' : '10 - 90',
                 'Model temperature range [C]' : '25 - 50',
                 'Specific heat capacity [J/(kgK)]' : 0.0008519,
                 'Equivalent convective heat transfer coefficient [W/(m2K)]' : 6.353,
                 'Entropic heat coefficient' : lambda SOC : 0.0007693*SOC**3 - 0.002115*SOC**2 + 0.001514*SOC - 2.375e-05}