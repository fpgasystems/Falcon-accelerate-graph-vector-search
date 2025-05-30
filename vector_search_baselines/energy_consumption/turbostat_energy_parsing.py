
import argparse 
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str, default='log_energy_idle')

args = parser.parse_args()
fname = args.fname

"""
Example: 

Avg_MHz	Busy%	Bzy_MHz	TSC_MHz	IPC	IRQ	SMI	POLL	C1	C1E	C3	C6	POLL%	C1%	C1E%	C3%	C6%	CPU%c1	CPU%c3	CPU%c6	CPU%c7	CoreTmp	PkgTmp	PkgWatt	RAMWatt	PKG_%	RAM_%
1	0.04	2095	2401	0.89	569	0	0	21	556	0	0	0.00	3.18	96.81	0.00	0.00	99.96	0.00	0.00	0.00	44	50	63.01	33.18	0.00	0.00
1	0.03	2102	2400	0.96	449	0	0	14	462	0	0	0.00	3.12	96.85	0.00	0.00	99.97	0.00	0.00	0.00	45	49	62.98	33.20	0.00	0.00
"""

# check if whether fname is a folder: if so, read out all the files in the folder
if os.path.isdir(fname):
	files = os.listdir(fname) 
	files.sort()
	
	for file in files:
		energy_array = []
		with open(os.path.join(fname, file)) as f:
			lines = f.readlines()
			for i, line in enumerate(lines):
				if i >= 1:
					core_power = float(line.split()[-4])
					ram_power = float(line.split()[-3])
					energy = core_power + ram_power
					energy_array.append(energy)
		print("File: {}\tAverage energy consumption: {:.2f} W".format(file, np.average(energy_array)))
else: 
	energy_array = []
	with open(fname) as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			if i >= 1:
				energy = float(line.split()[-4])
				energy_array.append(energy)
	print("File: {}\tAverage energy consumption: {:.2f} W".format(fname, np.average(energy_array)))
