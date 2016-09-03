#!/usr/bin/env python
import numpy as np
import h5py

dirname = '/Users/perdue/Documents/AI/MINERvA/HDF5files/'
f = h5py.File(dirname + 'minosmatch_nukecczdefs_127x94x47_xuv_me1Bmc.hdf5',
              'r')

nimages = 102
target_H = 100
target_W = 100

myarr = np.zeros((nimages, target_H, target_W))
source_counter = 0
target_counter = 0
while target_counter < nimages:
    if 1 == f['segments'][source_counter]:
        myarr[target_counter, :, 2:96] = f['hits-x'][source_counter,
                                                     0,
                                                     13:113,
                                                     :]
        target_counter += 1
    source_counter += 1

f.close()

myarr[49] = myarr[101]
myarr[63] = myarr[100]
np.save('img_data', myarr[:100])
