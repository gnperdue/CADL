#!/usr/bin/env python
import numpy as np
import h5py
import matplotlib.pyplot as plt
import shutil

f = h5py.File('minosmatch_nukecczdefs_127x94x47_xuv_me1Bmc.hdf5', 'r')

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

for i in range(nimages):
    fig = plt.figure(figsize=(1, 1))
    ax = plt.gca()
    ax.axis('off')
    im = ax.imshow(myarr[i], cmap=plt.get_cmap('jet'),
                   interpolation='nearest', vmin=0, vmax=1)
    figname = 'kadenze_%04d.png' % i
    plt.savefig(figname)
    plt.close()

shutil.move('kadenze_0101.png', 'kadenze_0049.png')
shutil.move('kadenze_0100.png', 'kadenze_0063.png')
