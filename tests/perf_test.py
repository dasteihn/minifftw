import minifftw as m
import sys
import time
import numpy as np
import random

exponent = 29
data_len = int(2**exponent)
nr_of_threads = 4
nr_of_loops = 100

print("Calculating an array of length 2^", exponent, " with ", nr_of_threads,
        " threads.")

m.init(sys.argv, nr_of_threads)
data_in = np.random.random(data_len) + np.random.random(data_len) * 1j
data_out = np.zeros(data_len, dtype="complex128")
#print(data)
p_fwd = m.plan_dft_1d(data_in, data_out, m.FFTW_FORWARD, m.FFTW_PATIENT)
p_bwd = m.plan_dft_1d(data_out, data_in, m.FFTW_BACKWARD, m.FFTW_PATIENT)

# note: in productive use you should rescale the array's members as described
# in the FFTW's documentation
for i in range(nr_of_loops):
    res = m.execute(p_fwd)
    m.execute(p_bwd)

erg = m.execute(p)
print(type(erg))
print(len(erg))
m.finit()
print("Test run successfully! :)")
