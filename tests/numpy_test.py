import minifftw as m
import sys
import time
import numpy as np
import random

exponent = 29
data_len = int(2**exponent)
nr_of_threads = 4

print("Calculating an array of length 2^", exponent, " with ", nr_of_threads,
        " threads.")

m.init(sys.argv, nr_of_threads)
data_in = np.random.random(data_len) + np.random.random(data_len) * 1j
data_out = np.zeros(data_len, dtype="complex128")
print(data_in)
p_fwd = m.plan_dft_1d(data_in, data_out, m.FFTW_FORWARD, m.FFTW_ESTIMATE)
erg = m.execute(p_fwd)
print(erg)

p_bwd = m.plan_dft_1d(data_out, data_in, m.FFTW_BACKWARD, m.FFTW_ESTIMATE)
erg = m.execute(p_bwd)
print(erg)

print(len(erg))
print(data_in / data_len)

m.finit()
print("Test run successfully! :)")
