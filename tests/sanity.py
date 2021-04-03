import minifftw as m
import sys
import time
import numpy as np
import random

def checker(a, b):
    ok = 0
    not_ok = 0

    for (i,j) in zip(a, b):
        if i < j + 0.05 and i > j - 0.05:
            ok += 1
        else:
            not_ok += 1

    print("Ok: {}, Not Ok: {}".format(ok, not_ok))

data_len = 8192
nr_of_threads = 1

m.init(sys.argv, nr_of_threads)
data_in = np.zeros(data_len) + np.zeros(data_len) * 1j
for i in range(0, len(data_in)):
    data_in[i] = i + i*1j

data_orig = data_in.copy()

p_fwd = m.plan_dft_1d(data_in, data_in, m.FFTW_FORWARD, m.FFTW_ESTIMATE)
p_bwd = m.plan_dft_1d(data_in, data_in, m.FFTW_BACKWARD, m.FFTW_ESTIMATE)

#print(data_in)

for i in range(0, 1000):
    m.execute(p_fwd)
    m.execute(p_bwd)
    data_in /= data_len
#    if m.get_mpi_rank() == 0:
#        print(data_in)

if m.get_mpi_rank() == 0:
    checker(data_orig, data_in)

print("executed")

#if m.get_mpi_rank() == 0:
#    print(data_orig, "\n", data_in)

time.sleep(1)
m.finit()
