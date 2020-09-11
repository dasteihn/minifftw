import sys
import time
import numpy as np
import random
sys.path.append("./build/lib.linux-x86_64-3.7")

import minifftw as m

data_len = int(2**20)

m.init(sys.argv, 4)
data_in = np.random.random(data_len) + np.random.random(data_len) * 1j
data_out = np.zeros(data_len, dtype="complex128")
#print(data)
p = m.plan_dft_1d(data_in, data_out, m.FFTW_FORWARD, m.FFTW_ESTIMATE)

erg = m.execute(p)
print(type(erg))
print(len(erg))
m.finit()
print("Test run successfully! :)")
