import sys
import time
import numpy as np
import random
sys.path.append("./build/lib.linux-x86_64-3.7")

import minifftw as m

data_len = 2048
m.init(sys.argv, 4)

try:
    m.import_wisdom("./my_wisdom")
    print("imported wisdom")
except:
    print("could not import wisdom")

data_in = np.random.random(data_len) + np.random.random(data_len) * 1j
data_out = np.zeros(data_len, dtype="complex128")
p = m.plan_dft_1d(data_in, data_out, m.FFTW_FORWARD, m.FFTW_PATIENT)
erg = m.execute(p)

try:
    m.export_wisdom("./my_wisdom")
    print("exported wisdom")
except:
    print("could not export wisdom")

print("Test run successfully! :)")
m.finit()
