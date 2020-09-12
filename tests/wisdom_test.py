import sys
import time
import numpy as np
import random
sys.path.append("./build/lib.linux-x86_64-3.7")

import minifftw as m

m.init(sys.argv, 4)

try:
    m.import_wisdom("./my_wisdom")
    print("imported wisdom")
except:
    print("could not import wisdom")

data = 0 * np.random.random(2048) + np.random.random(2048) * 1j
print(data)
p = m.plan_dft_1d(data, m.FFTW_FORWARD, m.FFTW_ESTIMATE)

erg = m.execute(p)
print(type(erg))
print(erg)
try:
    m.export_wisdom("./my_wisdom")
    print("exported wisdom")
except:
    print("could not export wisdom")

m.finit()
print("Test run successfully! :)")
