import sys
import time
import numpy as np
sys.path.append("./build/lib.linux-x86_64-3.7")

import minifftw as m

m.init(sys.argv, 4)

data = 0 * np.random.random(10) + np.random.random(10) * 1j
print(data)
p = m.plan_dft_1d(data, m.FFTW_FORWARD, m.FFTW_ESTIMATE)
erg = m.execute(p)
m.finit()
print("Test run successfully! :)")
