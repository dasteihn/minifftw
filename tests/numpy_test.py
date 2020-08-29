import sys
import time
import numpy as np
sys.path.append("./build/lib.linux-x86_64-3.7")

import minifftw as m

m.init(sys.argv, 4)
data = np.random.random(10) + np.random.random(10) * 1j
p = m.plan_dft_1d(data, m.FFTW_FORWARD, m.FFTW_ESTIMATE)

erg = m.execute(p)
time.sleep(1)
m.finit()
time.sleep(2)
print("Test run successfully! :)")
time.sleep(2)
print("und jetzt explodierts...")
