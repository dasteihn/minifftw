import sys
sys.path.append("../build/lib.linux-x86_64-3.7")

import minifftw as m

new_argv = []
for elm in sys.argv:
    new_argv.append(str(elm))
    

m.init(["hallo"], 4)
l = []
for i in range(0, 8192):
    l.append(-42 + 9001j)

p = m.plan_dft_1d(l, m.FFTW_FORWARD, m.FFTW_ESTIMATE)
erg = m.execute(p)
print("Test run successfully! :)")
