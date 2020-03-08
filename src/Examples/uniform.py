import numpy as np
import sys
import random

sys.path.append("../1D")
import renormalize as rn
from scipy.integrate import quad
import matplotlib.pylab as plt

n_vals = [10,20,40,80,160,320]

l2_track = np.zeros(len(n_vals))
w2_track = np.zeros(len(n_vals))

for i,n in enumerate(n_vals):
  xx   = np.linspace(0,1,n)
  yy   = np.zeros(n)
  zz   = np.zeros(n)
  for j in range(0,n):
    yy[j] = random.uniform(-1,1)
    zz[j] = random.uniform(-1,1)
  diff     = rn.linearize(xx,(yy - zz) ** 2)
  tmp      = quad(diff,0,1)
  l2_track[i] = np.sqrt(tmp[0])

  yy_cts   = rn.linearize(xx,yy)
  zz_cts   = rn.linearize(xx,zz)
  w2_track[i],_ = rn.partial_wasserstein(yy_cts, zz_cts, 0, 1, n)

  
print('L2: %s'%(l2_track))
print('W2: %s'%(w2_track))

plt.figure(1)
plt.plot(n_vals, l2_track, label='L2')
plt.plot(n_vals, w2_track, label='W2')
plt.legend()
plt.savefig('L2-W2.png')
