import renormalize as rn
import numpy as np
import matplotlib.pylab as plt

def f(x):
  if( x < 0 ):
    return 0
  else:
    return np.exp(-x)

F = rn.cumulative(f)
#G = rn.invert_cdf(F)

#x = np.linspace(0,.9999999,1000)
#g = np.array(list(map(G,x)))

#plt.plot(x,g)
#plt.show()

xx               = np.linspace(-100,100,1000)
#FF               = np.array(list(map(F,xx)))
#FF_vals, FF_errs = rn.cum_split(FF)  
FF_vals,FF_errs = rn.cum_disc(f, xx)

for x in FF_errs:
  print(x)

plt.plot(xx,FF_vals,xx,FF_errs)
plt.show()


