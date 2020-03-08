import numpy as np
import matplotlib.pylab as plt
import renormalize as rn
import test_cum as tc

def ricker(s,sigma):
   return lambda t : 2 / (np.sqrt(3 * sigma) * np.pi**(1/4)) \
            * (1 - (t/sigma)**2) * \
            np.exp(-(t - s)**2 / (2 * sigma**2))

def gauss(s,sigma):
   return lambda t : 1 / (np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(t-s)**2 / \
            (2 * sigma**2))

def split_normalize(f):
  f_plus  = abs(f)
  f_minus = abs(f)

  f_plus[np.where(f < 0)]   = 0
  f_minus[np.where(f >= 0)] = 0 

  c_plus  = sum(f_plus)
  c_minus = sum(f_minus)

  c_plus = c_plus if c_plus > 0 else 1
  c_minus = c_minus if c_minus > 0 else 1

  return f_plus / c_plus, f_minus / c_minus
 
def trapz(ff, xx):
  dx    = xx[1:] - xx[:-1]
  ave_y = (ff[1:] + ff[:-1]) / 2
  return np.dot(dx,ave_y)

def cum(f):
  y = np.zeros(len(f))
  y[0] = f[0]
  for i in range(1,len(f)):
    y[i] = y[i-1] + f[i-1]
  c = sum(f)
  c = c if c > 0 else 1
  return y / c

def invert_cdf(F, x, y):
  return np.array(list(\
           map(lambda z : x[min(np.digitize(z, F, True), len(x)-1)],y)))

def wasserstein_proper(f,g,x,y):
  #precondition assertions 
  assert( len(f) == len(g) and len(g) == len(x) )

  #build cumulative, quantile functions
  F = cum(f)
  G = cum(g)
  F_inv = invert_cdf(F, x, y)
  G_inv = invert_cdf(G, x, y)

  #integrate for W_2 norm
  diff = (F_inv - G_inv) ** 2
  return np.sqrt(trapz(diff, y))

def wasserstein(f,g,x,y):
  f_plus, f_minus = split_normalize(f)
  g_plus, g_minus = split_normalize(g)

  return np.sqrt( wasserstein_proper(f_plus, g_plus, x, y)**2 + \
           wasserstein_proper(f_minus, g_minus, x, y)**2 )
  
def tester(shifts):
  mu1    = 0.0
  sigma1 = 1.0

  N_x = 1000
  N_y = 1000
  a   = -4.0
  b   = 8.0
  x = np.linspace(a,b,N_x)
  y = np.linspace(0.0,1.0,N_y)

#  shift_dat = lambda s : np.exp( -((x-mu1+s)**2 / (2 * sigma1**2)) ) \
#                / np.sqrt( 2 * np.pi * sigma1**2 )
  shift_dat = lambda s : np.array(list(map(ricker(mu1+s, sigma1), x)))
#  shift_dat = lambda s : np.array(list(map(gauss(mu1+s,sigma1),x)))
#  f         = shift_dat(0)
#  w2        = lambda s : wasserstein(f, shift_dat(s), x, y)**2

#  dists = np.array(list(map(w2, shifts)))

  tmp_normalize = lambda f: rn.split_normalize(f,a,b)
#  return dists
  rn_shift_dat = lambda s : tc.gauss(mu1+s,sigma1,1.0)

  shift_functions = np.array(list(map(rn_shift_dat, shifts)))
  split_shifts = np.array(list(map(tmp_normalize, shift_functions)))
  
  print('yo')

  plus_dense = []
  neg_dense  = []
  for density in split_shifts:
    tmp1 = rn.cum_disc(density[0],x)
    tmp2 = rn.cum_disc(density[1],x)
    plus_dense.append(tmp1[0])
    neg_dense.append(tmp2[0])

  print('yo')
  
  invert_cdf_peval = lambda FF : rn.invert_cdf_disc_disc(FF, x, y, True)
  plus_quant = np.array(list(map(invert_cdf_peval, plus_dense)))
  neg_quant  = np.array(list(map(invert_cdf_peval, neg_dense )))

  print('yo')

  plt.figure(1)
  plt.title('Gauss Quantile')
  for pq in plus_quant:
    plt.plot(y, pq)

  plt.figure(2)
  plt.title('Gauss CDF')
  for pcdf in plus_dense:
    plt.plot(x, pcdf)

  plt.figure(3)
  plt.title('Ricker Quantile-')
  for nq in neg_quant:
    plt.plot(y, nq)

  plt.figure(4)
  plt.title('Ricker- CDF')
  for ncdf in neg_dense:
    plt.plot(x, ncdf)
  plt.show()
 

#shifts  = np.linspace(-4.0,4.0,160)
#dists   = tester(shifts)
#plt.plot(shifts, dists)
#plt.show() 

tester([-2.0, -1.0, 0.0, 1.0, 2.0])
