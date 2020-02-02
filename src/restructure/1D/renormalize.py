import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pylab as plt
from scipy import signal

"""
#splits wave f and g into positive and negative parts
def split(f):
  def f_plus(x):
    if( f(x) > 0 ):
      return f(x)
    else:
      return 0.0
  def f_minus(x):
    if( f(x) < 0 ):
      return -f(x)
    else:
      return 0.0
  return f_plus,f_minus
"""
#renormalize a given probability distribution
def normalize(f):
  return lambda x : f(x) / quad(f, -np.inf, np.inf)  

#get cdf from pdf
def cumulative(f):
  return lambda x : quad(f,-np.inf, x)
#  return lambda x : quad(f, 0, x)


def cum_disc(f,x):
  y    = np.zeros(len(x))
  errs = np.zeros(len(x))

  tmp     = quad(f, -np.inf, x[0])
  y[0]    = tmp[0]
  errs[0] = tmp[1] 
  for i in range(1,len(x)):
    tmp     = quad(f, x[i-1], x[i])
    y[i]    = y[i-1] + tmp[0]
    errs[i] = tmp[1]
  return y,errs

#cum split
# Functionality: splits values from integral errors
#  -- F -- cum. dist. in vectorized form
""" 
def cum_split(F):
  vals = np.zeros(len(F))
  errs = np.zeros(len(F))
  for i in range(0,len(F)):
    vals[i] = F[i][0]
    errs[i] = F[i][1]
  return vals,errs
"""

def invert_cdf(F):
  def the_inverse(y):
    return fsolve(lambda x : F(x) - y,0)
  return the_inverse

def invert_cdf_disc_cts(F, xx, interpolate=False):
  def G(y):
    def opt_interval(ii):
      return (F[ii] <= y and y <= F[ii+1])
    step = len(xx) - 2
    i    = 0 
    while( True ):
      if( step == 0 ):
        if( i == 0 ):
          return (xx[i] + xx[i+1]) / 2
        if( i == len(xx) - 1 ):
          return (xx[i-1] + xx[i]) / 2
        return (xx[i-1] + xx[i] + xx[i+1]) / 3
      if( opt_interval(i) ):
        if( i == 0 or i == (len(xx) - 1) ):
          return xx[i]
        if( interpolate ):
          return xx[i] + (y - F[i]) * (xx[i+1] - xx[i]) / (F[i+1] - F[i])
        else:
          return xx[i]
      elif( F[i] > y ):
        if( i == 0 ):
          return xx[0]
        i -= step
      else:
        if( i == len(xx) - 2 ):
          return xx[len(xx)-1]
        i += step
      step //= 2 
  return G
      
def invert_cdf_disc_disc(F, xx, yy, interpolate=False):
  G = invert_cdf_disc_cts(F, xx, interpolate)
  return np.array(list(map(G,yy)))

#computes the integrand that goes into 1D exact formulation for the W_2
#  formulation
#  f,g -- probability distributions -- must be normalized beforehand
#      -- must be squashed into [0,1]
def wasserstein_integrand(f,g,x,N):
   #Rewrite as F^{-1}(y) - G^{-1}(y)
   tol = 1.0e-05
   F,F_err     = cum_disc(f,x)
   G,G_err     = cum_disc(g,x)

   p = np.linspace(0.0,1.0,N)
   #F_inv = invert_cdf_disc_cts(F, x, interpolate=True)
   #G_inv = invert_cdf_disc_cts(G, x, interpolate=True)
   F_inv_disc = invert_cdf_disc_disc(F, x, p, interpolate=True)
   G_inv_disc = invert_cdf_disc_disc(G, x, p, interpolate=True)

   F_inv_smooth = signal.savgol_filter(F_inv_disc, 53, 5)
   G_inv_smooth = signal.savgol_filter(G_inv_disc, 53, 5)

   diff = abs(F_inv_smooth - G_inv_smooth) ** 2

   return linearize(p, diff)
#computes wasserstein distance
# precondition -- f,g are probability densities
def wasserstein_distance(f,g,x,N):
  return quad(wasserstein_integrand(f,g,x,N), 0.0, 1.0)
 
#compute direct wasserstein
def partial_wasserstein(f,g):
  f_norm = normalize(f)
  g_norm = normalize(g)
  return -1.0

#linearize a set of data points and their corresponding domain location
def linearize(xx,yy):
  def f(x):
    left_end = -1
    for i in range(0,len(xx)):
      if( i == (len(xx) - 1) ):
        print("WARNING: Possible domain error")
      elif( xx[i] <= x and x <= xx[i+1] ):
        left_end = i
        break
    if( left_end == -1 ):
      return 0.0
    left_val = yy[left_end]
    dy       = yy[left_end+1] - yy[left_end]
    dx       = xx[left_end+1] - xx[left_end]
    delta_x  = x - xx[left_end]
    fin      = left_val + dy/dx * delta_x
    return fin
  return f
