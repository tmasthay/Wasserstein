import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve

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

#renormalize a given probability distribution
def normalize(f):
  return lambda x : f(x) / quad(f, -np.inf, np.inf)  

#get cdf from pdf
def cumulative(f):
  return lambda x : quad(f,-np.inf, x)

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
      print("i=%s  xx[i]=%s x=%s"%(i,xx[i],x))
      if( i == (len(xx) - 1) ):
        print("WARNING: Possible domain error")
        left_end = len(xx) - 2
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

