import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pylab as plt
from scipy import signal

def qquad(f,a,b):
  return quad(f,a,b,limit=200)

def trapz(ff, xx):
  dx    = xx[1:] - xx[:-1]
  ave_y = (ff[1:] + ff[:-1]) / 2
  return np.dot(dx,ave_y)

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
def normalize(f, a, b):
#  C,D = quad(f,a,b)
  xx = np.linspace(a,b,1000)
  yy = np.array(list(map(f,xx)))
  C  = trapz(yy,xx)

  print('Normalize: %s'%(C))
  if( C == 0 ):
    plt.plot(xx,yy)
    plt.show()

  if( C == 0 ):
    return f
  return lambda x : f(x) / C

#splits signal into 
def split_normalize(f, a, b):
  f_p,f_m = split(f)  
  f_p = normalize(f_p,a,b)
  f_m = normalize(f_m,a,b)
  return f_p, f_m

def better_split_normalize(f,g,a,b):
  f_p,f_m = split_normalize(f,a,b)
  g_p,g_m = split_normalize(g,a,b)
  
  return [(f_p,g_p), (f_m,g_m)]

def exp_normalize_full(f,g,gamma,a,b):
   h = lambda x : np.exp(-gamma * (f(x) - g(x)))
   C = quad(h, a, b)
   return lambda x : h(x) / C

def exp_normalize(gamma):
  def helper(f,g,a,b):
    return exp_normalize_full(f,g,gamma,a,b)
  return [helper]

def normalize_L1(f,g,a,b):
  ff = lambda x : abs(f(x))
  gg = lambda x : abs(g(x))
  return [(normalize(ff,a,b), normalize(gg,a,b))]

def cum_disc(f,x,left_endpoint=-np.inf):
  y    = np.zeros(len(x))
  errs = np.zeros(len(x))

  tmp     = quad(f, left_endpoint, x[0])
  y[0]    = tmp[0]
  errs[0] = tmp[1] 
  for i in range(1,len(x)):
    tmp     = quad(f, x[i-1], x[i])
    y[i]    = y[i-1] + tmp[0]
    errs[i] = tmp[1]
  return y,errs
 
def invert_cdf_disc_disc(FF, xx, yy, interpolate=False):
  return np.array(list(map(\
      lambda z : xx[min(len(FF)-1, np.digitize(z, FF, True))],yy))) 

#computes the integrand that goes into 1D exact formulation for the W_2
#  formulation
#  f,g -- probability distributions -- must be normalized beforehand
#      -- must be squashed into [0,1]
def wasserstein_integrand(f,g,x,N,smooth=False):
   #Rewrite as F^{-1}(y) - G^{-1}(y)
   tol = 1.0e-05
   F,F_err     = cum_disc(f,x,x[0])
   G,G_err     = cum_disc(g,x,x[0])

   if( max(F_err) > tol ):
     print('Significant error in CDF')

   p = np.linspace(0.0,1.0,N)
   F_inv_disc = invert_cdf_disc_disc(F, x, p, 
     interpolate=False)
   G_inv_disc = invert_cdf_disc_disc(G, x, p, 
     interpolate=False)

   half = np.round(N/2)
   wind = half + (np.mod(half,2) + 1)
   if( smooth ):
     F_inv_disc = signal.savgol_filter(F_inv_disc, 53, 1)
     G_inv_disc = signal.savgol_filter(G_inv_disc, 53, 1)

   diff = abs(F_inv_disc - G_inv_disc) ** 2

   return linearize(p, diff)

#computes wasserstein distance
# precondition -- f,g are probability densities
# outputs W_2(f,g) where f,g are continuous and defined on the discrete
# set x
def wasserstein_distance(f,g,x,N):
  print('integrate wass')
  return np.sqrt(quad(wasserstein_integrand(f,g,x,N), 0.0, 1.0))
 
#compute direct wasserstein
def partial_wasserstein(f,g,a,b,N):
  f_p, f_m = split_normalize(f,a,b)
  g_p, g_m = split_normalize(g,a,b)

  x = np.linspace(a,b,N)

  pos_contrib = (wasserstein_distance(f_p,g_p,x,N))**2
  neg_contrib = (wasserstein_distance(f_m,g_m,x,N))**2

  return np.sqrt( pos_contrib + neg_contrib ) 

def L2_norm(f,g,a,b):
  h = lambda x : (f(x) - g(x))**2
  print('integrate L2')
  return np.sqrt( quad(h, a, b) )

#polymorphic wasserstein distance
#  input
#    -- f,g       -- any function
#    -- a,b       -- domain of integration
#    -- N         -- quantile function domain fineness
#    -- norm_func -- function that converts f,g into PDFs
#  output
#    sqrt(sum_{i=1}^{k} W_2(f_i,g_i)^2) where f_i,g_i are renormalized 
#      according to rule norm_func
def poly_wasserstein(f,g,a,b,N,norm_func):
  print('yo wass')
  rn_f_g  = norm_func(f,g,a,b)
  print('yo again')
  x       = np.linspace(a,b,N)
  tot     = 0

  print(rn_f_g)
 
  for h in rn_f_g:
    tmp = wasserstein_distance(h[0],h[1],x,N)**2
    tot += tmp
  
  print('Renormalization array length = %s with W_2 = %s'%(len(rn_f_g), \
    np.sqrt(tot)))
  return np.sqrt(tot)

def shift(f,s):
  return lambda t : f(t-s)


def linearize(xx,ff):
  def lin(x):
     i = np.digitize(x,xx,False)
     return ff[i-1] + (ff[i] - ff[i-1])\
            / (xx[i]-xx[i-1]) * (x - xx[i-1])
  return lin
