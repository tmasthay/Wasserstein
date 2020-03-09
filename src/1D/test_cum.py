#filename: test_cum.py
#
# description: Test W2 implementation with Gaussian and Ricker wavelets
# author: Tyler Masthay
# project: W2 norm for Full Waveform Inversion
# repo: https://github.com/tmasthay/wasserstein
#
#

import renormalize as rn
import numpy as np
import matplotlib.pylab as plt
import math
from scipy.integrate import quad
from scipy import signal
from scipy import special
import math

def ricker(s, sigma, c):
  if( isinstance(s, int) or isinstance(s, float) ):
    def f(t):
      u      = t - s
      C      = 2 / (np.sqrt(3 * sigma) * (math.pi ** (1/4)))
      poly   = 1 - (u/sigma)**2
      e_term = np.exp( -(u**2) / ( 2 * (sigma**2) ) )
      return c * C * poly * e_term
    return f
  else:
    print('ARRAY')
    modes = []
    for i,ss in enumerate(s):
      modes.append( ricker(ss, sigma[i], c[i]) )
    def f(t):
      the_sum = 0
      for m in modes:
        the_sum += m(t)
      return the_sum
    return f

def gauss(s,sigma, c):
  if( isinstance(s, int) or isinstance(s,float) ):
    def f(t):
      s_sq = 2 * sigma**2
      return np.exp( -(t-s)**2 / s_sq ) / np.sqrt( math.pi * s_sq )
    return f
  else:
    modes = []
    for i,ss in enumerate(s):
       modes.append( gauss(ss, sigma[i], c[i]) )
    the_sum = 0
    def f(t):
      the_sum = 0
      for m in modes:
        the_sum += m(t)
      return the_sum
    return f

def apply(f, x):
  return np.array(list(map(f,x)))

def test_convexity(f,shifts,norm_func,a,b,N):
  #don't think I need x here since I'm not plotting
#  x         = np.linspace(a,b,N)

  #initialize data
  dists     = []
  L2_dists  = []
  wass = lambda g : rn.poly_wasserstein(f,g,a,b,N,norm_func)
  L2   = lambda g : rn.L2_norm(f,g,a,b)
  for g in shifts:
    tmp  = wass(g)
    tmp2 = L2(g)
    print('W2 Norm!!! = %s'%(tmp[0]))
    dists.append(tmp[0]**2)
    L2_dists.append(tmp2[0]**2)
  return dists, L2_dists

def plot_convexity_test(f,s,norm_func,a,b,N,the_title='',fig_name=''):
  #generate shifted functions and get distances
  shifts = np.array(list(map(lambda t : lambda x: f(x-t), s)))
  W2_dists,L2_dists = test_convexity(f,shifts, norm_func,a, b, N)
  
  #create prob. density plot
  x     = np.linspace(a,b,N)
  ff    = np.array(list(map(f,x)))

  shift_plots    = []
  for i,y in enumerate(shifts):
    if(i == 0 or i == np.floor(len(shifts)/2) or i == len(shifts) - 1):
      shift_plots.append(np.array(list(map(y,x))))

  fig,axs = plt.subplots(3)
  fig.suptitle('Convexity Test: %s'%(the_title))
  for i,y in enumerate(shift_plots):
    axs[0].plot(x,y)
  axs[1].plot(s,W2_dists,label='W2(f,f_s)')
  axs[2].plot(s,L2_dists,label='L_2^2(f,f_s)')
  axs[1].legend()
  axs[2].legend()
  fig.savefig(fig_name)
  
  return s,W2_dists,L2_dists

mu        = [0.0]
sigma     = [0.2]
K         = np.sqrt(3 * sigma[0]) * (math.pi ** (.25)) / 2
c         = [K, K]

mu_step  = 0.04
N_s      = 60

N_x      = 1000
da_gauss = gauss(mu,sigma,c)
rick     = ricker(mu,sigma,c)
a        = -5.0
b        = 5.0
shifted  = []
rick_s   = []
s        = mu_step * np.array(range(-N_s,N_s+1))
norm_func = rn.better_split_normalize

#runtime decisions as to what we actually do
compute_gauss  = False
compute_ricker = True
go = True

if( compute_gauss and go ):
  plot_convexity_test(da_gauss,s,norm_func,a,b,N_x,\
                      'Ricker Convexity Test',\
                      'figures/Gauss/GaussConvexityTest.png')

if( compute_ricker and go ):
  s,W2_dists,L2_dists = plot_convexity_test(rick,s,norm_func,a,b,N_x,\
                          'Ricker Convexity Test',\
                          'figures/Ricker/RickerConvexityTest.png')
  print(list(zip(s,W2_dists)))
