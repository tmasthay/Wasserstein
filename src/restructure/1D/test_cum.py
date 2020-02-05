# filename: test_cum.py
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

def ricker(s, sigma):
  def f(t):
    u      = t - s
    C      = 2 / (np.sqrt(3 * sigma) * (math.pi ** (1/4)))
    poly   = 1 - (u/sigma)**2
    e_term = np.exp( -(u**2) / ( 2 * (sigma**2) ) )
    return C * poly * e_term
  return f

def gauss(s,sigma):
  def f(t):
    s_sq = 2 * sigma**2
    return np.exp( -(t-s)**2 / s_sq ) / np.sqrt( math.pi * s_sq )
  return f

def apply(f, x):
  return np.array(list(map(f,x)))

def test_convexity(f,shifts,a,b,N):
  x = np.linspace(a,b,N)
  dists = []
  for g in shifts:
    tmp = rn.partial_wasserstein(f,g,a,b,N)
    print('W2 Norm!!! = %s'%(tmp[0]))
    dists.append(tmp[0]**2)
  return dists

def plot_convexity_test(f,shifts,s,a,b,N,fig_name='',the_title=''):
  x     = np.linspace(a,b,N)
  dists = test_convexity(f,shifts, a, b, N)
  print(dists)
  ff    = np.array(list(map(f,x)))
  shift_plots = []
  for y in shifts:
    shift_plots.append(np.array(list(map(y,x))))

  fig,axs = plt.subplots(2)
  fig.suptitle('Convexity Test: %s'%(the_title))
  axs[0].plot(x,ff,label='Original')
  for i,y in enumerate(shift_plots):
    axs[0].plot(x,y)
  axs[1].plot(s,dists,label='W2(f,f_s)')
  axs[0].legend()
  axs[1].legend()
  fig.savefig(fig_name)

mu       = 0.0
sigma    = 1.0
mu_step  = 0.2
N_s      = 10
N_x      = 10000
h        = gauss(mu,sigma)
rick     = ricker(mu,sigma)
a        = -5.0
b        = 5.0
shifted  = []
rick_s   = []
s        = mu_step * np.array(range(-N_s,N_s+1))

compute_gauss  = False
compute_ricker = True

if( compute_gauss ):
  for shift_length in s:
    print(shift_length)
    shifted.append( gauss( mu + shift_length, sigma) )
  
  plt.figure(4)
  plot_convexity_test(h,                                  \
                      shifted,                            \
                      s,                                  \
                      a, b, N_x,                          \
                      fig_name ='figures/GaussCheck.png', \
                      the_title= 'Gaussian Packets')

if(compute_ricker):
  plt.figure(5)
  for shift_length in s:
    rick_s.append( ricker(mu + shift_length, sigma) )
  plot_convexity_test(rick,                                      \
                      rick_s,                                     
                      s,                                         \
                      a, b, N_x,                                 \
                      fig_name  = 'figures/RickerConvexity-Smaller.png', \
                      the_title = 'Ricker Wavelet')
