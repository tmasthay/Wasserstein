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

def test_convexity(f,shifts,a,b,N):
  x         = np.linspace(a,b,N)
  dists     = []
  L2_dists  = []
  #norm_func = rn.better_split_normalize
  gamma = -1.0
  norm_func = rn.exp_normalize(gamma)
#  norm_func = rn.normalize_L1
  wass = lambda g : rn.poly_wasserstein(f,g,a,b,N,norm_func)
  L2   = lambda g : rn.L2_norm(f,g,a,b)
  for g in shifts:
    tmp  = wass(g)
#    tmp  = [0.0, 0.0]
    tmp2 = L2(g)
    print('W2 Norm!!! = %s'%(tmp[0]))
    dists.append(tmp[0]**2)
    L2_dists.append(tmp2[0]**2)
  return dists, L2_dists

def plot_convexity_test(f,shifts,s,a,b,N,fig_name='',the_title=''):
  x     = np.linspace(a,b,N)
  dists,L2_dists = test_convexity(f,shifts, a, b, N)
  print(dists)
  ff    = np.array(list(map(f,x)))
  shift_plots    = []
  for i,y in enumerate(shifts):
    if(i == 0 or i == np.ceil(len(shifts)/2) or i == len(shifts) - 1):
      shift_plots.append(np.array(list(map(y,x))))

  fig,axs = plt.subplots(3)
  fig.suptitle('Convexity Test: %s'%(the_title))
  axs[0].plot(x,ff,label='Original')
  for i,y in enumerate(shift_plots):
    axs[0].plot(x,y)
  axs[1].plot(s,dists,label='W2(f,f_s)')
  axs[2].plot(s,L2_dists,label='L_2^2(f,f_s)')
  axs[0].legend()
  axs[1].legend()
  axs[2].legend()
  fig.savefig(fig_name)


mu        = [1.0, 3.0]
sigma     = [0.2, 0.2]
K         = np.sqrt(3 * sigma[0]) * (math.pi ** (.25)) / 2
c         = [K, K]

mu_step  = 0.08
N_s      = 50

N_x      = 1000
h        = gauss(mu,sigma,c)
rick     = ricker(mu,sigma,c)
a        = -4.0
b        = 8.0
shifted  = []
rick_s   = []
s        = mu_step * np.array(range(-N_s,N_s+1))

print(s)
input()

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
                      fig_name ='figures/GaussCheck-L1.png', \
                      the_title= 'Gaussian Packets')

if(compute_ricker):
  plt.figure(5)
  for s_len in s:
    rick_s.append( ricker(mu + s_len, sigma, c) )
  plot_convexity_test(rick,                                     \
                      rick_s,                                     
                      s,                                         \
                      a, b, N_x,                                 \
                      fig_name  = 'figures/RickerConvexity-Combo.png', \
                      the_title = 'Ricker Wavelet')
"""
mu     = 0.0
sigma  = 1.0
shift  = 2.0
c      = 1.0
f      = ricker(mu,sigma,c)
g      = ricker(mu + shift, sigma,c)
N_x    = 1000
N_y    = 1000
a      = -5.0
b      = 5.0
x      = np.linspace(a,b,N_x)
y      = np.linspace(0.0,1.0,N_y)

rn_fg  = rn.better_split_normalize(f,g,a,b)
tmp    = rn_fg[0]
f      = tmp[0]
g      = tmp[1]

def get_stuff(h):
  tmp    = rn.cum_disc(h,x)
  F_disc = tmp[0]
  F_inv  = rn.invert_cdf_disc_disc(F_disc, x, y, True)
  F_inv2 = rn.invert_cdf_disc_disc(F_disc, x, y, False)
  
  F_inv_smooth = signal.savgol_filter(F_inv, 53, 5)
  F_inv2_smooth = signal.savgol_filter(F_inv2, 53, 5)
  return F_disc,F_inv,F_inv2,F_inv_smooth,F_inv2_smooth

F_disc,F_inv,F_inv2,F_inv_smooth,F_inv2_smooth = get_stuff(f)
G_disc,G_inv,G_inv2,G_inv_smooth,G_inv2_smooth = get_stuff(g)

plt.figure(1)
plt.plot(x,F_disc,label='F')
plt.plot(x,G_disc,label='G')
plt.title('CDF for Ricker+, support [-5,5], mu = 0.0, sigma = 1.0')
plt.savefig('figures/cum_distribution.png')
plt.legend()

plt.figure(2)
plt.plot(y,F_inv, label='I,NS')
#plt.plot(y,F_inv2, label='NI,NS')
plt.plot(y,G_inv, label='shift I,NS')
#plt.plot(y,G_inv2, label='shift NI,NS')
plt.title('Quantile Function for Ricker+, support [-5,5], mu = 0.0, sigma = 1.0')
print('about to save')
plt.savefig('figures/inv_nonsmooth.png')
plt.legend()

plt.figure(3)
plt.plot(y,F_inv2, label='NI,NS')
plt.plot(y,G_inv2, label='shift NI,NS')
plt.savefig('figures/inv_interpolate_nonsmooth.png')
plt.legend()

plt.figure(4)
plt.plot(y,F_inv_smooth, label='I,S')
plt.plot(y,F_inv2_smooth, label='NI,S')
plt.plot(y,G_inv_smooth, label='shift I,S')
plt.plot(y, G_inv2_smooth, label='shift NI,S')
plt.savefig('figures/inv_smooth.png')
plt.legend()
"""
