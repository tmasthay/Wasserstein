import renormalize as rn
import numpy as np
import matplotlib.pylab as plt
import math
from scipy.integrate import quad
from scipy import signal
from scipy import special

def gauss(s,sigma):
  def f(t):
    s_sq = 2 * sigma**2
    return np.exp( -(t-s)**2 / s_sq ) / np.sqrt( math.pi * s_sq )
  return f

def cum_gauss(s, sigma):
  def f(t):
    return 0.5 * (1 + special.erf( ( t - s ) / (sigma * np.sqrt(2) ) ) )
  return f

def quantile_gauss(s, sigma):
  def f(p):
    return s + sigma * np.sqrt(2) * special.erfinv(2 * p - 1)
  return f

def correct_wass_int(s1,sigma1,s2,sigma2):
  F = cum_gauss(s1,sigma1)
  G = cum_gauss(s2,sigma2)

  def integrand(x):
    return abs( F(x) - G(x) ) ** 2
  return integrand

def correct_wass_dist(integrand, tol=1e-10):
  val,err = quad(integrand, 0.0, 1.0)
  if( err > tol ):
    print('WARNING: Inaccurate quadrature evaluation of W2 norm with err = ', err)
  return np.sqrt( val )

def apply(f, x):
  return np.array(list(map(f,x)))

def get_col(arr,i):
  return [row[i] for row in arr]

s1     = 0.0
sigma1 = 1.0
s2     = 1.0
sigma2 = 1.0

f = gauss(s1, sigma1)
g = gauss(s2, sigma2)

eps = 1.0e-05


a   = -5.0
b   = 5.0
N_x = 1000
x   = np.linspace(a,b,N_x)

eps           = 1.0e-05
N_p           = 1000
p             = np.linspace(eps,1.0 - eps,N_p)

obj           = rn.wasserstein_integrand(f,g,x,1000)
wass_int_disc = np.array(list(map(obj,p)))
wass_smooth   = signal.savgol_filter(wass_int_disc, 53, 10)

proper_int    = correct_wass_int(s1,sigma1,s2,sigma2)
W2_correct    = correct_wass_dist(proper_int, 1000)

print('W_2(f,g) = '       , rn.wasserstein_distance(f,g,x,1000))
print('PROPER W_2(f,g) = ', W2_correct)

F,F_err  = rn.cum_disc(f,x)
F_smooth = signal.savgol_filter(F, 53, 5)
F_corr   = cum_gauss(s1,sigma1)

G,G_err  = rn.cum_disc(g,x)
G_smooth = signal.savgol_filter(G, 53, 5)
G_corr   = cum_gauss(s2,sigma2)

"""
plt.figure(2) 
plt.plot(x, F       , label='Computed CDF'       )
plt.plot(x, F_smooth, label='Computed Smooth CDF')
plt.plot(x, apply(F_corr,x)  , label='Exact CDF' )
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
"""

#Comparisons above look good
#Smoothed data for the CDF even look the same as the non-smoothed data
#Now we move on to see if the inverison is done well
Fq_exact      = quantile_gauss(s1,sigma1)
Gq_exact      = quantile_gauss(s2,sigma2)
Fq_exact_disc = apply(Fq_exact,p)
Gq_exact_disc = apply(Gq_exact,p)

Fq_approx     = rn.invert_cdf_disc_disc(F, x, p, True)
Gq_approx     = rn.invert_cdf_disc_disc(G, x, p, True)

Fq_smooth     = signal.savgol_filter(Fq_approx, 53, 5)
Gq_smooth     = signal.savgol_filter(Gq_approx, 53, 5)

print('F_exact max/min = %s:::%s\n'%(max(Fq_exact_disc), min(Fq_exact_disc)))
print('F_approx max/min = %s:::%s\n'%(max(Fq_approx), min(Fq_approx)))
print('F_smooth max/min = %s:::%s\n'%(max(Fq_smooth), min(Fq_smooth)))

print('G_exact max/min = %s:::%s\n'%(max(Gq_exact_disc), min(Gq_exact_disc)))
print('G_approx max/min = %s:::%s\n'%(max(Gq_approx), min(Gq_approx)))
print('G_smooth max/min = %s:::%s\n'%(max(Gq_smooth), min(Gq_smooth)))

plt.figure(2)
diff = np.linalg.norm(Fq_exact_disc - Fq_approx)
plt.title('Quantile Comparison #1: %.3e'%(diff))
plt.plot(p, Fq_exact_disc, label='Exact Quantile' )
plt.plot(p, Fq_approx    , label='Approx Quantile')
plt.plot(p, Fq_smooth    , label='Smoothed Approx')
plt.legend()

plt.figure(3)
diff = np.linalg.norm(Gq_exact_disc - Gq_approx)
plt.title('Quantile Comparison #2: %.3e'%(diff))
plt.plot(p, Gq_exact_disc, label='Exact Quantile' )
plt.plot(p, Gq_approx    , label='Approx Quantile')
plt.plot(p, Gq_smooth    , label='Smoothed Approx')
plt.legend()

#Comparisons for above look good visually. Some discontinuities in the inversion
#But if we smooth, it looks really nice
#Now we move on to the total Wasserstein integral between the two of them
W2_integrand_exact = (Fq_exact_disc - Gq_exact_disc)
W2_approx          = (Fq_approx - Gq_approx        ) 
W2_approx_smooth   = (Fq_smooth - Gq_smooth        ) 


plt.figure(4)
plt.title('Wasserstein Integrand Comparison')
plt.plot(p, W2_integrand_exact, label='Exact')
plt.plot(p, W2_approx         , label='Approximate')
plt.plot(p, W2_approx_smooth  , label='Smoothed Approximate')
plt.legend()
plt.show()
