import numpy as np
import renormalize as rn
import matplotlib.pylab as plt

f = lambda x : np.power(x,3.0)

f_plus, f_minus = rn.split(f)

x = np.linspace(-3.0,3.0,20)

#plt.plot(x, np.array(list(map(f_plus,x ) ) ), \
#         x, np.array(list(map(f_minus,x) ) ) )

f_discrete = np.array(list(map(f,x)))
print(f_discrete)
x_tmp = np.linspace(-4.0,4.0,20)
f_tmp      = rn.linearize(x_tmp,f_discrete)
print(list(map(f_tmp,x)))

plt.plot(x, f_discrete, x, np.array(list(map(f_tmp,x))))
plt.show()


