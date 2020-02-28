#!/usr/bin/env python
# coding: utf-8

# In[69]:


from dolfin import *
import numpy as np
import matplotlib.pylab as plt

def wave_eq(c, mesh, dt, T_0, T_1, delta, 
            num_intervals=10,                          
            V=FunctionSpace(mesh, "Lagrange", 1),           
            out_name='soln.pvd'):
    # Time variables
    t = T_0
    T = T_1 - T_0

    plot_interval = np.floor(T / (dt * num_intervals))
    print(plot_interval)
    
    # Previous and current solution
    u1= interpolate(Constant(0.0), V)
    u0= interpolate(Constant(0.0), V)

    # Variational problem at each time
    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx + dt*dt*c*c*inner(grad(u), grad(v))*dx
    L = 2*u1*v*dx-u0*v*dx

    bc = DirichletBC(V, 0, "on_boundary")
    A, b = assemble_system(a, L, bc)

    i_plot = 0
    u=Function(V)
    up=[]
    fid = File('solution.pvd')
    while t <= T:
        A, b = assemble_system(a, L, bc)
        delta_eval = delta(t)
        delta_eval.apply(b)
        solve(A, u.vector(), b)
        u0.assign(u1)
        u1.assign(u)
        t += dt

        # Reduce the range of the solution so that we can see the waves
        j = 0
        for i in u.vector():
            i = min(.01, i)
            i = max(-.01, i)
            u.vector()[j] = i;
            j += 1

        if( (i_plot % plot_interval) == 0 ):
            #print('Hello about to plot...')
            up.append(Function(V))
            up[len(up) - 1].assign(u)
            #plot(up, interactive=False, title=str('t = ' + str(t)))
            #plt.plot(mesh.coordinates(), u.vector())
            #fig.write_png(str('u-' + t + '.png'))
            
            
        i_plot += 1
    fid = File(out_name)
    for i in range(0,len(up)):
        t = dt * ( 1 + i * (T_1-T_0) / (dt * num_plots) )
        fid << up[i], t


# In[70]:


c         = 5000
mesh      = RectangleMesh(Point(-2, -2), Point(2, 2),80,80)
dt        = 0.00004
T_0       = 0
T_1       = .004
num_plots = 10
V         = FunctionSpace(mesh, "Lagrange", 1)
delta     = lambda t : PointSource(V, Point(0.0, 0.0), sin(c * 10 * t))

up = wave_eq(c, mesh, dt, T_0, T_1, delta, num_plots, V, 'soln.pvd')


# In[58]:





# In[ ]:




