import numpy as np
import my_python as myp
import math
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import os

import matplotlib.pyplot as plt

#solver for 
#      u_tt + c^2 ( u_xx + u_yy ) = f
#      u(.,.,t)   restricted to boundary of square = 0
#      u(.,.,0)   = 0
#      u_t(.,.,0) = g(x,y)

def eval_grid(g, x, y, do_reshape):
    the_meat = myp.np_tensor_map(g,x,y)
    if( do_reshape ):
      return the_meat.reshape((len(x)*len(y)))
    return the_meat

def get_submatrices(c,x,y,dx,dy, j):
    x_stride = len(x) - 2
    start_x = j * x_stride
    end_x   = (j+1) * x_stride
    D = np.zeros((x_stride,x_stride))
    O = np.zeros((x_stride,x_stride))
    print(start_x)
    print(x_stride)
    for i in range(start_x, end_x):
        local_i = i - start_x
        speed_sq = c(x[local_i],y[j]) ** 2
        D[local_i,local_i] = -2 * speed_sq * (1/(dx**2)+1/(dy**2))
        O[local_i,local_i] = 1/(dy**2) * speed_sq
        if( i != start_x ):
          D[local_i,local_i-1] = speed_sq / (dx ** 2)
        if( i != end_x - 1):
          print local_i, local_i + 1
          D[local_i, local_i+1] = speed_sq / (dx**2)
    print 'D = '
    print(D)
    print 'O = '
    print(O)
    return D,O

def build_stiff(c,x,y,dx,dy):
  x_stride = len(x) - 2
  y_stride = len(y) - 2
  dofs = x_stride * y_stride
  A = np.zeros((dofs,dofs))
  for i in range(0,y_stride):
      start_sub = i * x_stride
      end_sub   = (i+1) * x_stride
      diag_indices  = range(start_sub,end_sub)
      right_indices = range(start_sub+x_stride,end_sub)
      left_indices  = range(start_sub,end_sub+x_stride)
      D,O = get_submatrices(c,x,y,dx,dy,i)
      A[start_sub:end_sub, start_sub:end_sub]  = D
      if( i != 0 ):
        A[start_sub:end_sub, (start_sub-x_stride):start_sub]  = O
      if( i != y_stride - 1 ):
          A[start_sub:end_sub, (end_sub):(end_sub+x_stride)] = O
  return A

def solve_forward(n_x,n_y,a,b,T,n_t,cts_force,cts_speed,cts_ic):
  x = np.linspace(a,b,n_x)
  y = np.linspace(a,b,n_y)

  xx = x[1:(len(x)-1)]
  yy = y[1:(len(y)-1)]

  dx = x[1] - x[0]
  dy = y[1] - y[0]
  
  dt  = 1. / (n_t - 1)

  #  cts_force = lambda x,y: 0
  #  cts_speed = lambda x,y: 1
  #  cts_ic    = lambda x,y: cts_speed(x,y) * math.pi * math.sqrt(2) * math.sin(
  #        math.pi * x ) * math.sin( math.pi * y )
  
  disc_ic    = eval_grid(cts_ic,xx,yy, True)
  disc_force = eval_grid(cts_force,xx,yy, True)
  #note...it's better to discretize at this point for c!!!
  #rather than passing x,y,c throughout all these functions!!!

  dofs = (n_x - 2) * (n_y - 2)
  
  u = np.zeros((dofs, n_t+1))
  u[:,1] = dt * disc_ic
  
  stiff_mat = build_stiff(cts_speed,x,y,dx,dy)
  sys_mat   = np.eye(stiff_mat.shape[0]) - dt * stiff_mat 
  
  for n in range(1,n_t):
     rhs = dt * disc_force + 2 * u[:,n] - u[:,n-1]
     print 'About to solve for ', n, '-th time!'
     print 'rhs size = ', rhs.shape
     u[:,n+1] = np.linalg.solve(sys_mat,rhs)    
     print 'Just solved for the ', n, '-th time'
  
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  
  pause_res = .25
  
  #for n in range(0,n_t+1):
  #    X,Y = np.meshgrid(x,y)
  #    Z = np.zeros((len(x),len(y)))
  #    Z[1:(len(x)-1),1:(len(y)-1)] = u[:,n].reshape(len(x)-2,len(y)-2)
  #    ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap='viridis')
  #    print 'Pushed ', n, '-th time!'
  #    plt.pause(pause_res)
  
  X,Y = np.meshgrid(x,y)
  
  def animate(n):
      ax.clear()
      Z = np.zeros((len(x),len(y)))
      Z[1:(len(x)-1),1:(len(y)-1)] = u[:,n].reshape(len(x)-2,len(y)-2)
      ax.plot_surface(X,Y,Z, rstride=1,cstride=1, cmap='viridis')
      ax.set_zlim3d(-.1, .1)
      ax.set_xlim3d(0. , 1.)
      ax.set_ylim3d(0. , 1.)
      ax.set_title('n = %d'%(n))
  
  intvl = .1
  ani = animation.FuncAnimation(fig,animate,interval=intvl*1e+3,blit=False)
  
  return u,ax

#def solve_forward(n_x,n_y,a,b,T,n_t,cts_force,cts_speed,cts_ic)
def solve_forward(arg_dict):
  return solve_forward( arg_dict["n_x"]      ,
                        arg_dict["n_y"]      ,
                        arg_dict["a"]        ,
                        arg_dict["b"]        ,
                        arg_dict["T"]        ,
                        arg_dict["n_t"]      ,
                        arg_dict["cts_force"],
                        arg_dict["cts_speed"],
                        arg_dict["cts_ic"]   )

case_1 = { "n_x":    100,
           "n_y":    100,
           "a"  :      0,
           "b"  :      1,
           "T"  :    .01,
           "n_t":    100 }

case_1["cts_force"] = lambda x,y: 0
case_1["cts_speed"] = lambda x,y: 1
case_1["cts_ic"]    = lambda x,y: cts_speed(x,y) * math.pi * math.sqrt(2) * math.sin(
        math.pi * x ) * math.sin( math.pi * y )


