#/usr/bin/env python
# coding: utf-8

#all imports
from dolfin import *
from ufl import nabla_div
import matplotlib.pyplot as plt
import os
import numpy as np

# Physical parameters
# rho -- density
# lmbda -- first Lame parameter, intentionally mispelled since "lambda" is a Python keyword
# mu -- second Lame parameter, shear modulus
rho = 1.0
lmbda = 1.0
mu = 10.0

#Spatial domain variables
#Domain = [0,L]x[0,L], with N grid points in each direction
box_length = 2.0
N = 50

#time stepping
#  cfl_padding : parameter <= 1 to allow for extra stringent cfl condition if
#                    we see instabilities
#  max_speed : p-wave velocity -- max information transmission rate
#  box_length/N : grid size given for cfl condition
cfl_padding = 0.25
max_speed = sqrt((lmbda + 2 * mu) / rho)
dt = cfl_padding * (box_length/N) / max_speed

#terminal time T
T = .4

#defining mesh, functions spaces, and translating numerical values into Fenics objects

#define mesh = with domain [0,L]x[0,L] with N grid points in each direction
mesh = RectangleMesh(Point(0.0, 0.0), Point(box_length, box_length), N,N)

#define Function space -- V is vector space, S is analoguous scalar field with same mesh and degree
deg = 1 
V = VectorFunctionSpace(mesh, "CG", deg)
S = FunctionSpace(mesh, V.ufl_element().family(), V.ufl_element().degree())

#fieldname_f = "fieldname function" = "Translated version into Fenics"
rho_f = interpolate(Expression('rho'    , degree=deg, rho=rho        ), S)
lmbda_f = interpolate(Expression('lmbda', degree=deg, lmbda=lmbda), S)
mu_f = interpolate(Expression('mu'     , degree=deg, mu=mu          ), S)


#set parameters for the analytic solution
def k(m):
    return pi / box_length * m

k_x = k(0)
k_y = k(1)
k_tot = sqrt(k_x**2 + k_y**2)
A = 1.0
#w = sqrt((lmbda + 2 * mu) / rho) * k_x
w = k_tot * sqrt(mu / rho)
print('angular frequency omega = %s and period %s'%(w, 2 * pi / w))

#setup auxiliary Trial and test functions
v_trial = TrialFunction(V)
v_test  = TestFunction(V)

#u1 -- u_{n-1}
#u2 -- u_{n}
#u  -- u_{n+1}
u1 = Function(V)
u2 = Function(V)
u = Function(V)
soln = Function(V)

#helper functions for building terms
def sigma(u,l,m):
    """returns first component of body forces
    
    :param t: time
    :type t: float
    :rtype: string
    :return: functional representation of first component
    """
    return l*nabla_div(u)*Identity(2) + m*(grad(u) + grad(u).T)

def body_forces(t):
    """returns body forces
    
    :param t: time
    :type t: float
    :rtype: Function
    :return: functional representation of forcing term
    """
    """stddev = box_length / 10
    C = 1/sqrt(2 * pi * stddev**2)
    gaussian = '%s * exp(-(pow(x[0] - %s, 2) + pow(x[1] - %s, 2))/%s)' \
                   %(C, 1.0, 1.0, 2 * (stddev**2))
    ricker = '(1 - %s) * exp(-%s)'%((t/stddev)**2, t**2 / (2 * stddev**2))
    if( t > stddev ):
        ricker = '0.0'
    f_1 = '%s * %s'%(ricker, gaussian)
    f_2 = f_1
    """

    f_1 = '0.0'
    f_2 = '0.0'
    return interpolate(Expression((f_1,f_2), degree=deg), V)

def boundary_dirichlet(x, on_boundary):
    """returns boundary for dirichlet condition
    
    :param x -- position
           on_boundary -- tells if we are on boundary dofs
           
    :type x -- array of length d, d = dimension
          on_boundary -- boolean
          
    :rtype: bool
    :return: whether we are on dirichlet boundary or not
    """
    return on_boundary

def get_dirichlet_condition(t):
    """applies Dirichlet boundary condition to solution
    
    :param: t -- time
    :type: t -- float
    :rtype: DirichletBC 
    :return: Dirichlet boundary condition
    """
#    bc_1 = '%s * cos(%s * x[0]) * cos(%s * x[1])'%(\
#          k_x * k_y * A * cos(w * t) * (lmbda + mu), \
#          k_x, k_y)
#    bc_2 = bc_1
    bc_1 = '%s * sin(%s * x[1])'%(A * cos(w * t), k_y)
    bc_2 = '0.0'
    return DirichletBC(V, Expression((bc_1,bc_2), degree=deg), 
               boundary_dirichlet)

def update_analytic_soln(t):
    """Implements analytic solution to problem for testing purposes
    
    :param: t -- time
    :type: t -- float
    :rtype: Function 
    :return: None -- updated soln function
    """
    x_comp = '%s * sin(%s * x[1])'%(A * cos(w * t),\
             k_y)
    y_comp = '0.0'
    assign(soln, interpolate(Expression((x_comp, y_comp), degree=deg), V))

def set_initial_conditions():
    """
    Sets initial conditions for u1,u2
    
    :param:
    :type:
    :rtype: void -- side effect
    :return: void -- side effect to initialize u1,u2
    """
    #x_comp_1 = '%s * sin(%s * x[0]) * sin(%s * x[1])'%(amp[0], k_x, k_y)
    #x_comp_2 = '%s * sin(%s * x[0]) * sin(%s * x[1])'%(amp[1], k_x, k_y)
    
    x_comp_1 = '%s * sin(%s * x[1])'%(A, k_y)
    y_comp_1 = '0.0'

    x_comp_2 = x_comp_1
    y_comp_2 = y_comp_1

    assign(u1, interpolate(Expression((x_comp_1, y_comp_1), degree=deg), V))
    assign(u2, interpolate(Expression((x_comp_2, y_comp_2), degree=deg), V))

def take_step(t, L, M, the_solver):
    """solves for solution for next time
    
    :param: t -- time
    :type: t -- float
    :rtype: void
    :return: void -- side effect to update solution
    """
    b = assemble(L)
    bc = get_dirichlet_condition(t)
    bc.apply(b)
    the_solver.solve(M, u.vector(), b)
    #solve(A == L(u1,u2,t), u, get_dirichlet_condition(t))
    
    assign(u1,u2)
    assign(u2,u)
    update_analytic_soln(t)
    
def get_l2_norm():
    """calculates l2 norm between the solution and analytic solution
    
    :param: t -- time
    :type: t -- float
    :rtype: float
    :return:  \|u - u_{analytic}\|_{L^2}
    """
    x_comp = interpolate(Expression('u_x - soln_x', u_x=u.sub(0), 
                 soln_x=soln.sub(0), degree=deg), S)
    y_comp = interpolate(Expression('u_y - soln_y', u_y=u.sub(1), 
                 soln_y=soln.sub(1), degree=deg), S)
    return sqrt( (norm(x_comp, 'L2')**2) + (norm(y_comp, 'L2')**2) )
    
def go():
    set_initial_conditions()
    t = 2 * dt
    j = 2

    #linear form
    L = 2 * rho_f * inner(u2, v_test) * dx - \
           rho_f * inner(u1, v_test) * dx - \
           Constant(dt**2) * inner(sigma(u2,lmbda, mu), grad(v_test)) \
           * dx +\
           Constant(dt**2) * inner(body_forces(t), v_test) * dx

    #setup up stiffness matrix (constant in time)
    A = rho_f * inner(v_trial, v_test) * dx
    bc = get_dirichlet_condition(t)
    M, res = assemble_system(A, L, bc)
    solver = LUSolver(M, "mumps")
    solver.parameters["symmetric"] = True
   
    #clean out past files
    os.system('rm *.vtu *.pvd')

    #variables that track difference
    diff_x = Function(S)
    diff_y = Function(S)

    #declare files for each vector component
    storage_directory = 'figures'
    cr_file = lambda x : File('%s/%s'%(storage_directory,x))
    xfile = cr_file('x.pvd')
    yfile = cr_file('y.pvd')
    soln_file_x = cr_file('solnx.pvd')
    soln_file_y = cr_file('solny.pvd')
    diff_file_x = cr_file('diffx.pvd')
    diff_file_y = cr_file('diffy.pvd')

    #update the difference
    def update_diff(s, v):
       if( s == 'x' ):
          diff_x.vector()[:] = soln.sub(0).vector()[:] - \
                               v.sub(0).vector()[:]
       else:
          diff_y.vector()[:] = soln.sub(1).vector()[:] - \
                               v.sub(0).vector()[:]

    def write_to_vtu(tt, v):
        xfile << (v.sub(0), tt)
        yfile << (v.sub(1), tt)
        soln_file_x << (soln.sub(0), tt)
        soln_file_y << (soln.sub(1), tt)
        #update_diff('x', v)
        #update_diff('y', v)
        #diff_file_x << diff_x
        #diff_file_y << diff_y

    main_write_vtu = lambda tt : write_to_vtu(tt,u)

    #write for first time step
    update_analytic_soln(0.0)
    write_to_vtu(0.0, u1)
    
    #write for first time step
    update_analytic_soln(dt)
    write_to_vtu(dt, u2)
    
    #main while loop   
    while(t <= T):
        #move forward with bilinear form solve
        #  TODO: check ILU preconditioner

        print('t = %s and l2 norm = %s'%(t, get_l2_norm()))
        """
        bc = get_dirichlet_condition(0.0)
        the_vals = bc.get_boundary_values()
        the_keys = np.array(list(the_vals.keys()))
        mine = u.sub(0).vector()[the_keys]
        analytic = u.sub(0).vector()[the_keys]
        tmp = 0
        for i in range(0,len(mine)):
            print(mine[i])
        """
        take_step(t, L, M, solver)
    
         #plot every set number of time steps and guarantee plotting last 
         #  time step
        if( j % 1 == 0 or ((t+dt) > T) ):
            main_write_vtu(t)
            print('j = %s, t = %s'%(j,t))
        j = j+1
        t = t + dt

    #reformat vtu files to work properly
    os.system('sed -i \'\' \'s/f_[0-9]*[-]*[0-9]*/f/g\' *.vtu')
    os.system('sed -i \'\' \'s/UInt32/Int32/g\' *.vtu')
    print('Success!')

go()

bc = get_dirichlet_condition(0.0)
the_vals = bc.get_boundary_values()
the_keys = np.array(list(the_vals.keys()))
print(the_keys)
print(u.sub(0).vector()[:])
