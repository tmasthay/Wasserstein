#!/usr/bin/env python
# coding: utf-8

#all imports
from dolfin import *
from ufl import nabla_div
import matplotlib.pyplot as plt
import os

# Physical parameters
# rho -- density
# lambbda -- first Lame parameter, intentionally mispelled since "lambda" is a Python keyword
# mu -- second Lame parameter, shear modulus
rho = 1.0
lambbda = 1.0
mu = 10.0

#Spatial domain variables
#Domain = [0,L]x[0,L], with N grid points in each direction
box_length = 2.0
N = 20

#time stepping
#  clf_padding : parameter <= 1 to allow for extra stringent cfl condition if
#                    we see instabilities
#  max_speed : p-wave velocity -- max information transmission rate
#  box_length/N : grid size given for cfl condition
cfl_padding = 0.25
max_speed = sqrt((lambbda + 2 * mu) / rho)
dt = cfl_padding * (box_length/N) / max_speed

#terminal time T
T = 1.0


# In[3]:


#defining mesh, functions spaces, and translating numerical values into Fenics objects

#define mesh = with domain [0,L]x[0,L] with N grid points in each direction
mesh = RectangleMesh(Point(0.0, 0.0), Point(box_length, box_length), N,N)

#define Function space -- V is vector space, S is analoguous scalar field with same mesh and degree
deg = 1 
V = VectorFunctionSpace(mesh, "Lagrange", deg)
S = FunctionSpace(mesh, V.ufl_element().family(), V.ufl_element().degree())

#fieldname_f = "fieldname function" = "Translated version into Fenics"
rho_f = interpolate(Expression('rho'    , degree=deg, rho=rho        ), S)
lambbda_f = interpolate(Expression('lambbda', degree=deg, lambbda=lambbda), S)
mu_f = interpolate(Expression('mu'     , degree=deg, mu=mu          ), S)

# In[4]:


#any helper functions used down the road for terms, plots, etc.

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
    stddev = box_length / 10
    C = 1/sqrt(2 * pi * stddev**2)
    gaussian = '%s * exp(-(pow(x[0] - %s, 2) + pow(x[1] - %s, 2))/%s)' \
                   %(C, 1.0, 1.0, 2 * (stddev**2))
    ricker = '(1 - %s) * exp(-%s)'%((t/stddev)**2, t**2 / (2 * stddev**2))
    if( t > stddev ):
        ricker = '0.0'
    f_1 = '%s * %s'%(ricker, gaussian)
    f_2 = f_1

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
    bc_1 = 'sin(%s * (%s - x[1]))'%(pi / box_length, t)
    bc_2 = '0.0'
    bc_1 = '0.0'
    return DirichletBC(V, Expression((bc_1,bc_2), degree=deg), 
               boundary_dirichlet)


# In[5]:


#setup solution


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

#bilinear form
A = rho_f * inner(v_trial, v_test) * dx
M = assemble(A)

#define linear form for a given time step
def L(u1, u2, t):
    """Generates linear form
    
    :param: u1 -- solution at time t_{n-1}
            u2 -- solution at time t_{n}
            t -- time
    :type: u1 -- Function
           u2 -- Function
           t -- Float
    :rtype: LinearForm
    :return: the linear form to solve for u_{n+1}
    """
    return 2 * rho_f * inner(u2, v_test) * dx - \
           rho_f * inner(u1, v_test) * dx - \
           Constant(dt**2) * inner(sigma(u2,lambbda, mu), grad(v_test)) * dx +\
           Constant(dt**2) * inner(body_forces(t), v_test) * dx

def update_analytic_soln(t):
    """Implements analytic solution to problem for testing purposes
    
    :param: t -- time
    :type: t -- float
    :rtype: Function 
    :return: None -- updated soln function
    """
    k = pi / box_length
    comp_1 = 'sin(%s * (%s - x[1]))'%(k, t)
    comp_2 = '0.0'
    assign(soln, interpolate(Expression((comp_1, comp_2), degree=deg), V))

def set_initial_conditions():
    """
    Sets initial conditions for u1,u2
    
    :param:
    :type:
    :rtype: void -- side effect
    :return: void -- side effect to initialize u1,u2
    """
    #x_comp_1 = 'sin(4 * pi / %s * x[0]) * sin(4 * pi / %s * x[1])'%(box_length,box_length)
    #y_comp_1 = 'sin(4 * pi / %s * x[0]) * sin(4 * pi / %s * x[1])'%(box_length,box_length)
    
    #x_comp_2 = 'sin(4 * pi / %s * x[0]) * sin(4 * pi / %s * x[1])'%(box_length,box_length)
    #y_comp_2 = 'sin(4 * pi / %s * x[0]) * sin(4 * pi / %s * x[1])'%(box_length,box_length)
    #x_comp_1 = '-sin(%s * x[1])'%(pi / box_length)
    x_comp_1 = '0.0'
    y_comp_1 = '0.0'
    
    #x_comp_2 = '%s + %s * cos(%s * x[1])'%(x_comp_1, dt * pi / box_length, pi / box_length)
    x_comp_2 = '0.0'
    y_comp_2 = '0.0'
    
    assign(u1, interpolate(Expression((x_comp_1, y_comp_1), degree=deg), V))
    assign(u2, interpolate(Expression((x_comp_2, y_comp_2), degree=deg), V))
    
def take_step(t):
    """solves for solution for next time
    
    :param: t -- time
    :type: t -- float
    :rtype: void
    :return: void -- side effect to update solution
    """
    b = assemble(L(u1, u2, t))
    bc = get_dirichlet_condition(t)
    bc.apply(M,b)
    solve(M, u.vector(), b)
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
    
    #clean out past files
    os.system('rm *.vtu *.pvd')

    #declare files for each vector component
    xfile = File('x.pvd')
    yfile = File('y.pvd')
    soln_file_x = File('solnx.pvd')
    soln_file_y = File('solny.pvd')

    #write the first two initial conditions to their appropriate files
    xfile << (u1.sub(0), 0.0)
    xfile << (u2.sub(0), dt)
    yfile << (u1.sub(1), 0.0)
    yfile << (u2.sub(1), dt)

    #write analytic solution to appropriate file
    #t = 0
    update_analytic_soln(0.0)
    soln_file_x << (soln.sub(0), 0.0)
    soln_file_y << (soln.sub(1), 0.0)

    #  t = dt
    update_analytic_soln(dt)  
    soln_file_x << (soln.sub(0), dt)
    soln_file_y << (soln.sub(1), dt)

    #main while loop   
    while(t <= T):
        #move forward with bilinear form solve
        #  TODO: check ILU preconditioner
        take_step(t)
        print('t = %s and l2 norm = %s'%(t, get_l2_norm()))
    
         #plot every set number of time steps and guarantee plotting last 
         #  time step
        if( j % 1 == 0 or ((t+dt) > T) ):
            xfile << (u.sub(0), t)
            yfile << (u.sub(1), t)
            soln_file_x << (soln.sub(0), t)
            soln_file_y << (soln.sub(1), t)
            print('j = %s, t = %s'%(j,t))
        j = j+1
        t = t + dt
    #reformat vtu files to work properly
    os.system('sed -i \'\' \'s/f_[0-9]*[-]*[0-9]*/f/g\' *.vtu')
    os.system('sed -i \'\' \'s/UInt32/Int32/g\' *.vtu')
    print('Success!')

go()
