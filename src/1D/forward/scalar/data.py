from dolfin import *
from ufl import nabla_div
import os
import matplotlib.pyplot as plt
import numpy as np

class wave_functions:
    def ricker(mu, sigma, t):
        return (1-(t/sigma)**2) * exp((t-mu)**2 / (2 * sigma**2))
 
    def my_gauss(mu, sigma, dim):
        s = '1.0'
        for i in range(0,dim):
           s = '%s * exp(pow(x[%s] - %s, 2) / %s)'%(s, i, mu, 
               2 * sigma**2)
        return s

    def ricker_gauss(mu1,sigma1,mu2,sigma2, dim=2):
        return lambda t : '%s * %s'%(wave_functions.ricker(mu1,sigma1,t), 
            wave_functions.my_gauss(mu2,sigma2,2))
   
    def tmp(mu1,sigma1, mu2, sigma2, dim=2):
        g = wave_functions.ricker_gauss(mu1,sigma1,mu2,sigma2, dim)
        return lambda t : (g(t), g(t))

class elastic_data:
    def __init__(self,
                   mesh,
                   on_dirichlet,
                   on_neumann,
                   dirichlet,
                   neumann,
                   body_forces,
                   rho,
                   lmbda,
                   mu,
                   first,
                   second,
                   cfl,
                   T=1.0,
                   basis='CG',
                   deg=1,
                   analytic=(lambda t : ('0.0', '0.0')),
                   storage_dir='figures'):
        self.mesh = mesh
        self.on_dirichlet = dirichlet
        self.on_neumann = on_neumann
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.body_forces = body_forces
        self.basis = basis
        self.deg = deg
        self.T = T
        self.rho = rho
        self.lmbda = lmbda
        self.mu = mu
        self.first = first
        self.second = second
        self.cfl = cfl 
        self.analytic = analytic
        self.storage_dir = storage_dir

class elastic:
    def __init__(self, d): 
        """
        mesh
        on_dirichlet
        on_neumann
        dirichlet
        neumann
        basis
        deg
        rho
        lmbda
         mu
        """
        #mesh info
        self.mesh = d.mesh

        #Function space info
        self.V = VectorFunctionSpace(self.mesh, d.basis, d.deg)
        self.S = FunctionSpace(self.mesh, self.V.ufl_element().family(),
                               self.V.ufl_element().degree())
        #BC info
        self.on_dirichlet = d.on_dirichlet
        self.on_neumann = d.on_neumann
        self.dirichlet = lambda t : DirichletBC(self.V, 
            Expression(d.dirichlet(t), degree=d.deg), self.on_dirichlet)

        self.neumann = lambda t : interpolate(Expression(d.neumann(t),
            degree=d.deg), self.V)

        self.analytic = lambda t : interpolate(Expression(d.analytic(t),
            degree=d.deg), self.V)
                
        self.body_forces = d.body_forces

        
        #define scalar fields
        self.rho = interpolate(Expression(d.rho, degree=d.deg), \
                               self.S)
        self.lmbda = interpolate(Expression(d.lmbda, degree=d.deg),  \
                                 self.S)
        self.mu = interpolate(Expression(d.mu, degree=d.deg), self.S)
        
        #define running solutions
        #current solution
        self.u = Function(self.V)
 
        #solution two steps ago
        self.u0 = interpolate(Expression(d.first, degree=d.deg), self.V)
 
        #previous solution
        self.u1 = interpolate(Expression(d.second, degree=d.deg), self.V)
  
        #time step initialization, based on cfl condition
        p_wave = interpolate(Expression('pow(%s + 2 * %s / %s, 0.5)'%(
                             d.lmbda, d.mu, d.rho),
                             degree=d.deg), self.S)
        h = self.mesh.hmin()
        c = max(p_wave.vector()[:])
        self.dt = d.cfl * h / c
        self.T = d.T
        self.t = 0.0
        self.deg = d.deg
        self.dim = len(d.first)
        self.storage_dir = d.storage_dir

    def __sigma(self, uu):
        return self.lmbda * nabla_div(uu) * Identity(self.dim) + \
               self.mu * (grad(uu) + grad(uu).T)

    def __update_dirichlet(self):
        self.curr_dirichlet = self.dirichlet(self.t)
 
    def __update_neumann(self):
        self.curr_neumann = self.neumann(self.t)
 
    def __update_analytic(self):
        self.curr_analytic = self.analytic(self.t)
 
    def __update_body_forces(self):
        self.curr_body_forces = Expression(self.body_forces(self.t), 
            degree=self.deg)

    def __update_linear_form(self):
        v_test = TestFunction(self.V)
        self.lin_form = 2 * self.rho * inner(self.u1, v_test) * dx - \
                       self.rho * inner(self.u0, v_test) * dx - \
                       Constant(self.dt**2) * \
                       inner(self.__sigma(self.u1),
                           grad(v_test)) * dx + \
                       Constant(self.dt**2) * inner(self.curr_body_forces,
                           v_test) * dx + \
                       Constant(self.dt**2) * inner(self.curr_neumann,
                           v_test) * ds

    def __create_bilinear_form(self):
        v_trial = TrialFunction(self.V)
        v_test = TestFunction(self.V)
        self.B = self.rho * inner(v_trial, v_test) * dx

    def __update(self):
        self.__update_dirichlet()
        self.__update_neumann()
        self.__update_analytic()
        print('about to body force')
        self.__update_body_forces()
        print('body forces created')
        self.__update_linear_form()
 
    def __take_step(self):
       assign(self.u0, self.u1)
       assign(self.u1, self.u)
      # A, b = assemble_system(self.B, self.lin_form, self.curr_dirichlet)
       tmp = DirichletBC(self.V, Expression(('0.0', '0.0'), degree=1), \
                 lambda x,b: b)
       A,b = assemble_system(self.B, self.lin_form, tmp)
       solve(A, self.u.vector(), b)
 
    def __create_files(self):
        cr_file = lambda x : File('%s/%s'%(self.storage_dir, x))
        self.solver_files = [cr_file('x.pvd'), cr_file('y.pvd')]
        self.analytic_files = [cr_file('x_analytic.pvd'), 
            cr_file('y_analytic.pvd')]
        if( self.dim == 3 ):
            self.solver_files.append('z.pvd')
            self.analytic_files.append('z_analytic.pvd')

    def __write_to_vtu(self, the_soln):
        for (i, curr_file) in enumerate(self.solver_files):
            curr_file << (the_soln.sub(i), self.t)
        for (i, curr_analytic_file) in enumerate(self.analytic_files):
            curr_analytic_file << (self.curr_analytic.sub(i), self.t)

    def __clean_output(self, remove=False):
        if(remove):
            os.system("rm -rf %s/*.vtu"%(self.storage_dir))
        os.system("sed -i \'\' \'s/UInt32/Int32/g\' %s"%(
            '%s/*.vtu'%(self.storage_dir)))
        os.system("sed -i \'\' \'s/f_[0-9]*-[0-9]*/f/g\' %s"%(
            '%s/*.vtu'%(self.storage_dir)))
       
  
    def go(self, plot_interval, debug=True):
        if(debug):
            print('dt = %s'%(self.dt))
            print('Time steps = %s'(round(self.T / self.dt)))
            os.system('sleep 5')
        self.__clean_output(True)
        self.__update()
        self.__create_bilinear_form()
        self.__create_files()
        self.__write_to_vtu(self.u)
        self.t = 2 * self.dt
        time_step = 2 
        while(self.t <= self.T):
            self.__update()
            self.__take_step()
            print(self.u.vector()[:])
            self.t = self.t + self.dt
            if( time_step % plot_interval == 0 ):
                self.__write_to_vtu(self.u)
            time_step += 1
        self.__clean_output(False)
  
f = wave_functions.tmp(0.0, 0.5, 0.5, 2.0, 2)

input(f(0))
input(f(5))
 
d = elastic_data(mesh=RectangleMesh(Point(0.0,0.0), Point(1.0,1.0), 10, 10),
     on_dirichlet=(lambda x,b : b),
     on_neumann=(lambda x,b : b and x[1] <= 1.0 - DOLFIN.EPS and False),
     dirichlet=(lambda t : ('0.0', '0.0')),
     neumann=(lambda t : ('0.0', '0.0')),
     body_forces=wave_functions.tmp(0.0,0.5,0.5,2.0,2),
     first=('0.0', '0.0'),
     second=('0.0','0.0'),
     rho='1.0',
     lmbda='1.0',
     mu='1.0',
     cfl=0.1)

elastic(d).go(1)
print('SUCCESS')

