from dolfin import *
from ufl import nabla_div
import os
import signal
import sys
import matplotlib.pyplot as plt
import numpy as np
import time

class wave_functions:
    def ricker(mu, sigma, t):
        return (1-(t/sigma)**2) * exp(-(t-mu)**2 / (2 * sigma**2))
 
    def my_gauss(mu, sigma, dim):
        s = '1.0'
        for i in range(0,dim):
           s = '%s * exp(-pow(x[%s] - %s, 2) / %s)'%(s, i, mu, 
               2 * sigma**2)
        return s

    def ricker_gauss(mu1,sigma1,mu2,sigma2, dim=2):
        return lambda t : '%s * %s'%(wave_functions.ricker(mu1,sigma1,t), 
            wave_functions.my_gauss(mu2,sigma2,2))
   
    def tmp(mu1,sigma1, mu2, sigma2, dim=2):
        g = wave_functions.ricker_gauss(mu1,sigma1,mu2,sigma2, dim)
        return lambda t : (g(t), g(t))

#d term in Duke paper
def PML_damping(d):
   return Expression('x[i] - a <= L ? d_0 * left * pow((L-x[i])/L, p) %s'%(
       ': (b-x[i] < L ? d_0 * right * pow((x[i]-b+L)/L,p) : 0)'),
       L=d['L'], left=d['left'], right=d['right'],
       d_0=d['amp'], a=d['a'], b=d['b'],
       p=d['p'], i=d['coord'], degree=d['deg'])

#beta term in Duke paper
def PML_butter(d):
    return Expression('x[i] - a <= L ? 1 + (b_0 - 1) * left * %s'%(
        'pow((L-x[i])/L, p) %s'%(
        ': (b-x[i] <  L ? 1 + (b_0 - 1) * right * pow((x[i]-b+L)/L, p) : 1)'
        )),
        L=d['L'], left=d['left'], right=d['right'],
        b_0=d['amp'], a=d['a'], b=d['b'],
        p=d['p'], i=d['coord'], degree=d['deg'])

#alpha term in Duke paper
def PML_shift(d):
   return Expression('x[i] - a <= L ? a_0 * left * %s'%( 
	       '(1 - pow((L-x[i])/L, p)) %s'%(
       ': (b-x[i] < L ? a_0 * right * (1 - pow((x[i]-b+L)/L, p)) : a_0)')),
       L=d['L'], left=d['left'], right=d['right'],
       a_0=d['amp'], a=d['a'], b=d['b'],
       p=d['p'], i=d['coord'], degree=d['deg'])

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
                   damp,
                   butter,
                   shift,
                   T=1.0,
                   basis='CG',
                   deg=1,
                   analytic=(lambda t : ('0.0', '0.0')),
                   storage_dir='figures'):
        self.mesh = mesh
        self.on_dirichlet = on_dirichlet
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
        self.damp = damp
        self.butter = butter
        self.shift = shift
        self.storage_dir = storage_dir
        if( not os.path.isdir(self.storage_dir) ):
            os.system('mkdir %s'%(self.storage_dir))

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

        #degree
        self.deg = d.deg

        #Function space info
        self.V = VectorFunctionSpace(self.mesh, d.basis, d.deg)
        self.S = FunctionSpace(self.mesh, self.V.ufl_element().family(),
                               self.V.ufl_element().degree())
        #BC info
        self.on_dirichlet = d.on_dirichlet
        self.on_neumann = d.on_neumann

        #I believe following version has issues with the stack
        #self.dirichlet = lambda t : DirichletBC(self.V, 
        #    interpolate(Expression(d.dirichlet(t), degree=d.deg), self.V),
        #    self.on_dirichlet)

        self.dirichlet = lambda t : Expression(d.dirichlet(t), degree=d.deg)

        self.neumann = lambda t : interpolate(Expression(d.neumann(t),
            degree=d.deg), self.V)

        self.analytic = lambda t : interpolate(Expression(d.analytic(t),
            degree=d.deg), self.V)
 
#        self.diff = interpolate(Expression(('0.0','0.0'), degree=d.deg),
#            self.V)
        self.diff = Function(self.V)
                
        self.body_forces = d.body_forces
        
        #define scalar fields
        self.rho = interpolate(Expression(d.rho, degree=d.deg), \
                               self.S)
        self.lmbda = interpolate(Expression(d.lmbda, degree=d.deg),  \
                                 self.S)
        self.mu = interpolate(Expression(d.mu, degree=d.deg), self.S)       
        #solution two steps ago
        self.u0 = interpolate(Expression(d.first, degree=d.deg), self.V)
 
        #previous solution
        self.u1 = interpolate(Expression(d.second, degree=d.deg), self.V)
  
        zero_expr = Expression('0.0', degree=d.deg)

        #initialize solution to first time step
        self.u = Function(self.V)
        assign(self.u, self.u0)

        #setup PML information
        self.aux_var = {'tau' : 
            [[zero_expr, zero_expr], [zero_expr, zero_expr]],
            'u' :
            [[zero_expr, zero_expr], [zero_expr, zero_expr]]
            }
  
        self.mod_var = {'tau' : 
            [[zero_expr, zero_expr], [zero_expr, zero_expr]],
            'u' :
            [[zero_expr, zero_expr], [zero_expr, zero_expr]]
            }
        self.tau = [[Expression('0.0', degree=self.deg), 
            Expression('0.0', degree=self.deg)],
            [Expression('0.0', degree=self.deg),
            Expression('0.0', degree=self.deg)]]
        self.damp = d.damp
        self.butter = d.butter
        self.shift = d.shift
  
        #time step initialization, based on cfl condition
        p_wave = interpolate(Expression('pow(%s + 2 * %s / %s, 0.5)'%(
                             d.lmbda, d.mu, d.rho),
                             degree=d.deg), self.S)
        h = self.mesh.hmin()
        c = max(p_wave.vector()[:])
        self.dt = d.cfl * h / c
        self.T = d.T
        self.t = 0.0
        self.dim = len(d.first)
        self.storage_dir = d.storage_dir

    def __aux_step(self, name, i1, i2, F):
        self.aux_var[name][i1][i2] = \
            Expression('(1 - dt * (alpha + d/beta)) * T_prev + %s'%(
                'd * dt / beta * F'), 
                alpha=self.shift[i1], 
                beta=self.butter[i1], 
                d=self.damp[i1], 
                F=F, 
                T_prev=self.aux_var[name][i1][i2],
                degree=self.deg,
                dt=self.dt)

    def __step_PML_u(self):
        dim = 2
        for i1 in range(0,dim):
            for i2 in range(0, dim):
                aux_step('u', i1, i2, self.u1)
                mod_field('u', i1, i2, self.u1)

    def __mod_field(self, lhs, F,T,coord):
#        assign(lhs, \
#            Expression("(F-T) / butter", \
#                F=F, T=T, butter=self.butter[coord], degree=self.deg))
        return Expression("(F-T) / butter", \
            F=F, T=T, butter=self.butter[coord], degree=self.deg)

    def __update_mods(self):
        time1 = time.time()
        dim = 2
        for i1 in range(0,dim):
            for i2 in range(0,dim):
                self.__aux_step('u', i1, i2, self.u.sub(i2))
#                self.__mod_field(self.mod_var['u'][i1][i2], 
#                    self.u1.sub(i1),
#                    self.aux_var['u'][i1][i2],
#                    i2)
                self.mod_var['u'][i1][i2] = self.__mod_field('ignore',
                    self.u1.sub(i1), 
                    self.aux_var['u'][i1][i2],
                    i2)
##                print(type(self.mod_var['u'][i1][i1]))
#                print(self.mod_var['u'][i1][i1].ufl_shape)
                c = plot(self.u1.sub(i1))
                plt.colorbar(c)
                plt.title('True solution')
                plt.show()
                c = plot(interpolate(self.aux_var['u'][i1][i2], self.S))
                plt.colorbar(c)
                plt.title('Auxiliary variable')
                plt.show()
                c = plot(interpolate(self.mod_var['u'][i1][i2], self.S))
                plt.title('Modified u')
                plt.colorbar(c)
                plt.show()
#                c = plot(interpolate(self.damp[i2], self.S))
#                plt.title('Damping')
#                plt.colorbar(c)
#                plt.show()
#                c = plot(interpolate(self.butter[i2], self.S))
#                plt.title('Butterworth Filter')
#                plt.show()
#                c = plot(interpolate(self.shift[i2], self.S))
#                plt.title('Shifting Filter')
#                plt.show()
#                exit(1)
                if( i1 == i2 ): 
                    s = "(lmbda + 2 * mu) * derivU"
                    self.tau[i1][i1] = Expression(s,
                        lmbda=self.lmbda,
                        mu=self.mu,
                        derivU=project(
                            Dx(project(self.mod_var['u'][i1][i1], self.S), 
                                i1),
                            self.S),
                        degree=1)
                else:
                    s = "mu * (u12 + u21)"
                    self.tau[i1][i2] = Expression(s,
                        mu=self.mu,
                        u12=project(
                            Dx(project(self.mod_var['u'][i1][i2], self.S),
                                i2),
                            self.S),
                        u21=project(
                            Dx(project(self.mod_var['u'][i2][i1], self.S),
                                i1),
                            self.S),
                        degree=self.deg)

                self.__aux_step('tau', i1, i2, self.tau[i1][i2])
#                self.__mod_field(self.mod_var['tau'][i1][i2], 
#                    self.tau[i1][i2],
#                    self.aux_var['tau'][i1][i2],
#                    i2)
                self.mod_var['tau'][i1][i2] = \
                    self.__mod_field('ignore',
                    self.tau[i1][i2], 
                    self.aux_var['tau'][i1][i2],
                    i2)

        self.tau_bar = Expression((('tau11', 'tau12'),
            ('tau21', 'tau22')), 
            tau11=self.mod_var['tau'][0][0],
            tau12=self.mod_var['tau'][0][1],
            tau21=self.mod_var['tau'][1][0],
            tau22=self.mod_var['tau'][1][1],
            degree=self.deg)
        print('mods updated: %s' % (time.time() - time1)) 
        

    def __sigma(self, uu):
        return self.lmbda * nabla_div(uu) * Identity(self.dim) + \
               self.mu * (grad(uu) + grad(uu).T)

    def __update_dirichlet(self):
        self.curr_dirichlet = DirichletBC(self.V, self.dirichlet(self.t),
            self.on_dirichlet)
 
    def __update_neumann(self):
        self.curr_neumann = self.neumann(self.t)
 
    def __update_analytic(self):
        self.curr_analytic = self.analytic(self.t)

    def __update_diff(self):
        self.diff.vector()[:] = self.u.vector()[:] - \
            self.curr_analytic.vector()[:]
        self.err = norm(self.diff, 'L2')
#        soln_norm = norm(self.u, 'L2')
#        if( soln_norm == 0 ):
#            self.err = 10000.0
#        else:
#            self.err = norm(self.diff, 'L2') / soln_norm
         
    def __update_body_forces(self):
        self.curr_body_forces = interpolate(Expression(
            self.body_forces(self.t), degree=self.deg), self.V)

    def __update_linear_form(self):
        self.__update_mods()
        v_test = TestFunction(self.V)
        self.lin_form = 2 * self.rho * inner(self.u1, v_test) * dx - \
                       self.rho * inner(self.u0, v_test) * dx - \
                       Constant(self.dt**2) * \
                       inner(self.tau_bar, \
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
        self.__update_diff()
        self.__update_body_forces()
        self.__update_linear_form()
 
    def __take_step(self):
        assign(self.u0, self.u1)
        assign(self.u1, self.u)
        A, b = assemble_system(self.B, self.lin_form, self.curr_dirichlet)
        
        solve(A, self.u.vector(), b)
        
    def __create_files(self):
        cr_file = lambda x : File('%s/%s'%(self.storage_dir, x))
        self.solver_files = [cr_file('x.pvd'), cr_file('y.pvd')]
#        self.analytic_files = [cr_file('x_analytic.pvd'), 
#            cr_file('y_analytic.pvd')]
#        self.diff_files = [cr_file('x_diff.pvd'), cr_file('y_diff.pvd')]
        self.u_tilde_files = [cr_file('u11.pvd'), cr_file('u12.pvd'), \
            cr_file('u21.pvd'), cr_file('u22.pvd')]
        self.tau_tilde_files = [cr_file('tau11.pvd'), \
            cr_file('tau12.pvd'), cr_file('tau21.pvd'), \
            cr_file('tau22.pvd')]
        if( self.dim == 3 ):
            self.solver_files.append('z.pvd')
            self.analytic_files.append('z_analytic.pvd')
            self.diff_files.append('z_diff.pvd')

    def __write_to_vtu(self):
        for i in range(0,len(self.solver_files)):
            self.solver_files[i] << (self.u.sub(i), self.t)
#            self.analytic_files[i] << (self.curr_analytic.sub(i),
#                self.t)
#            self.diff_files[i] << (self.diff.sub(i), self.t)
            self.tau_tilde_files[0] << (self.mod_var['tau'][0][0],
                self.t)
            self.tau_tilde_files[1] << (self.mod_var['tau'][0][1],
                self.t)
            self.tau_tilde_files[2] << (self.mod_var['tau'][1][0],
                self.t)
            self.tau_tilde_files[3] << (self.mod_var['tau'][1][1],
                self.t)
            self.u_tilde_files[0] << (self.mod_var['u'][0][0],
                self.t)
            self.u_tilde_files[1] << (self.mod_var['u'][0][1],
                self.t)
            self.u_tilde_files[2] << (self.mod_var['u'][1][0],
                self.t)
            self.u_tilde_files[3] << (self.mod_var['u'][1][1],
                self.t)
    def __clean_output(self, remove=False):
        if(remove and os.path.isdir(self.storage_dir)):
            os.system("rm -rf %s/*.vtu"%(self.storage_dir))
            os.system("rm -rf %s/*.pvd"%(self.storage_dir))
        os.system("sed -i \'\' \'s/UInt32/Int32/g\' %s"%(
            '%s/*.vtu'%(self.storage_dir)))
        os.system("sed -i \'\' \'s/f_[0-9]*-[0-9]*/f/g\' %s"%(
            '%s/*.vtu'%(self.storage_dir)))
 
    def __setup_and_plotIC(self):
        self.__clean_output(True)
        self.__create_files()
        self.__update()
        self.__write_to_vtu()
        assign(self.u, self.u1)
        self.t = self.dt
        self.__update()
        self.__write_to_vtu()
        self.__create_bilinear_form()
       
    def go(self, plot_interval, debug=True):
        if(debug):
            print('dt = %s'%(self.dt))
            print('Time steps = %s'%(round(self.T / self.dt)))
            os.system('sleep 2')
        self.__setup_and_plotIC()
        time_step = 2 

        def controlC_handler(sig,frame):
            print('You hit CTRL+C...cleaning output')
            self.__clean_output()
            exit(0)
         
        signal.signal(signal.SIGINT, controlC_handler) 
        while(self.t <= self.T - self.dt):
            time1 = time.time()
            self.__take_step()
            self.t += self.dt
            self.__update()       
            if( time_step % plot_interval == 0 ):
                self.__write_to_vtu()
            time2 = time.time()
            print('(step, exe_time,err) = (%s,%s,%s)'%(
                time_step, time2 - time1, self.err))
            time_step += 1
        self.__clean_output(False)

mesh = RectangleMesh(Point(0.0,0.0), Point(1.0, 1.0), 5, 5)
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)


