from __future__ import absolute_import, division, print_function
from shutil import copyfile
import math
import numpy as np
import dolfin as dl
import pickle
import ufl
from mshr import *
import sys
import os
sys.path.append( '../' )
from soupy import *
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.pyplot as plt
cm2 = cm.get_cmap('jet')
from scipy.optimize import fmin_l_bfgs_b

import shutil
try:
    shutil.rmtree('data')
except Exception:
    pass

output_dir = 'Results'
current_path = os.getcwd()
path = os.path.join(current_path, 'data/'+output_dir)
print(path)
os.makedirs(path)


betaV_input = 1.0
W_MC_input  = 1.0
betaR_input = 1e-4

meshR_input = 60
Ntr_input   = 25

corrlen_input =  0.02


############################################################################

def computeGammaDelta(corr_len, std, alpha=2, ndim=2):
    
    nu = alpha - 0.5*float(ndim)
    assert alpha > 0., "Alpha must be larger than ndim/2"
    kappa = math.sqrt(8.*nu)/corr_len
    gamma = math.sqrt(math.gamma(nu)/math.gamma(alpha))/(  math.pow( 4.0*math.pi, 0.25*float(ndim) )*math.pow(kappa, nu)*std   )
    delta = kappa*kappa*gamma

    return gamma, delta

def true_model(prior):
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    parRandom.normal(1., noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise,mtrue)
    return mtrue

################### 1. Define the Geometry ###########################
# 1. Define the Geometry
CornerRadius = 0.1
domain = Polygon(  [dl.Point(0.,0.),dl.Point(1.,0.),dl.Point(1.,.5),
                    dl.Point(.5+CornerRadius,.5),
                    dl.Point(.5+CornerRadius,.5+CornerRadius),
                    dl.Point(.5,.5+CornerRadius),
                    dl.Point(.5,1.),dl.Point(0.,1.),]  )
Circle = Circle(dl.Point(.5+CornerRadius,.5+CornerRadius), CornerRadius)

#define mesh
mesh = generate_mesh(domain-Circle, meshR_input)

comm = mesh.mpi_comm()
mpi_comm = mesh.mpi_comm()
if hasattr(mpi_comm, "rank"):
    mpi_rank = mpi_comm.rank
    mpi_size = mpi_comm.size
else:
    mpi_rank = 0
    mpi_size = 1

if mpi_rank == 0:
    fig = plt.figure(figsize=(6.4,4.8),dpi=200)
    dl.plot(mesh,linewidth=0.1)
    fig.savefig('data/mesh'+str(meshR_input)+'.png')
    plt.close(fig)
    with dl.XDMFFile('data/mesh'+str(meshR_input)+".xdmf") as fid:
        fid.write(mesh)


def saveXDMF(func, func_name, file):
    xf = dl.XDMFFile(comm, file)
    xf.write_checkpoint(func, func_name, 0, dl.XDMFFile.Encoding.HDF5, False)

#################### 2. Define optimization PDE problem #######################
v_el = dl.VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
ts_el = dl.FiniteElement("CG", mesh.ufl_cell(), 2)
tf_el = dl.FiniteElement("CG", mesh.ufl_cell(), 2)
Vh_STATE = dl.FunctionSpace(mesh, dl.MixedElement([v_el, ts_el, tf_el]))
Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
Vh_OPTIMIZATION = dl.FunctionSpace(mesh, "CG", 1)
Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_OPTIMIZATION]

class ColdSurf1(dl.SubDomain):
    def inside(self, x, on_boundary):
            innerleft    =  dl.near(x[0],.5) and x[1]>(.5+CornerRadius-dl.DOLFIN_EPS)
            innerbottom  =  dl.near(x[1],.5) and x[0]>(.5+CornerRadius-dl.DOLFIN_EPS)
            foo = innerleft or innerbottom
            return on_boundary and foo


class OuterLeft(dl.SubDomain):
    def inside(self, x, on_boundary):
            return on_boundary and dl.near(x[0],0.)
class OuterBottom(dl.SubDomain):
    def inside(self, x, on_boundary):
            return on_boundary and dl.near(x[1],0.)

class InnerLeft(dl.SubDomain):
    def inside(self, x, on_boundary):
            return on_boundary and x[1]>(0.5+CornerRadius-dl.DOLFIN_EPS) and dl.near(x[0],.5)
class InnerBottom(dl.SubDomain):
    def inside(self, x, on_boundary):
            return on_boundary and x[0]>(0.5+CornerRadius-dl.DOLFIN_EPS) and dl.near(x[1],.5)

class InnerCorner(dl.SubDomain):
    def inside(self, x, on_boundary):
            foo = (x[0]-0.5)**2 + (x[1]-0.5)**2
            status = foo < (CornerRadius+dl.DOLFIN_EPS)
            return on_boundary and   status

class UpSide(dl.SubDomain):
    def inside(self, x, on_boundary):
            return on_boundary and dl.near(x[1],1.)
class RightSide(dl.SubDomain):
    def inside(self, x, on_boundary):
            return on_boundary and dl.near(x[0],1.)

boundary_parts = dl.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundary_parts.set_all(0) #marks whole cube as domain 0
hotleft = OuterLeft()
hotleft.mark(boundary_parts, 1)
cold = ColdSurf1()
cold.mark(boundary_parts, 2)
hotbottom = OuterBottom()
hotbottom.mark(boundary_parts, 3)

cold2 = InnerCorner()
cold2.mark(boundary_parts, 4)

#Defining boundary conditions
ds=dl.Measure('ds',subdomain_data=boundary_parts)
disp_bc_upside = dl.DirichletBC(Vh[STATE].sub(0).sub(1), dl.Constant(0.0), UpSide())
disp_bc_rightside = dl.DirichletBC(Vh[STATE].sub(0).sub(0), dl.Constant(0.0), RightSide())
disp_bc0_upside = dl.DirichletBC(Vh[ADJOINT].sub(0).sub(1), dl.Constant(0.0), UpSide())
disp_bc0_rightside = dl.DirichletBC(Vh[ADJOINT].sub(0).sub(0), dl.Constant(0.0), RightSide())

disp_bc_innerleft = dl.DirichletBC(Vh[STATE].sub(0), dl.Constant((0.,0.)), InnerLeft())
disp_bc_innerbottom = dl.DirichletBC(Vh[STATE].sub(0), dl.Constant((0.,0.)), InnerBottom())
disp_bc0_innerleft = dl.DirichletBC(Vh[ADJOINT].sub(0), dl.Constant((0.,0.)), InnerLeft())
disp_bc0_innerbottom = dl.DirichletBC(Vh[ADJOINT].sub(0), dl.Constant((0.,0.)), InnerBottom())


disp_bc_innercorner = dl.DirichletBC(Vh[STATE].sub(0), dl.Constant((0.,0.)), InnerCorner())
disp_bc0_innercorner = dl.DirichletBC(Vh[ADJOINT].sub(0), dl.Constant((0.,0.)), InnerCorner())

bcs = [disp_bc_upside, disp_bc_rightside,
       disp_bc_innerleft, disp_bc_innerbottom,      disp_bc_innercorner]

bcs0 = [disp_bc0_upside, disp_bc0_rightside,
        disp_bc0_innerleft, disp_bc0_innerbottom,   disp_bc0_innercorner]


def param_interpolate(z,low=np.log(.1), high=np.log(10.), p=0.):
    return dl.Constant(low) + z/(  dl.Constant(1.)+dl.Constant(p)*(dl.Constant(1.)-z)  )  *(high-low)

def porosity(z,low=np.log(.1/(1.-.1)), high=np.log(.9/(1.-.9)), p=0.):
        z = dl.Constant(1.)-z
        return dl.Constant(low) + z/(  dl.Constant(1.)+dl.Constant(p)*(dl.Constant(1.)-z)  )  *(high-low)
def sigmoid(x):
    return dl.Constant(1.)/( dl.Constant(1.) + ufl.exp(-x) )


def epsilon(u):return ufl.sym(ufl.grad(u))
def sigma(u,E):
    return dl.Constant(0.5769)*E*ufl.div(u)*ufl.Identity(2) + dl.Constant(2.*0.3846)*E*epsilon(u)


def bilinear_thermo(us,ps,  uf,pf,  porosity):

    h = dl.Constant(81059.)

    phi_f = porosity
    phi_s = dl.Constant(1.) - porosity

    Ta_solid = ufl.inner( phi_s * dl.Constant(0.477) * ufl.grad(us), ufl.grad(ps) )     *dl.dx
    Ta_fluid = ufl.inner( phi_f * dl.Constant(0.085) * ufl.grad(uf), ufl.grad(pf) )     *dl.dx

    Ta_solid += h * (us-uf) * ps     *dl.dx
    Ta_fluid -= h * (us-uf) * pf     *dl.dx

    Ta_solid_hot1  = phi_s * us * ps                        *ds(1)
    Ta_solid_hot2  = phi_s * us * ps                        *ds(3)
    Ta_solid_cold  = phi_s * us * ps                        *ds(2)

    Ta_fluid_hot1  = phi_f * uf * pf                        *ds(1)
    Ta_fluid_hot2  = phi_f * uf * pf                        *ds(3)
    Ta_fluid_cold  = phi_f * uf * pf                        *ds(2)


    A = Ta_solid + Ta_fluid
    A += Ta_solid_hot1 + Ta_solid_hot2 + Ta_solid_cold
    A += Ta_fluid_hot1 + Ta_fluid_hot2 + Ta_fluid_cold

    return A

def linear_thermo(us,ps,  uf,pf,  porosity):
    phi_f = porosity
    phi_s = dl.Constant(1.) - porosity

    Tl_solid_hot1  =  phi_s * ps*dl.Constant(1.)  *ds(1)
    Tl_solid_hot2  =  phi_s * ps*dl.Constant(1.)  *ds(3)
    Tl_solid_cold  =  phi_s * ps*dl.Constant(0.)  *ds(2)

    Tl_fluid_hot1  =  phi_f * pf*dl.Constant(1.)  *ds(1)
    Tl_fluid_hot2  =  phi_f * pf*dl.Constant(1.)  *ds(3)
    Tl_fluid_cold  =  phi_f * pf*dl.Constant(0.)  *ds(2)

    L  = Tl_solid_hot1 + Tl_solid_hot2 + Tl_solid_cold
    L += Tl_fluid_hot1 + Tl_fluid_hot2 + Tl_fluid_cold
    return  L




def bilinear_mech(u,p, materialprop):
    return ufl.inner( sigma(u,materialprop), epsilon(p) )     *dl.dx

def linear_mech(u,p):
    tract_left = ufl.inner(dl.Constant((.1,0.)),p)    *ds(1)
    tract_bottom = ufl.inner(dl.Constant((0.,.1)),p)    *ds(3)
    return (tract_left+tract_bottom)


def residual(u, m, p, z):
    D,Ts, Tf      = ufl.split(u)
    pD,pTs, pTf  = ufl.split(p)

    poro = sigmoid(porosity(z)+m)

    E = ufl.exp(param_interpolate(z)+m)

    thermo = bilinear_thermo(Ts,pTs, Tf,pTf, poro) - linear_thermo(Ts,pTs, Tf,pTf, poro)
    mechan = bilinear_mech(D,pD, E) - linear_mech(D,pD)

    return thermo + mechan


pde = ControlPDEProblem(Vh, residual, bcs, bcs0, is_fwd_linear=True)

################## 3. Define the quantity of interest (QoI) ############
def MyObjectives(u,m,z):
    D,Ts, Tf      = ufl.split(u)
    poro = sigmoid(porosity(z)+m)

    E = ufl.exp(param_interpolate(z)+m)

    thermo = dl.Constant(.5)*bilinear_thermo(Ts,Ts, Tf,Tf, poro) + linear_thermo(Ts,Ts, Tf,Tf, poro)
    mechan = dl.Constant(.5)*bilinear_mech(D,D, E) + linear_mech(D,D)
    return thermo, mechan


from myqoi import QoI
qoi = QoI(mesh, Vh, MyObjectives, W_MC_input)

################## 4. Define Penalization term ############################

penalization = H1Penalization(Vh[OPTIMIZATION], dl.dx, betaR_input)


################## 5. Define the prior ##################### ##############
def MyBiLaplacePrior(corr_len, sigma, theta_x, theta_y, phi  = math.pi/2.):

    anis_diff = dl.Constant( [[theta_x,0.],[0., theta_y]]  )

    gamma, delta = computeGammaDelta(corr_len,  sigma )
    prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, Theta = anis_diff)
    return delta, gamma, prior


theta_x = 1.
theta_y = 1E-2
marg_std = 0.1
delta, gamma, prior = MyBiLaplacePrior(corrlen_input   , marg_std, theta_x, theta_y)




############ 0 parameters to config ##############################
plotFlag = False
check_gradient = False
check_hessian = False

correction = False

if correction:
    parameter = pickle.load(open('data/parameter.p', 'rb'))
    parameter["correction"] = correction
else:
    parameter = dict()
    # with Monte Carlo correction or not
    parameter["correction"] = False
    # optimization method
    parameter["optMethod"] = 'scipy_bfgs'
    # number of eigenvalues for trace estimator
    parameter['N_tr'] = Ntr_input
    parameter['N_mc'] = 1
    # regularization coefficient
    parameter['alpha'] =  betaR_input  # alpha*R(z)
    # variance coefficient
    parameter['beta'] = betaV_input     # E[Q] + beta Var[Q]
    # prior covariance, correlation length prop to gamma/delta
    parameter['delta'] = delta    # delta*Identity
    parameter['gamma'] = gamma    # gamma*Laplace
    parameter['theta_x'] = theta_x
    parameter['theta_y'] = theta_y
    parameter['marg_std'] = marg_std
    parameter['corr_len'] = corrlen_input

    parameter["dim"] = 1
    parameter["optDimension"] = mesh.num_vertices()
    parameter["bounds"] = [(0., 1.0) for i in range(parameter["optDimension"])]

    pickle.dump(parameter, open('data/parameter.p','wb'))



##########################################################################################

def savedata(type):

    x_fun = vector2Function(cost.x, Vh_STATE) 
    m_fun = vector2Function(cost.m, Vh_PARAMETER, name="m")
    p_fun = vector2Function(cost.y, Vh[ADJOINT]) 
    z_fun = vector2Function(cost.z, Vh_OPTIMIZATION,name="design")
    D,Ts, Tf      = x_fun.split(deepcopy= True)
    
    
    
    saveXDMF(z_fun, "Optimal Design", "data/"+type+"/d_opt.xdmf")
    saveXDMF(m_fun, "m", "data/"+type+"/m.xdmf")
    file = dl.File('data/D.pvd')
    file << D
    file = dl.File('data/TS.pvd')
    file << Ts
    file = dl.File('data/TF.pvd')
    file << Tf
    E = ufl.exp(param_interpolate(z_fun)+m_fun)
    d= D.geometric_dimension()
    

    if mpi_rank == 0:
        print("optimization result", opt_result)
        print("func_ncalls, grad_ncalls, hess_calls = ", cost.func_ncalls, cost.grad_ncalls, cost.hess_ncalls)

        data = dict()
        data['opt_result'] = opt_result
        data['tobj'] = cost.tobj
        data['tgrad'] = cost.tgrad
        data['trand'] = cost.trand
        data['func_ncalls'] = cost.func_ncalls
        data['grad_ncalls'] = cost.grad_ncalls
        data['hess_ncalls'] = cost.hess_ncalls

        data['lin_mean'] = cost.lin_mean
        data['lin_diff_mean'] = cost.lin_diff_mean
        data['lin_fval_mean'] = cost.lin_fval_mean
        data['lin_var'] = cost.lin_var
        data['lin_diff_var'] = cost.lin_diff_var
        data['lin_fval_var'] = cost.lin_fval_var

        data['quad_mean'] = cost.quad_mean
        data['quad_var'] = cost.quad_var
        data['quad_diff_mean'] = cost.quad_diff_mean
        data['quad_fval_mean'] = cost.quad_fval_mean
        data['quad_diff_var'] = cost.quad_diff_var
        data['quad_fval_var'] = cost.quad_fval_var

        pickle.dump( data, open( "data/"+type+"/data.p", "wb" ) )

        # copyfile("iterate.dat", "data/"+type+"/iterate.dat")


################# 7. Solve the optimization algorithms #####################
maxiter = 250

def initial_guess(z_fun):
    saveXDMF(z_fun,"Initial Design", 'data/initial_z.xdmf')

    z = z_fun.vector()


    return z

def optimization(cost, z_fun):
    z = initial_guess(z_fun)


    if check_gradient:
        zfoo = dl.interpolate(dl.Expression("sin(x[0])", degree=5), Vh[OPTIMIZATION])
        for foo in range(10):
            cost.checkGradient(zfoo.vector(), mpi_comm, plotFlag=plotFlag, id=foo)
    if check_hessian:
        zfoo = dl.interpolate(dl.Expression("sin(x[0])", degree=5), Vh[OPTIMIZATION])
        for foo in range(10):
            cost.checkHessian(zfoo.vector(), mpi_comm, plotFlag=plotFlag, id=foo)


    opt_result = fmin_l_bfgs_b(func=cost.costValue, x0=z, fprime=cost.costGradient,
                    disp=99, pgtol=1e-12, maxiter=maxiter, factr=10.0, bounds=cost.parameter["bounds"], iprint=99, maxls=50)

    return cost, opt_result


cost = CostFunctionalQuadratic(parameter, Vh, pde, qoi, prior, penalization, tol=1e-10)

z_fun = dl.project(dl.Constant(1.),Vh[OPTIMIZATION])
cost, opt_result = optimization(cost,z_fun)
savedata(output_dir)
