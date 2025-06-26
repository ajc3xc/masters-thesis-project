# fenics_crack_test.py
from dolfin import *
import mshr
import numpy as np
import matplotlib.pyplot as plt

# 1) Geometry: plate with a central notch
domain = mshr.Rectangle(Point(0,0), Point(100,50)) \
       - mshr.Rectangle(Point(49,0), Point(51,25))
mesh = mshr.generate_mesh(domain, 64)

# 2) Define function space & BCs
V = VectorFunctionSpace(mesh, 'P', 1)
u = TrialFunction(V); v = TestFunction(V)
E, nu = 210e3, 0.3
mu = E/2/(1+nu); lmbda = E*nu/(1+nu)/(1-2*nu)
def sigma(u): return 2*mu*sym(grad(u)) + lmbda*tr(sym(grad(u)))*Identity(2)
a = inner(sigma(u), sym(grad(v))) * dx
from dolfin import dot, Constant, dx

f = Constant((0.1, 0.0))         # traction or body‐force vector
L = dot(f, v) * dx               # scalar form for the right‐hand side  


# BCs: left fixed, right displacement
bc1 = DirichletBC(V, Constant((0,0)), 'near(x[0], 0)')
bc2 = DirichletBC(V.sub(0), Constant(0.1), 'near(x[0], 100)')
u_sol = Function(V)
solve(a == L, u_sol, [bc1, bc2])

# 3) Compute COD: sample u_y above/below crack line
# (e.g. integrate jump across y=25 line)
print("FEniCS simulation done; export u_sol for COD analysis.")
