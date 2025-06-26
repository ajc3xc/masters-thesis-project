#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────
# 1) MASK → MESH (Gmsh via Python API)
# ────────────────────────────────────────────────────────────
from skimage.io       import imread
from skimage.measure  import label, regionprops, find_contours
from shapely.geometry import Polygon, LinearRing
import gmsh, meshio

# path to your binary crack mask (white=crack)
BW_PATH = "mask.png"
MESH_DIR = "mesh_out"
os.makedirs(MESH_DIR, exist_ok=True)
MSH_PATH  = os.path.join(MESH_DIR, "crack_domain.msh")
XDMF_PATH = os.path.join(MESH_DIR, "crack_domain.xdmf")

# load & threshold
mask = imread(BW_PATH, as_gray=True) > 0.5

# keep only the largest connected component
lbl   = label(mask)
props = sorted(regionprops(lbl), key=lambda r: r.area, reverse=True)
mask  = (lbl == props[0].label)

# extract the outer contour
contours = find_contours(mask.astype(float), 0.5)
contour  = max(contours, key=lambda c: c.shape[0])
pts      = [(float(p[1]), float(p[0])) for p in contour]

# clean & decimate with shapely
poly  = Polygon(pts).buffer(0)
poly  = poly.simplify(1.0, preserve_topology=True)
shell = list(poly.exterior.coords)
if not LinearRing(shell).is_ccw:
    shell.reverse()

# build Gmsh geometry
gmsh.initialize()
gmsh.model.add("crack")

# add points & lines
p_tags = [gmsh.model.geo.addPoint(x, y, 0, meshSize=5) for x,y in shell]
l_tags = []
n_pts  = len(p_tags)
for i in range(n_pts):
    l_tags.append(gmsh.model.geo.addLine(p_tags[i], p_tags[(i+1)%n_pts]))

gmsh.model.geo.addCurveLoop(l_tags, tag=1)
gmsh.model.geo.addPlaneSurface([1], tag=1)
gmsh.model.geo.synchronize()

# mesh 2D
gmsh.model.mesh.generate(2)
gmsh.write(MSH_PATH)
gmsh.finalize()

# convert to XDMF for FEniCS
m = meshio.read(MSH_PATH)
tri = [c for c in m.cells if c.type=="triangle"][0].data
meshio.write(XDMF_PATH, meshio.Mesh(points=m.points, cells={"triangle":tri}))

print("✅ Mesh written to", XDMF_PATH)


# ────────────────────────────────────────────────────────────
# 2) PHASE‐FIELD FRACTURE SOLVER (FEniCS)
# ────────────────────────────────────────────────────────────
from dolfin import *

# -- concrete defaults (bridge concrete) --
E_modulus      = Constant(30e3)   # [MPa]
nu             = 0.2
Gc             = Constant(100.0)  # [N/m] ~ [MPa·mm]
sigma_t        = Constant(3.0)    # [MPa]
ell            = Constant(0.02)   # [mm]
kappa          = Constant(1e-6)   # residual stiffness

# read mesh
mesh = Mesh()
with XDMFFile(XDMF_PATH) as f:
    f.read(mesh)

# function spaces
V_u = VectorFunctionSpace(mesh, "CG", 1)
V_d = FunctionSpace(mesh, "CG", 1)
W   = MixedFunctionSpace([V_u, V_d])

# trial & test
w    = Function(W)
(u, d) = split(w)
(vu, vd) = TestFunctions(W)

# material constants
mu     = E_modulus/(2*(1+nu))
lmbda  = E_modulus*nu/((1+nu)*(1-2*nu))

# strain energy (no tension/compression split for simplicity)
def psi_elastic(u):
    eps = sym(grad(u))
    return 0.5*(2*mu*inner(eps,eps) + lmbda*(tr(eps)**2))

# degradation
g_d = (1-d)**2 + kappa

# total energy
psi = g_d*psi_elastic(u)*dx \
    + Gc*(d**2/(2*ell) + ell*inner(grad(d),grad(d))/2)*dx

# variational form
F = derivative(psi, w, TestFunction(W))

# boundary conditions: left edge fixed, right edge small opening in y
tol = 1e-6
xmax = mesh.coordinates()[:,0].max()
left  = CompiledSubDomain("near(x[0], 0.0, tol)", tol=tol)
right = CompiledSubDomain("near(x[0], xmax, tol)", tol=tol, xmax=xmax)
bc_u0 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), left)
# impose a tiny opening δ=0.1 mm in +y at the right boundary
bc_u1 = DirichletBC(W.sub(0), Constant((0.0, 0.1)), right)
bc_d  = DirichletBC(W.sub(1), Constant(0.0), left)  # crack can't heal

bcs = [bc_u0, bc_u1, bc_d]

# solve
problem = NonlinearVariationalProblem(F, w, bcs)
solver  = NonlinearVariationalSolver(problem)
prm     = solver.parameters
prm["newton_solver"]["relative_tolerance"] = 1e-6
solver.solve()

# split solutions
u_, d_ = w.split()

# save fields
File("u_phasefield.pvd") << u_
File("d_phasefield.pvd") << d_

print("✅ Phase‐field solve complete")


# ────────────────────────────────────────────────────────────
# 3) EXTRACT CRACK‐OPENING PROFILE ALONG THE SKELETON
# ────────────────────────────────────────────────────────────
from skimage.morphology import skeletonize

# build skeleton of mask
skel = skeletonize(mask)

# gather skeleton coordinates
ys, xs = np.nonzero(skel)
pts    = list(zip(ys, xs))

# build adjacency to find main path
from collections import defaultdict, deque
nbrs = defaultdict(list)
dirs = [(-1,0),(1,0),(0,-1),(0,1)]
sset = set(pts)
for y,x in pts:
    for dy,dx in dirs:
        if (y+dy,x+dx) in sset:
            nbrs[(y,x)].append((y+dy,x+dx))

# find two farthest endpoints → main crack
endpts = [p for p,n in nbrs.items() if len(n)==1]
src     = endpts[0]
# BFS to get predecessors
pred = {src: None}
q    = deque([src])
while q:
    u = q.popleft()
    for v in nbrs[u]:
        if v not in pred:
            pred[v] = u
            q.append(v)
# farthest
far = max(endpts, key=lambda e: (e[0]-src[0])**2 + (e[1]-src[1])**2)
# backtrack
path = []
cur  = far
while cur is not None:
    path.append(cur)
    cur = pred[cur]
path = path[::-1]

# for each skeleton point, approximate normal by finite difference along path
# and measure opening w = (u_+(n) - u_-(n))·n
widths = []
dists  = [0.0]
for i,pt in enumerate(path):
    y,x = pt
    if i>0:
        y0,x0 = path[i-1]
        dists.append(dists[-1] + np.hypot(x-x0, y-y0))
    # local tangent
    if 0<i<len(path)-1:
        y1,x1 = path[i-1]
        y2,x2 = path[i+1]
        t = np.array([x2-x1, y2-y1],float)
        t /= np.linalg.norm(t)
        # normal
        n = np.array([-t[1], t[0]])
    else:
        n = np.array([0,1])
    # sample points ±r along normal
    r = 1  # px sampling distance
    p_minus = Point(x - r*n[0], y - r*n[1])
    p_plus  = Point(x + r*n[0], y + r*n[1])
    # eval U at those
    try:
        u_minus = u_(p_minus)
        u_plus  = u_(p_plus)
        w_open  = float((u_plus[1] - u_minus[1]))  # opening in y
    except:
        w_open = np.nan
    widths.append(w_open)

dists  = np.array(dists)
widths = np.array(widths)

# save CSV
np.savetxt("crack_opening_phasefield.csv",
           np.column_stack([dists, widths]),
           header="distance_mm,opening_mm", delimiter=",")

# quick plot
plt.figure(figsize=(8,4))
plt.plot(dists, widths, "-o", ms=3, label="phase‐field opening")
plt.xlabel("Distance along crack (px)")
plt.ylabel("Opening (mm)")
plt.title("Crack opening profile (phase‐field)")
plt.legend()
plt.tight_layout()
plt.savefig("phasefield_opening_profile.png", dpi=300)
plt.show()

print("✅ Crack‐opening profile saved")
