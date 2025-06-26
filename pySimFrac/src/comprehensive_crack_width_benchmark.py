#!/usr/bin/env python3
"""
Comprehensive Crack Width Benchmarking
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
This script runs four crack‐width estimation methods side by side:
  1. Medial‐Axis + DSE Pruning (CuCIM + dsepruning)
  2. LEFM √r Tip‐Fit along the pruned skeleton
  3. Synthetic Aperture Field (pySimFrac)
  4. (Placeholder) Phase‐Field Fracture Simulation (SfePy)

Material & model defaults for “typical bridge concrete”:
  • Young’s modulus E = 30 GPa  (20–40 GPa)            [fib Model Code 2010]
  • Poisson’s ratio ν = 0.2  (0.10–0.20)               [ACI, fib MC]
  • Fracture energy Gf = 100 N/m  (~100 J/m²)         [fib MC formula]
  • Tensile strength σₜ = 3 MPa  (2–5 MPa)            [standard concrete data]
  • Phase‐field length ℓ = 0.02 mm                     [typical PF studies]
  
References:
  • fib Model Code 2010: https://www.fib-international.org  
  • Concrete tensile strength: ACI 318  
  • Phase‐field examples: https://sfepy.org/doc-devel/examples/phase_field.html  
  • pySimFrac intro: https://lanl.github.io/pySimFrac/intro.html
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 0. Paths & Parameters
# ------------------------------------------------------------------------------
MASK_PATH   = "/mnt/e/camerer_ml/datasets/METU_Concrete_Crack_Seg/BW_visible/236.png"        # Path to your binary crack mask (BW image)
OUTPUT_DIR  = "width_benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Concrete defaults (bridge concrete)
E_modulus      = 30e3   # Young’s modulus [N/mm²] = 30 GPa
nu             = 0.2    # Poisson’s ratio
fracture_energy= 100.0  # Fracture energy Gf [N/m]
tensile_strength = 3.0  # Tensile strength σₜ [MPa]
phase_length   = 0.02   # Phase‐field length ℓ [mm]

# Medial‐axis / DSE settings
MIN_AREA_PX = 1000
THRESHOLD   = 0.5

# LEFM fit settings
from scipy.optimize import curve_fit
def lefm_model(r, C): return C * np.sqrt(r)

# pySimFrac settings
from pysimfrac.src.general.simFrac import SimFrac
PSF_h        = 0.01    # grid spacing [mm]
PSF_lx, PSF_ly = 3.0, 1.0   # domain size [mm]
PSF_nx, PSF_ny = 300, 100   # grid resolution
PSF_mean_ap = 1.0    # mean aperture [mm]
PSF_rough   = 0.01   # RMS roughness [mm]
PSF_Hurst   = 0.5    # Hurst exponent

# ------------------------------------------------------------------------------
# 1. Medial‐Axis + DSE Width Map
# ------------------------------------------------------------------------------
#from skimage.io import imread
from cucim.skimage.morphology import medial_axis
import cupy as cp
from dsepruning.dsepruning import skel_pruning_DSE

# Load & binarize
from PIL import Image
import numpy as np

img_gray = np.array(Image.open(MASK_PATH).convert('L'))
bw = img_gray > 40

# Compute medial axis + distance on GPU
sk_gpu, dist_gpu = medial_axis(cp.array(bw), return_distance=True)
sk, dist = sk_gpu.get(), dist_gpu.get()

# DSE prune
pruned = skel_pruning_DSE(sk, dist, min_area_px=MIN_AREA_PX, return_graph=False)

# Width map (px)
width_map_px = np.zeros_like(bw, float)
width_map_px[pruned] = dist[pruned] * 2

# Plot & save
plt.figure(figsize=(6,6))
plt.imshow(bw, cmap='gray', alpha=0.5)
ys, xs = np.nonzero(pruned)
plt.scatter(xs, ys, c=width_map_px[ys, xs], s=1, cmap='plasma')
plt.colorbar(label='Width (px)')
plt.title('1. Medial‐Axis + DSE Width')
plt.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, "medial_dse_width.png"), dpi=300)
plt.close()
print("medial done")

# -------------------------------------------------------------------------------
# 2. LEFM √r Fit Along Pruned Skeleton (unchanged except for Cval fix)
# -------------------------------------------------------------------------------
from collections import defaultdict, deque

# (re-use your ys, xs coords from the medial+DSE step)
coords = list(zip(ys, xs))
nbrs = defaultdict(list)
offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
coord_set = set(coords)
for y,x in coords:
    for dy,dx in offsets:
        if (y+dy, x+dx) in coord_set:
            nbrs[(y,x)].append((y+dy, x+dx))

# pick a tip (end-point of the skeleton graph)
endpoints = [pt for pt,n in nbrs.items() if len(n)==1]
tip = endpoints[0] if endpoints else coords[0]

# BFS to order
visited, order = set(), []
q = deque([tip])
while q:
    pt = q.popleft()
    if pt in visited: continue
    visited.add(pt)
    order.append(pt)
    for nb in nbrs[pt]:
        if nb not in visited:
            q.append(nb)

# distances and widths aligned
px2mm = 1.0
dists = [0.0]
for (y0,x0),(y1,x1) in zip(order[:-1], order[1:]):
    dists.append(dists[-1] + np.hypot(y1-y0, x1-x0)*px2mm)
dists = np.array(dists[:-1])  # drop last so same length as widths

widths = np.array([width_map_px[y0,x0]*px2mm for (y0,x0) in order[:-1]])

# LEFM fit
mask_fit = (dists>0)&np.isfinite(widths)&(widths>0)
if mask_fit.sum()>5:
    C_fit, _ = curve_fit(lefm_model, dists[mask_fit], widths[mask_fit])
    Cval = float(C_fit[0])
else:
    Cval = np.nan
w_fit = lefm_model(dists, Cval)

# Plot LEFM
plt.figure(figsize=(8,4))
plt.scatter(dists, widths, s=3, alpha=0.4, label='Measured')
plt.plot(dists, w_fit, 'r-', lw=2, label=f'LEFM fit: C={Cval:.2f}')
plt.xlabel("Distance from tip (mm)")
plt.ylabel("Width (mm)")
plt.title("2. LEFM √r Crack Opening Fit")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lefm_fit.png"), dpi=300)
plt.close()
print("lefm done")


# ────────────────────────────────────────────────────────────────────────────────
# 2b. Simple “Profile‐Normal on Main Path” (no gradient threshold, just mask hits)
# ────────────────────────────────────────────────────────────────────────────────
import networkx as nx
from sklearn.decomposition import PCA

def extract_longest_path(pruned):
    """Extract the tip-to-tip longest path of the largest connected skeleton."""
    from skimage.morphology import label
    lbl, n = label(pruned, return_num=True)
    sizes = [(lbl==i).sum() for i in range(1,n+1)]
    main = np.argmax(sizes)+1
    sk = (lbl==main)

    # build graph
    G = nx.Graph()
    ys,xs = np.nonzero(sk)
    for y,x in zip(ys,xs):
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            yy,xx = y+dy, x+dx
            if 0<=yy<sk.shape[0] and 0<=xx<sk.shape[1] and sk[yy,xx]:
                G.add_edge((y,x),(yy,xx))

    # find diameter endpoints
    def far(u):
        dist = nx.single_source_shortest_path_length(G, u)
        return max(dist, key=dist.get)
    # start from any endpoint
    ends = [n for n,d in G.degree() if d==1]
    if len(ends)<2:
        return np.array(list(G.nodes()))
    u1 = far(ends[0])
    u2 = far(u1)
    path = nx.shortest_path(G, u1, u2)
    return np.array(path)


def profile_normal_simple(bw, path, max_dist=200):
    """
    For each (y,x) on 'path', find the PCA normal, then march
    +/- along it until you leave the mask.  Return dist_along_path,
    widths, and endpoint pairs.
    """
    d_along = [0.0]
    widths = []
    ends = []

    # rolling distance
    for i in range(1, len(path)):
        y0,x0 = path[i-1]
        y1,x1 = path[i]
        d_along.append(d_along[-1] + np.hypot(y1-y0, x1-x0))

    d_along = np.array(d_along)

    # for each point, compute local normal via PCA on a tiny window of skeleton
    for i,(y,x) in enumerate(path):
        # collect neighbors in window
        win = 5
        ys,xs = np.nonzero(bw[max(0,y-win):y+win+1, max(0,x-win):x+win+1])
        if len(ys)<3:
            normal = np.array([0,1])
        else:
            pts = np.column_stack((ys + max(0,y-win),
                                   xs + max(0,x-win)))
            pca = PCA(n_components=2).fit(pts)
            tangent = pca.components_[0]
            normal = np.array([-tangent[1], tangent[0]])
            normal /= np.linalg.norm(normal)

        hits = []
        for sign in (+1,-1):
            for k in range(1, max_dist):
                yy = int(round(y + sign*normal[0]*k))
                xx = int(round(x + sign*normal[1]*k))
                if not (0<=yy<bw.shape[0] and 0<=xx<bw.shape[1]):
                    # off-image → treat as boundary
                    hits.append((yy,xx))
                    break
                if not bw[yy,xx]:
                    hits.append((yy,xx))
                    break

        if len(hits)==2:
            w = np.hypot(hits[0][0]-hits[1][0],
                         hits[0][1]-hits[1][1])
        else:
            w = np.nan
        widths.append(w)
        ends.append(hits)

    return d_along, np.array(widths), ends


# --- invoke and plot ---
path = extract_longest_path(pruned)
d2, w2, ends2 = profile_normal_simple(bw, path, max_dist=200)
print(f"Measured simple profile on path of length {len(path)} points.")

# Plot the scatter of width vs distance
# And the ribbon map (robust to bad ends2 entries)
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(bw, cmap='gray', alpha=0.5)

# Filter out cases where ends2 isn't two points, or width is not finite
valid = []
for e, ww in zip(ends2, w2):
    if ww is not None and np.isfinite(ww) and isinstance(e, (list, tuple)) and len(e)==2:
        valid.append((e, ww))

if not valid:
    raise RuntimeError("No valid normals to plot in ribbons_simple.png!")

ends2_filt, w2_filt = zip(*valid)
norm = plt.Normalize(np.nanmin(w2_filt), np.nanmax(w2_filt))

for ((y1,x1),(y2,x2)), ww in zip(ends2_filt, w2_filt):
    ax.plot([x1,x2],[y1,y2], color=plt.cm.plasma(norm(ww)), lw=2)

sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Width (px)')
ax.axis('off')
ax.set_title("2b. Profile‐Normal Ribbons (mask hits)")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "ribbons_simple.png"), dpi=400)
plt.close()

print("simple profile ribbons done.")

# ------------------------------------------------------------------------------
# 3. Synthetic Aperture Field (pySimFrac)
# ------------------------------------------------------------------------------
sf = SimFrac(
    h=PSF_h, lx=PSF_lx, ly=PSF_ly,
    nx=PSF_nx, ny=PSF_ny,
    method='spectral', units='mm'
)
sf.params["mean-aperture"]["value"] = PSF_mean_ap
sf.params["roughness"]["value"]    = PSF_rough
sf.params["H"]["value"]            = PSF_Hurst
sf.create_fracture()
ap = sf.aperture  # 2D array in mm

# Plot the field
plt.figure(figsize=(6,3))
plt.imshow(ap, cmap='viridis', aspect='auto')
plt.colorbar(label='Aperture (mm)')
plt.title('3. pySimFrac Synthetic Aperture')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pysimfrac_field.png"), dpi=300)
plt.close()

# Extract & plot centerline profile
profile = ap[ap.shape[0]//2, :]
x_mm = np.linspace(0, PSF_lx, len(profile))
plt.figure(figsize=(8,3))
plt.plot(x_mm, profile, lw=1.5, label='pySimFrac centerline')
plt.xlabel("Position (mm)")
plt.ylabel("Aperture (mm)")
plt.title("3. pySimFrac Centerline Profile")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pysimfrac_profile.png"), dpi=300)
plt.close()
print("pysimfrac done")

# ------------------------------------------------------------------------------
# 4. Phase‐Field Fracture Placeholder (SfePy)
# ------------------------------------------------------------------------------
# NOTE: A full PF simulation requires non-linear solve & damage field. 
# Here we only set up material & mesh placeholders.
try:
    from sfepy.discrete.fem import Mesh, FEDomain, Field
    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
    from sfepy.terms import Term
    from sfepy.discrete import Problem
    from sfepy.base.base import Struct

    # Load or generate a simple mesh
    mesh = Mesh.from_file('meshes/2d/square_quad.mesh')
    domain = FEDomain('domain', mesh)
    omega = domain.create_region('Omega', 'all')
    field = Field.from_args('fu', np.float64, 'vector', omega, approx_order=1)

    # Material stiffness
    D = stiffness_from_youngpoisson(2, E_modulus, nu)
    mat = {'name':'solid', 'values':{'D': D}}
    term = Term.new('dw_lin_elastic(solid.D, v, u)', 'i', omega,
                    solid=mat, v='v', u='u')
    equations = {'balance': term}

    # Problem setup (no actual crack growth)
    pb = Problem('elasticity', equations=equations)
    pb.set_bcs(ebcs=Struct(name='fix_left', region={'name':'Left','select':'vertices in x<1e-6'}, dofs={'u.all':0.0}))
    # pb.set_bcs(epbcs=...)  # apply loads here
    state = pb.solve()

    # Placeholder message
    print("4. SfePy elasticity solve complete (no PF).")
except Exception as e:
    print("4. Phase‐Field placeholder skipped:", e)

print("\nAll outputs saved to:", OUTPUT_DIR)
print("sfepy placeholder completed (if available).")
