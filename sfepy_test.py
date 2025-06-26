# sfepy_crack_test.py
import meshio, pygmsh
from sfepy.discrete import Problem
from sfepy.base.base import Struct
import numpy as np

# 1) Mesh a rectangular plate with a central notch from your BW mask:
#    (Here we hardcode a simple notch)
import pygmsh
geom = pygmsh.opencascade.Geometry()
plate = geom.add_rectangle(0,100, 0,50, 0)
notch = geom.add_rectangle(49,51, 0,25, 0)
geom.boolean_difference([plate], [notch])
mesh = pygmsh.generate_mesh(geom, dim=2)
meshio.write('crack_plate.vtk', mesh)

# 2) Set up basic linear elasticity with displacement BCs
conf = Struct(
    filename_mesh='crack_plate.vtk',
    materials={'solid':({'D':210e3, 'nu':0.3},)},
    regions={'Omega':'all', 'Left':('vertices in (x<1)','facet'),
             'Right':('vertices in (x>99)','facet')},
    fields={'displacement':('real',2,'Omega',1)},
    variables={'u':('unknown','displacement',0),'v':('test','displacement','u')},
    ebcs={'fix_left':('Left',{'u.all':0.0}),'load_right':('Right',{'u.0':0.1})},
    integrals={'i':2},
    equations={'balance':"dw_lin_elastic.i.Omega(solid.D, v, u) = 0"},
)
pb = Problem.from_conf(conf)
pb.solve()

# 3) Extract opening displacement across the crack faces
u = pb.get_variables()['u'].get_state_parts()['u']
# Post-processing: sample u_y on nodes just above/below notch to compute width

print("Simulation complete; inspect VTK for COD field.")
