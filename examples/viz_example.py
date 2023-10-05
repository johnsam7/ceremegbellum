

import numpy as np
from cmb import MLabEmulator

print("Creating instance of viz wrapper...")

mlab = MLabEmulator()

print("Creating figure...")
mlab.figure(bgcolor=(1., 1., 1.),
            fgcolor=(0., 0., 0.),
            size=(1200, 1200))

print("Making bogus data...")
n = 8
t = np.linspace(-np.pi, np.pi, n)
z = np.exp(1j * t)
x = z.real.copy()
y = z.imag.copy()
z = np.zeros_like(x)

triangles = [(0, i, i + 1) for i in range(1, n)]
x = np.r_[0, x]
y = np.r_[0, y]
z = np.r_[1, z]
t = np.r_[0, t]

figures = list()

print("Creating meshes...")
figures.append(mlab.triangular_mesh(x, y, z, triangles, scalars=t, colormap='bwr'))

t2 = [3 for _ in t]
mlab.figure(bgcolor=(1., 1., 1.),
            fgcolor=(0., 0., 0.),
            size=(1200, 1200))

figures.append(mlab.triangular_mesh(x, y, z, triangles, scalars=t2, colormap='bwr'))
print("Setting colorbar...")
mlab.colorbar()

input("Press 'Return' to exit...")

print("Done!")
