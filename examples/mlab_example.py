import numpy as np
from cmb import MLab

print("Creating instance of viz wrapper...")


def plot_ex(force_pyvista):
    mlab = MLab(force_pyvista=force_pyvista)

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

    print("Creating mesh...")
    mlab.triangular_mesh(x, y, z, triangles, scalars=t, colormap='bwr')

    print("Setting colorbar...")
    mlab.colorbar()

    print("Showing...")
    mlab.show()

    print("Done!")


for opt in (True, False):
    plot_ex(opt)

