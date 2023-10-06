import numpy as np
from cmb import MLabEmulator


def get_figures():
    mlab = MLabEmulator()

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

    figures.append(mlab.triangular_mesh(x, y, z, triangles,
                                        scalars=t, colormap='bwr'))

    t2 = [3 for _ in t]
    mlab.figure(bgcolor=(1., 1., 1.),
                fgcolor=(0., 0., 0.),
                size=(1200, 1200))

    figures.append(mlab.triangular_mesh(x, y, z, triangles,
                                        scalars=t2, colormap='bwr'))

    mlab.colorbar()
    return figures


def take_screenshot(figure, file_name):
    # We can set camera position using angles
    figure.camera.azimuth = -90
    figure.camera.elevation = -90

    # Alternativley, we can set camera position using points.
    figure.camera.position = (2.0, 1.0, 2.0)

    # Depending on your 3D framework, you might need to
    # manually query an update.
    figure.update()

    # Takes a screenshot of the current camera view
    figure.screenshot(file_name)


figures = get_figures()

count = 0

for figure in figures:
    count += 1
    take_screenshot(figure, f'screenshot_{count}')

print("Done!")
