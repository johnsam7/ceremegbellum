import numpy as np
from cmb import MLabEmulator

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

figure = mlab.triangular_mesh(x, y, z, triangles,
                              scalars=t, colormap='bwr')

# We can set camera position using angles
figure.camera.azimuth = -90
figure.camera.elevation = -90

# Alternativley, we can set camera position using points.
figure.camera.position = (3.0, 1.5, 2.5)

# Depending on your 3D framework, you might need to
# manually query an update.
figure.app.processEvents()  # makes camera move requests take effect
figure.render()  # forces a redraw

# Takes a screenshot of the current camera view
figure.screenshot('my_screenshot')


print("Done!")
