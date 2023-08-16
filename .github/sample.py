import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from disba import PhaseDispersion, depthplot

from evodcinv import EarthModel, Layer, Curve


# Generate synthetic data
# Example taken from <https://www.geopsy.org/wiki/index.php/Dispersion_curve_inversion>
velocity_model = np.array([
    [7.5, 500.0, 200.0, 1700.0],
    [25.0, 1350.0, 210.0, 1900.0],
    [0.0, 2000.0, 1000.0, 2500.0],
]) * 1.0e-3

pd = PhaseDispersion(*velocity_model.T)
f = np.linspace(2.0, 20.0, 50)
t = 1.0 / f[::-1]
cp = pd(t, mode=0, wave="rayleigh")

# Initialize model
model = EarthModel()

# Build model search boundaries from top to bottom
# First argument is the bounds of layer's thickness [km]
# Second argument is the bounds of layer's S-wave velocity [km/s]
model.add(Layer([0.001, 0.1], [0.1, 3.0]))
model.add(Layer([0.001, 0.1], [0.1, 3.0]))

# Constant density (=2 g/cm3)
# First argument is P-wawe velocity [km/s]
density = lambda vp: 2.0

# Configure model
model.configure(
    optimizer="cpso",
    misfit="rmse",
    density=density,
    optimizer_args={
        "popsize": 10,
        "maxiter": 100,
        "workers": -1,
        "seed": 0,
    },
)

# Define dispersion curves to invert
curves = [Curve(cp.period, cp.velocity, 0, "rayleigh", "phase")]

# Run inversion
# See stochopy's documentation for optimizer options <https://keurfonluu.github.io/stochopy/>
res = model.invert(curves)
res = res.threshold(0.02)

# Plot results
fig, ax = plt.subplots(1, 3, figsize=(15, 6))

for a in ax:
    a.grid(True, linestyle=":")

zmax = 0.04
cmap = "viridis_r"

# Velocity model
res.plot_model(
    "vs",
    zmax=zmax,
    show="all",
    ax=ax[0],
    plot_args={"cmap": cmap},
)
depthplot(
    velocity_model[:, 0],
    velocity_model[:, 2],
    zmax=zmax,
    ax=ax[0],
    plot_args={
        "color": "black",
        "linewidth": 2,
        "label": "True",
    },
)
res.plot_model(
    "vs",
    zmax=zmax,
    show="best",
    ax=ax[0],
    plot_args={
        "color": "red",
        "linestyle": "--",
        "label": "Best",
    },
)
ax[0].legend(loc=1, frameon=False)

# Dispersion curve
res.plot_curve(
    t, 0, "rayleigh", "phase",
    show="all",
    ax=ax[1],
    plot_args={
        "type": "semilogx",
        "xaxis": "frequency",
        "cmap": cmap,
    },
)
ax[1].semilogx(
    1.0 / cp.period, cp.velocity,
    color="black",
    linewidth=2,
    label="True",
)
res.plot_curve(
    t, 0, "rayleigh", "phase",
    show="best",
    ax=ax[1],
    plot_args={
        "type": "semilogx",
        "xaxis": "frequency",
        "color": "red",
        "linestyle": "--",
        "label": "Best",
    },
)
ax[1].set_xlim(2.0, 20.0)
ax[1].xaxis.set_major_formatter(ScalarFormatter())
ax[1].xaxis.set_minor_formatter(ScalarFormatter())
ax[1].legend(loc=1, frameon=False)

# Misfit
res.plot_misfit(ax=ax[2])

# Colorbar
norm = Normalize(vmin=res.misfits.min(), vmax=res.misfits.max())
smap = ScalarMappable(norm=norm, cmap=cmap)
axins = inset_axes(
    ax[1],
    width="150%",
    height="6%",
    loc="lower center",
    borderpad=-6.0,
)

cb = plt.colorbar(smap, cax=axins, orientation="horizontal")
cb.set_label("Misfit value")

plt.show()
