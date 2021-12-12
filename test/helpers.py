import numpy as np

from evodcinv import Curve, EarthModel, Layer


period = np.linspace(0.05, 0.5, 20)
data = [
    0.19000963, 0.19268745, 0.19508398, 0.19702339, 0.19911409,
    0.20199262, 0.20634765, 0.21308453, 0.22405367, 0.24406380,
    0.28188331, 0.34339948, 0.39387778, 0.43183718, 0.46434448,
    0.49418631, 0.52596588, 0.56283204, 0.59969819, 0.63656434,
]
curves = [
    Curve(period, data, 0, "rayleigh", "phase", weight=1.0, uncertainties=None)
]


model = EarthModel()
model.add(Layer([0.001, 0.1], [0.1, 3.0]))
model.add(Layer([0.001, 0.1], [0.1, 3.0]))
model.configure(
    optimizer="cpso",
    misfit="rmse",
    density="nafe-drake",
    optimizer_args={
        "popsize": 5,
        "maxiter": 5,
        "seed": 0,
    },
)