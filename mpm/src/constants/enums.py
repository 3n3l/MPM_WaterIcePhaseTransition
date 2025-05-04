class Classification:
    Empty = 22
    Colliding = 33
    Interior = 44


class Phase:
    Ice = 55
    Water = 66


class Color:
    Ice = (0.81, 0.88, 1.0)
    Water = (0.27, 0.35, 1.0)
    Background = (0.09, 0.07, 0.07)
    # TODO: GUI and GGUI expect the colors in different forms
    #       unify this somehow
    _Ice = 0xEDF5FF
    _Water = 0x4589FF
    _Background = 0x171414


class Capacity:
    Water = 4.186  # j/dC
    Ice = 2.093  # j/dC
    # Water = 4186  # j/dC
    # Ice = 2093  # j/dC
    Zero = 0.0


class Conductivity:
    Water = 0.55
    Ice = 2.33
    Zero = 0


class Density:
    Water = 997.0
    Ice = 400.0


class LatentHeat:
    Water = 334.4
    Ice = 0.0


class State:
    Active = 0
    Hidden = 1


# TODO: find good values for ice???
# TODO: refactor this, maybe use variable values again???
# FIXME: E = 1.4e3 -> collapse, E = 1.4e4 -> explosion
ice_E = 1.4e4
ice_nu = 0.2
# water_E = 5e5
# water_nu = 0.45

# import numpy as np

# Compute max dt from parameters:
# TODO: move this somewhere else
# dx = 128.0
# ice_rho = Density.Ice / 1000
# water_rho = Density.Water / 1000
# max_dt_ice = dx / np.sqrt((ice_E * (1 - ice_nu)) / (ice_rho * (1 + ice_nu) * (1 - 2 * ice_nu)))
# max_dt_water = dx / np.sqrt((water_E * (1 - water_nu)) / (water_rho * (1 + water_nu) * (1 - 2 * water_nu)))
# print(f"max dt ice, water = {max_dt_ice}, {max_dt_water}")
# print(f"water mu = {water_E / (2 * (1 + water_nu))}")


class Lambda:
    Water = 5e9  # TODO: this could be lower, and then saved into f32 field again?!
    # Water = water_E * water_nu / ((1 + water_nu) * (1 - 2 * water_nu))
    Ice = ice_E * ice_nu / ((1 + ice_nu) * (1 - 2 * ice_nu))


class Mu:
    Water = 0
    # Water = water_E / (2 * (1 + water_nu))
    Ice = ice_E / (2 * (1 + ice_nu))
