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
_E = 1.4e5
_nu = 0.2


class Lambda:
    Water = 5e9  # TODO: this could be lower, and then saved into f32 field again?!
    Ice = _E * _nu / ((1 + _nu) * (1 - 2 * _nu))


class Mu:
    Water = 0
    Ice = _E / (2 * (1 + _nu))
