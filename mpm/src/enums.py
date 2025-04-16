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


class LatenHeat:
    Water = 334.4
    Ice = 0.0
