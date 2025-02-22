class Classification:
    Empty = 22
    Colliding = 33
    Interior = 44


class Phase:
    Ice = 55
    Water = 66


class Color:
    Ice = (0.8156862745098039, 0.8862745098039215, 1.0)
    Water = (0.27058823529411763, 0.3284671532846715, 1.0)
    Background = (0.09019607843137255, 0.0784313725490196, 0.0784313725490196)


class State:
    Inactive = 77
    Active = 88


# TODO: Capacity should not be an enum!?
class Capacity:
    Water = 4186  # j/dC
    Ice = 2093  # j/dC
    Zero = 0


class Conductivity:
    Water = 0.55
    Ice = 2.33
    Zero = 0
