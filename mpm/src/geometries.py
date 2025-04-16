import taichi as ti

from src.enums import Capacity, Color, Conductivity, Phase, LatenHeat
from abc import ABC, abstractmethod
from typing import Tuple


class Geometry(ABC):
    def __init__(
        self,
        velocity: Tuple[float, float],
        phase: int,
        temperature: float,
        frame_threshold: int,
    ) -> None:
        self.conductivity = Conductivity.Water if phase == Phase.Water else Conductivity.Ice
        self.capacity = Capacity.Water if phase == Phase.Water else Capacity.Ice
        self.heat = LatenHeat.Water if phase == Phase.Water else LatenHeat.Ice
        self.color = Color.Water if phase == Phase.Water else Color.Ice

        self.frame_threshold = frame_threshold
        self.velocity = list(velocity)  # TODO: should be Tuple, but Taichi expects list?
        self.temperature = temperature
        self.phase = phase

        self.center: list  # x, y coordinates of the center, used as initial sample

    @abstractmethod
    # TODO: this might better be a ti.field for performance?
    def in_bounds(self, x: float, y: float) -> bool:
        pass


class Circle(Geometry):
    def __init__(
        self,
        center: Tuple[float, float],
        velocity: Tuple[float, float],
        radius: float,
        temperature: float = 0.0,
        frame_threshold: int = 0,
        phase: int = Phase.Water,
    ) -> None:
        super().__init__(velocity, phase, temperature, frame_threshold)

        self.center = list(center)
        self.x = center[0]
        self.y = center[1]
        self.height = radius # for convenience while sampling
        self.radius = radius
        self.squared_radius = radius * radius

    @ti.func
    def in_bounds(self, x: float, y: float) -> bool:
        return (self.x - x) ** 2 + (self.y - y) ** 2 <= self.squared_radius


class Rectangle(Geometry):
    def __init__(
        self,
        lower_left: Tuple[float, float],
        velocity: Tuple[float, float],
        size: Tuple[float, float],
        temperature: float = 0.0,
        frame_threshold: int = 0,
        phase: int = Phase.Water,
    ) -> None:
        super().__init__(velocity, phase, temperature, frame_threshold)

        self.width, self.height = size
        self.x, self.y = lower_left

        # The bounding box.
        self.l_bound = self.x
        self.b_bound = self.y
        self.r_bound = self.x + self.width
        self.t_bound = self.y + self.height
        # print()
        # print(self.l_bound)
        # print(self.b_bound)
        # print(self.r_bound)
        # print(self.t_bound)
        self.center = [self.x + 0.5 * self.width, self.y + 0.5 * self.height]
        # print(self.center)

    @ti.func
    def in_bounds(self, x: float, y: float) -> bool:
        return self.l_bound <= x <= self.r_bound and self.b_bound <= y <= self.t_bound
