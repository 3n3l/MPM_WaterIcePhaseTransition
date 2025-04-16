import taichi as ti

from src.enums import Capacity, Color, Conductivity, Phase, LatenHeat
from abc import ABC, abstractmethod
from typing import Tuple


class Geometry(ABC):
    def __init__(
        self,
        velocity: Tuple[float, float],
        frame_threshold: int,
        temperature: float,
        phase: int,
    ) -> None:
        self.conductivity = Conductivity.Water if phase == Phase.Water else Conductivity.Ice
        self.capacity = Capacity.Water if phase == Phase.Water else Capacity.Ice
        self.heat = LatenHeat.Water if phase == Phase.Water else LatenHeat.Ice
        self.color = Color.Water if phase == Phase.Water else Color.Ice
        self.frame_threshold = frame_threshold
        self.temperature = temperature
        self.velocity = list(velocity)
        self.phase = phase

    @abstractmethod
    def in_bounds(self, x: float, y: float) -> bool:
        pass

    @abstractmethod
    def random_seed(self) -> ti.Vector:
        pass


class Circle(Geometry):
    def __init__(
        self,
        velocity: Tuple[float, float],
        center: Tuple[float, float],
        radius: float,
        temperature: float = 0.0,
        frame_threshold: int = 0,
        phase: int = Phase.Water,
    ) -> None:
        super().__init__(velocity, frame_threshold, temperature, phase)
        self.x, self.y = list(center)
        self.squared_radius = radius * radius
        self.radius = radius

    @ti.func
    def in_bounds(self, x: float, y: float) -> bool:
        return (self.x - x) ** 2 + (self.y - y) ** 2 <= self.squared_radius

    @ti.func
    def random_seed(self) -> ti.Vector:
        r = self.radius * ti.math.sqrt(ti.random())
        t = 2 * ti.math.pi * ti.random()
        x = (r * ti.sin(t)) + self.x
        y = (r * ti.cos(t)) + self.y
        return ti.Vector([x, y])


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
        super().__init__(velocity, frame_threshold, temperature, phase)
        self.width, self.height = size
        self.x, self.y = lower_left
        self.r_bound = self.x + self.width
        self.t_bound = self.y + self.height

    @ti.func
    def in_bounds(self, x: float, y: float) -> bool:
        return self.x <= x <= self.r_bound and self.y <= y <= self.t_bound

    @ti.func
    def random_seed(self) -> ti.Vector:
        x = self.x + ti.random() * self.width
        y = self.y + ti.random() * self.height
        return ti.Vector([x, y])
