import numpy as np
import taichi as ti

from src.enums import Capacity, Color, Conductivity, Phase
from src.mpm_solver import LATENT_HEAT
from typing import Tuple
from abc import ABC, abstractmethod

LATENT_HEAT = 334.4  # TODO: set this to correct value, maybe hold in enums or constants file?


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
        self.color = Color.Water if phase == Phase.Water else Color.Ice
        self.heat = LATENT_HEAT if phase == Phase.Water else 0.0

        self.frame_threshold = frame_threshold
        self.velocity = list(velocity)  # TODO: should be Tuple, but Taichi expects list?
        self.temperature = temperature
        self.phase = phase

        self.center: list # x, y coordinates of the center, used as initial sample

        # self.frame_threshold = np.zeros(shape=n_particles, dtype=int)
        # self.state = np.zeros(shape=n_particles, dtype=int)

        # self.n_particles = n_particles
        # self.position = np.zeros(shape=(n_particles, 2), dtype=np.float32)

    @abstractmethod
    # TODO: this might better be a ti.field for performance?
    def in_bounds(self, x: float, y: float) -> bool:
        pass


class Circle(Geometry):
    def __init__(
        self,
        phase: int,
        radius: float,
        # n_particles: int,
        velocity: Tuple[float, float],
        center: Tuple[float, float],
        frame_threshold: int = 0,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(velocity, phase, temperature, frame_threshold)

        self.center = list(center)
        self.x = center[0]
        self.y = center[1]
        self.radius = radius
        self.squared_radius = radius * radius

        # TODO: remove this
        # for p in range(n_particles):
        #     t = 2 * np.pi * np.random.rand()
        #     r = radius * np.sqrt(np.random.rand())
        #     x = (r * np.sin(t)) + coordinates[0]
        #     y = (r * np.cos(t)) + coordinates[1]
        #     self.velocity[p] = velocity
        #     self.position[p] = [x, y]
        #     self.state[p] = State.Active if frame_threshold == 0 else State.Inactive
        #     self.frame_threshold[p] = frame_threshold

    @ti.func
    def in_bounds(self, x: float, y: float) -> bool:
        return (self.x - x) ** 2 + (self.y - y) ** 2 <= self.squared_radius


class Rectangle(Geometry):
    def __init__(
        self,
        phase: int,
        width: float,
        height: float,
        # n_particles: int,
        velocity: Tuple[float, float],
        # TODO: instead of width, height and coordinates it should be lower left corner
        #       and upper right corner?
        lower_left: Tuple[float, float],
        frame_threshold: int = 0,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(velocity, phase, temperature, frame_threshold)

        # The bounding box.
        self.l_bound = lower_left[0]
        self.b_bound = lower_left[1]
        self.r_bound = self.l_bound + width
        self.t_bound = self.b_bound + height

        self.center = [self.l_bound + 0.5 * self.r_bound, self.b_bound + 0.5 * self.t_bound]

        # Only used for testing
        self.width = width
        self.height = height
        self.x = lower_left[0]
        self.y = lower_left[1]

        # TODO: this can go
        # for p in range(n_particles):
        #     x = np.random.rand() * width + coordinates[0]
        #     y = np.random.rand() * height + coordinates[1]
        #     self.velocity[p] = velocity
        #     self.position[p] = [x, y]
        #     self.state[p] = State.Active if frame_threshold == 0 else State.Inactive
        #     self.frame_threshold[p] = frame_threshold

    @ti.func
    def in_bounds(self, x: float, y: float) -> bool:
        return self.l_bound <= x <= self.r_bound and self.b_bound <= y <= self.t_bound
