import numpy as np

from src.enums import State
from typing import Tuple
from abc import ABC


class Geometry(ABC):
    def __init__(self, n_particles, phase, temperature) -> None:
        self.temperature = np.full(shape=n_particles, fill_value=temperature, dtype=np.float32)
        self.phase = np.full(shape=n_particles, fill_value=phase, dtype=int)
        self.position = np.zeros(shape=(n_particles, 2), dtype=np.float32)
        self.velocity = np.zeros(shape=(n_particles, 2), dtype=np.float32)
        self.frame_threshold = np.zeros(shape=n_particles, dtype=int)
        self.state = np.zeros(shape=n_particles, dtype=int)
        self.n_particles = n_particles


class Circle(Geometry):
    def __init__(
        self,
        phase: int,
        radius: float,
        n_particles: int,
        velocity: Tuple[float, float],
        coordinates: Tuple[float, float],
        frame_threshold: int = 0,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(n_particles, phase, temperature)
        for p in range(n_particles):
            t = 2 * np.pi * np.random.rand()
            r = radius * np.sqrt(np.random.rand())
            x = (r * np.sin(t)) + coordinates[0]
            y = (r * np.cos(t)) + coordinates[1]
            self.velocity[p] = velocity
            self.position[p] = [x, y]
            self.state[p] = State.Active if frame_threshold == 0 else State.Inactive
            self.frame_threshold[p] = frame_threshold


class Rectangle(Geometry):
    def __init__(
        self,
        phase: int,
        width: float,
        height: float,
        n_particles: int,
        velocity: Tuple[float, float],
        coordinates: Tuple[float, float],
        frame_threshold: int = 0,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(n_particles, phase, temperature)
        for p in range(n_particles):
            x = np.random.rand() * width + coordinates[0]
            y = np.random.rand() * height + coordinates[1]
            self.velocity[p] = velocity
            self.position[p] = [x, y]
            self.state[p] = State.Active if frame_threshold == 0 else State.Inactive
            self.frame_threshold[p] = frame_threshold
