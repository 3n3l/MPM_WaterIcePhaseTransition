import numpy as np

from src.enums import State
from typing import Tuple
from abc import ABC


class Geometry(ABC):
    def __init__(self, n_particles, phase) -> None:
        self.n_particles = n_particles
        self.position = np.zeros(shape=(n_particles, 2), dtype=np.float32)
        self.velocity = np.zeros(shape=(n_particles, 2), dtype=np.float32)
        self.phase = np.full(shape=n_particles, fill_value=phase, dtype=int)
        self.frame_threshold = np.zeros(shape=n_particles, dtype=int)
        self.state = np.zeros(shape=n_particles, dtype=int)


class Circle(Geometry):
    def __init__(
        self,
        phase: int,
        radius: float,
        n_particles: int,
        velocity: Tuple[float, float],
        coordinates: Tuple[float, float],
        frame_threshold: int = 0,
    ) -> None:
        super().__init__(n_particles, phase)
        for p in range(n_particles):
            t = 2 * np.pi * np.random.rand()
            r = radius * np.sqrt(np.random.rand())
            x = (r * np.sin(t)) + coordinates[0]
            y = (r * np.cos(t)) + coordinates[1]
            self.velocity[p] = velocity
            self.position[p] = [x, y]
            self.state[p] = State.Active if frame_threshold == 0 else State.Inactive
            self.frame_threshold[p] = frame_threshold


class Square(Geometry):
    def __init__(
        self,
        phase: int,
        size: float,
        n_particles: int,
        velocity: Tuple[float, float],
        coordinates: Tuple[float, float],
        frame_threshold: int = 0,
    ) -> None:
        super().__init__(n_particles, phase)
        for p in range(n_particles):
            x = np.random.rand() * size + coordinates[0]
            y = np.random.rand() * size + coordinates[1]
            self.velocity[p] = velocity
            self.position[p] = [x, y]
            self.state[p] = State.Active if frame_threshold == 0 else State.Inactive
            self.frame_threshold[p] = frame_threshold
