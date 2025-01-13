import numpy as np

from typing import Tuple
from abc import ABC


class Geometry(ABC):
    def __init__(self) -> None:
        self.n_particles: int
        self.position: np.ndarray
        self.velocity: np.ndarray
        self.phase: np.ndarray


class Circle(Geometry):
    def __init__(
        self,
        phase: int,
        radius: float,
        n_particles: int,
        velocity: Tuple[float, float],
        coordinates: Tuple[float, float],
    ) -> None:
        self.n_particles = n_particles
        self.position = np.zeros(shape=(n_particles, 2), dtype=np.float32)
        self.velocity = np.zeros(shape=(n_particles, 2), dtype=np.float32)
        self.phase = np.full(shape=n_particles, fill_value=phase, dtype=np.int32)

        # Create the position of the circle.
        for i in range(n_particles):
            t = 2 * np.pi * np.random.rand()
            r = radius * np.sqrt(np.random.rand())
            x = (r * np.sin(t)) + coordinates[0]
            y = (r * np.cos(t)) + coordinates[1]
            self.velocity[i] = velocity
            self.position[i] = [x, y]


class Square(Geometry):
    def __init__(
        self,
        phase: int,
        size: float,
        n_particles: int,
        velocity: Tuple[float, float],
        coordinates: Tuple[float, float],
    ) -> None:
        self.n_particles = n_particles
        self.position = np.zeros(shape=(n_particles, 2), dtype=np.float32)
        self.velocity = np.zeros(shape=(n_particles, 2), dtype=np.float32)
        self.phase = np.full(shape=n_particles, fill_value=phase, dtype=np.int32)

        # Translate from center-coordinates to lower-left-coordinates.
        coordinates = (coordinates[0] - 0.5 * size, coordinates[1] - 0.5 * size)

        # Create the position of the square.
        for i in range(n_particles):
            x = np.random.rand() * size + coordinates[0]
            y = np.random.rand() * size + coordinates[1]
            self.velocity[i] = velocity
            self.position[i] = [x, y]
