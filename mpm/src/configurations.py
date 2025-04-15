from src.geometries import Geometry
from functools import reduce

import taichi as ti
import numpy as np


@ti.data_oriented
class Configuration:
    """This class represents a starting configuration for the MLS-MPM algorithm."""

    def __init__(
        self,
        name: str,
        geometries: list[Geometry],
        nu=0.2,  # Poisson's ratio (0.2)
        E=1.4e5,  # Young's modulus (1.4e5)
        zeta=10,  # Hardening coefficient (10)
        stickiness=1.0,  # Higher value means a stickier border
        friction=1.0,  # Higher value means the border has more friction
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=0.0,  # temperature of empty (air) cells
    ):
        self.E = E
        self.nu = nu
        self.name = name
        self.zeta = zeta
        self.theta_c = theta_c
        self.theta_s = theta_s
        self.friction = friction
        self.stickiness = stickiness
        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.ambient_temperature = ambient_temperature

        # TODO: build list of configurations that are added in the beginning (frame_threshold == 0)
        # TODO: build list of configurations that are added at a specific frame (frame_threshold > 0)
        self.initial_geometries = geometries
        # self.initial_geometries = []
        # self.subsequent_geometries = []
        # for geometry in geometries:
        #     if geometry.frame_threshold == 0:
        #         self.initial_geometries.append(geometry)
        #     else:
        #         self.subsequent_geometries.append(geometry)
        #
        # # Sort this by frame_threshold, so only the first element has to be checked against.
        # self.subsequent_geometries.sort(key=(lambda g: g.frame_threshold))

    # TODO: this can all go?
    #     # Properties.
    #     self.n_particles = reduce(lambda sum, g: sum + g.n_particles, geometries, 0)
    #
    #     # Arrays holding properties for all geometries.
    #     self._arr_activation_threshold = np.concatenate([g.frame_threshold for g in geometries], dtype=int)
    #     self._arr_temperature = np.concatenate([g.temperature for g in geometries], dtype=np.float32).flatten()
    #     self._arr_position = np.concatenate([g.position for g in geometries], dtype=np.float32)
    #     self._arr_velocity = np.concatenate([g.velocity for g in geometries], dtype=np.float32)
    #     self._arr_phase = np.concatenate([g.phase for g in geometries], dtype=np.float32).flatten()
    #     self._arr_state = np.concatenate([g.state for g in geometries], dtype=int)
    #
    # def build(self):
    #     """This builds the Taichi fields, must be called before use and after ti.init(...)."""
    #     # Declare fields.
    #     self.activation_threshold_p = ti.field(dtype=int, shape=self.n_particles)
    #     self.temperature_p = ti.field(dtype=ti.f32, shape=self.n_particles)
    #     self.position_p = ti.Vector.field(2, dtype=ti.f32, shape=self.n_particles)
    #     self.velocity_p = ti.Vector.field(2, dtype=ti.f32, shape=self.n_particles)
    #     self.phase_p = ti.field(dtype=ti.f32, shape=self.n_particles)
    #     self.state_p = ti.field(dtype=int, shape=self.n_particles)
    #
    #     # Initialize fields.
    #     self.activation_threshold_p.from_numpy(self._arr_activation_threshold)
    #     self.temperature_p.from_numpy(self._arr_temperature)
    #     self.position_p.from_numpy(self._arr_position)
    #     self.velocity_p.from_numpy(self._arr_velocity)
    #     self.phase_p.from_numpy(self._arr_phase)
    #     self.state_p.from_numpy(self._arr_state)
