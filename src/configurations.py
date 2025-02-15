from functools import reduce
from geometries import Geometry

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

        # Properties.
        self.n_particles = reduce(lambda sum, g: sum + g.n_particles, geometries, 0)

        # Declare fields.
        self.p_position = ti.Vector.field(2, dtype=ti.f32, shape=self.n_particles)
        self.p_velocity = ti.Vector.field(2, dtype=ti.f32, shape=self.n_particles)
        # TODO: rename p_activity_bound to something more meaningful
        self.p_activity_bound = ti.field(dtype=int, shape=self.n_particles)
        self.p_phase = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.p_state = ti.field(dtype=int, shape=self.n_particles)

        # Initialize fields.
        self.p_position.from_numpy(np.concatenate([g.position for g in geometries], dtype=np.float32))
        self.p_velocity.from_numpy(np.concatenate([g.velocity for g in geometries], dtype=np.float32))
        self.p_activity_bound.from_numpy(np.concatenate([g.frame_threshold for g in geometries], dtype=int))
        self.p_phase.from_numpy(np.concatenate([g.phase for g in geometries], dtype=np.float32).flatten())
        self.p_state.from_numpy(np.concatenate([g.state for g in geometries], dtype=int))
