from src.configurations.geometries import Geometry

import taichi as ti


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
        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.ambient_temperature = ambient_temperature

        self.initial_geometries = []
        self.subsequent_geometries = []
        for geometry in geometries:
            if geometry.frame_threshold == 0:
                self.initial_geometries.append(geometry)
            else:
                self.subsequent_geometries.append(geometry)

        # Sort this by frame_threshold, so only the first element has to be checked against.
        self.subsequent_geometries.sort(key=(lambda g: g.frame_threshold))
