"""
Testing a thermodynamic model, roughly based on
'Augmented MPM for phase-change and varied materials'
"""

from taichi.linalg import MatrixFreeCG, LinearOperator
from datetime import datetime
import numpy as np
import taichi as ti
import os

# ti.init(arch=ti.cpu, debug=True)
ti.init(arch=ti.vulkan)

WATER_CONDUCTIVITY = 0.55  # Water: 0.55, Ice: 2.33
ICE_CONDUCTIVITY = 2.33
WATER_HEAT_CAPACITY = 4.186  # j/dC
ICE_HEAT_CAPACITY = 2.093  # j/dC
LATENT_HEAT = 0.334  # J/kg
GRAVITY = -9.81

R = 0.2
QUALITY = 1
N_PARTICLES = 1_000 * (QUALITY**2)
T = np.linspace(0, 2 * np.pi, N_PARTICLES + 2, dtype=np.float32)[1:-1]
THETAS = ti.field(ti.f32, shape=(N_PARTICLES))
THETAS.from_numpy(T)


class Classification:
    Empty = 0
    Colliding = 1
    Interior = 2


class Phase:
    Ice = 0
    Water = 1


class Color:
    Ice = [0.8, 0.8, 1]
    Water = [0.4, 0.4, 1]


@ti.data_oriented
class ThermodynamicModel:
    def __init__(
        self,
        quality: int,
        n_particles: int,
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        zeta=10,  # Hardening coefficient (10)
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
    ):
        # Parameters to control the simulation
        self.window = ti.ui.Window(name="Thermodynamic Model", res=(720, 720), fps_limit=60)
        self.gui = self.window.get_gui()
        self.canvas = self.window.get_canvas()
        self.is_paused = True
        self.should_write_to_disk = False
        self.is_showing_settings = not self.is_paused
        # Create a parent directory, in there folders will be created containing
        # newly created frames, videos and GIFs.
        self.parent_dir = ".output"
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)

        # MPM Parameters that are configuration independent
        self.quality = quality
        self.n_particles = n_particles
        self.n_grid = 128 * quality
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = 1e-4 / self.quality
        self.rho_0 = 4e2
        self.particle_vol = (self.dx * 0.1) ** 2

        # Number of dimensions
        self.n_dimensions = 2

        # Parameters to control melting/freezing
        # TODO: these are variables and need to be put into fields
        # TODO: these depend not only on phase, but also on temperature,
        #       so ideally they are functions of these two variables
        # self.heat_conductivity = 0.55 # Water: 0.55, Ice: 2.33
        # self.heat_capacity = 4.186 # Water: 4.186, Ice: 2.093 (j/dC)
        # self.latent_heat = 0.334 # in J/kg

        # Properties on MAC-faces.
        self.face_conductivity_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_conductivity_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.face_velocity_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_velocity_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.face_mass_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_mass_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))

        # Properties on MAC-cells.
        self.cell_classification = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_temperature = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_divergence = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_capacity = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_mass = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))

        # TODO: are these all needed?
        self.cell_inv_lambda = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_JE = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid + 1))
        self.cell_JP = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid + 1))

        # These fields will be used to solve for pressure, where Ap = b
        # TODO: Better names for Ap and b?
        self.face_volume_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_volume_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        # self.cell_pressure = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid + 1))
        self.cell_pressure = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        # self.cell_corrected_pressure = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid + 1))
        # self.cell_Ap = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid + 1))
        self.b = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.Ax = LinearOperator(self.compute_Ap)

        # Properties on particles.
        self.particle_conductivity = ti.field(dtype=ti.float32, shape=self.n_particles)
        self.particle_temperature = ti.field(dtype=ti.float32, shape=self.n_particles)
        self.particle_inv_lambda = ti.field(dtype=ti.float32, shape=self.n_particles)
        self.particle_position = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.particle_velocity = ti.Vector.field(2, dtype=float, shape=self.n_particles)
        self.particle_capacity = ti.field(dtype=ti.float32, shape=self.n_particles)
        self.particle_color = ti.Vector.field(3, dtype=float, shape=self.n_particles)
        self.particle_phase = ti.field(dtype=ti.float32, shape=self.n_particles)
        self.particle_mass = ti.field(dtype=ti.float32, shape=self.n_particles)
        self.particle_FE = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)
        self.particle_C = ti.Matrix.field(2, 2, dtype=float, shape=self.n_particles)

        self.particle_JE = ti.field(dtype=float, shape=self.n_particles)
        self.particle_JP = ti.field(dtype=float, shape=self.n_particles)

        # Variables controlled from the GUI, stored in fields to be accessed from compiled kernels.
        self.stickiness = ti.field(dtype=float, shape=())
        self.friction = ti.field(dtype=float, shape=())
        self.lambda_0 = ti.field(dtype=float, shape=())
        self.theta_c = ti.field(dtype=float, shape=())
        self.theta_s = ti.field(dtype=float, shape=())
        self.mu_0 = ti.field(dtype=float, shape=())
        self.zeta = ti.field(dtype=int, shape=())
        self.nu = ti.field(dtype=float, shape=())
        self.E = ti.field(dtype=float, shape=())

        # Initialize fields
        self.lambda_0[None] = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu_0[None] = E / (2 * (1 + nu))
        self.theta_c[None] = theta_c
        self.theta_s[None] = theta_s
        self.zeta[None] = zeta
        self.nu[None] = nu
        self.E[None] = E

    @ti.kernel
    def reset_grids(self):
        self.cell_classification.fill(0)
        self.face_velocity_x.fill(0)
        self.face_velocity_y.fill(0)
        self.cell_pressure.fill(0)
        self.face_mass_x.fill(0)
        self.face_mass_y.fill(0)
        self.cell_mass.fill(0)

        self.cell_JE.fill(1)
        self.cell_JP.fill(1)
        self.particle_JE.fill(1)
        self.particle_JP.fill(1)

        # TODO: implemented volume computation
        self.face_volume_x.fill(self.rho_0)
        self.face_volume_y.fill(self.rho_0)

    @ti.kernel
    def particle_to_grid(self):
        for p in self.particle_position:
            # Deformation gradient update.
            self.particle_FE[p] = (ti.Matrix.identity(float, 2) + self.dt * self.particle_C[p]) @ self.particle_FE[p]

            # Clamp singular values to simulate plasticity and elasticity.
            U, sigma, V = ti.svd(self.particle_FE[p])
            JE, JP = 1.0, 1.0
            for d in ti.static(range(self.n_dimensions)):
                # Clamp singular values to [1 - theta_c, 1 + theta_s]
                singular_value = float(sigma[d, d])
                clamped = max(singular_value, 1 - self.theta_c[None])
                clamped = min(clamped, 1 + self.theta_s[None])
                sigma[d, d] = clamped
                JP *= singular_value / clamped
                JE *= clamped

            la = self.lambda_0[None]
            mu = self.mu_0[None]
            if self.particle_phase[p] == Phase.Water:
                # TODO: Apply correction for dilational/deviatoric stresses?
                # Reset elastic deformation gradient to avoid numerical instability.
                self.particle_FE[p] = ti.Matrix.identity(float, self.n_dimensions) * JE ** (1 / self.n_dimensions)
                # Set the viscosity to zero.
                mu = 0
            elif self.particle_phase[p] == Phase.Ice:
                # Reconstruct elastic deformation gradient after plasticity
                self.particle_FE[p] = U @ sigma @ V.transpose()
                # Apply ice hardening by adjusting Lame parameters
                hardening = ti.max(0.1, ti.min(20, ti.exp(self.zeta[None] * (1.0 - JP))))
                la *= hardening
                mu *= hardening

            # Compute Piola-Kirchhoff stress P(F), (JST16, Eqn. 52)
            stress = 2 * mu * (self.particle_FE[p] - U @ V.transpose()) @ self.particle_FE[p].transpose()
            stress += ti.Matrix.identity(float, 2) * la * JE * (JE - 1)

            # Compute D^(-1), which equals constant scaling for quadratic/cubic kernels.
            D_inv = 4 * self.inv_dx * self.inv_dx  # Quadratic interpolation
            # D_inv = 3 * self.inv_dx * self.inv_dx  # Cubic interpolation

            # TODO: What happens here exactly? Something with Cauchy-stress?
            stress *= -self.dt * self.particle_vol * D_inv

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            affine = stress + self.particle_mass[p] * self.particle_C[p]
            x_affine = affine @ ti.Vector([1, 0])
            y_affine = affine @ ti.Vector([0, 1])

            # We use an additional offset of 0.5 for element-wise flooring.
            x_stagger = ti.Vector([self.dx / 2, 0])
            y_stagger = ti.Vector([0, self.dx / 2])
            c_base = (self.particle_position[p] * self.inv_dx - 0.5).cast(int)
            x_base = (self.particle_position[p] * self.inv_dx - (x_stagger + 0.5)).cast(int)
            y_base = (self.particle_position[p] * self.inv_dx - (y_stagger + 0.5)).cast(int)
            c_fx = self.particle_position[p] * self.inv_dx - c_base.cast(float)
            x_fx = self.particle_position[p] * self.inv_dx - x_base.cast(float)
            y_fx = self.particle_position[p] * self.inv_dx - y_base.cast(float)
            # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1,fx-2)
            c_w = [0.5 * (1.5 - c_fx) ** 2, 0.75 - (c_fx - 1) ** 2, 0.5 * (c_fx - 0.5) ** 2]
            x_w = [0.5 * (1.5 - x_fx) ** 2, 0.75 - (x_fx - 1) ** 2, 0.5 * (x_fx - 0.5) ** 2]
            y_w = [0.5 * (1.5 - y_fx) ** 2, 0.75 - (y_fx - 1) ** 2, 0.5 * (y_fx - 0.5) ** 2]
            # Cubic kernels (JST16 Eqn. 122 with x=fx, abs(fx-1), abs(fx-2))
            # Taken from https://github.com/taichi-dev/advanced_examples/blob/main/mpm/mpm99_cubic.py
            # TODO: calculate own weights for x=fx, fx-1, fx-2
            # c_w = [
            #     0.5 * c_fx**3 - c_fx**2 + 2.0 / 3.0,
            #     0.5 * (-(c_fx - 1.0)) ** 3 - (-(c_fx - 1.0)) ** 2 + 2.0 / 3.0,
            #     1.0 / 6.0 * (2.0 + (c_fx - 2.0)) ** 3,
            # ]
            # x_w = [
            #     0.5 * x_fx**3 - x_fx**2 + 2.0 / 3.0,
            #     0.5 * (-(x_fx - 1.0)) ** 3 - (-(x_fx - 1.0)) ** 2 + 2.0 / 3.0,
            #     1.0 / 6.0 * (2.0 + (x_fx - 2.0)) ** 3,
            # ]
            # y_w = [
            #     0.5 * y_fx**3 - y_fx**2 + 2.0 / 3.0,
            #     0.5 * (-(y_fx - 1.0)) ** 3 - (-(y_fx - 1.0)) ** 2 + 2.0 / 3.0,
            #     1.0 / 6.0 * (2.0 + (y_fx - 2.0)) ** 3,
            # ]

            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                c_weight = c_w[i][0] * c_w[j][1]
                x_weight = x_w[i][0] * x_w[j][1]
                y_weight = y_w[i][0] * y_w[j][1]
                # c_dpos = (offset.cast(float) - c_fx) * self.dx
                x_dpos = (offset.cast(float) - x_fx) * self.dx
                y_dpos = (offset.cast(float) - y_fx) * self.dx

                # Rasterize mass to grid faces.
                self.face_mass_x[x_base + offset] += x_weight * self.particle_mass[p]
                self.face_mass_y[y_base + offset] += y_weight * self.particle_mass[p]

                # Rasterize velocity to grid faces.
                x_velocity = self.particle_mass[p] * self.particle_velocity[p][0] + x_affine @ x_dpos
                y_velocity = self.particle_mass[p] * self.particle_velocity[p][1] + y_affine @ y_dpos
                self.face_velocity_x[x_base + offset] += x_weight * x_velocity
                self.face_velocity_y[y_base + offset] += y_weight * y_velocity

                # Rasterize conductivity to grid faces.
                conductivity = self.particle_mass[p] * self.particle_conductivity[p]
                self.face_conductivity_x[x_base + offset] += x_weight * conductivity
                self.face_conductivity_y[y_base + offset] += y_weight * conductivity

                # Rasterize to cell centers.
                self.cell_mass[c_base + offset] += c_weight * self.particle_mass[p]
                self.cell_capacity[c_base + offset] += c_weight * self.particle_capacity[p]
                self.cell_temperature[c_base + offset] += c_weight * self.particle_temperature[p]

                # self.cell_inv_lambda[c_base + offset] += c_weight * self.particle_inv_lambda[p]
                self.cell_inv_lambda[c_base + offset] += c_weight * self.lambda_0[None]

                # NOTE: the old JE, JP values are used here to compute the cell values.
                # print("WEIGHT", c_weight)
                # print("WEIGHT", x_weight)
                # print("WEIGHT", y_weight)
                # print("CELLJE", self.cell_JE[i, j])
                # print("PARTJE", self.particle_JE[p])
                # print("RESULT", c_weight * self.particle_JE[p])

                self.cell_JE[c_base + offset] += c_weight * self.particle_JE[p]
                self.cell_JP[c_base + offset] += c_weight * self.particle_JP[p]
                # self.cell_JE[c_base + offset] += c_weight * JE
                # self.cell_JP[c_base + offset] += c_weight * JP

            self.particle_JE[p] = JE
            self.particle_JP[p] = JP
        # for i, j in self.cell_pressure:
        #     self.cell_pressure[i, j] = (-1 / self.cell_JP[i, j]) * self.lambda_0[None] * (self.cell_JE[i, j] - 1)

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.face_mass_x:
            if self.face_mass_x[i, j] > 0:  # No need for epsilon here
                self.face_velocity_x[i, j] *= 1 / self.face_mass_x[i, j]
                collision_left = i < 3 and self.face_velocity_x[i, j] < 0
                collision_right = i > (self.n_grid - 3) and self.face_velocity_x[i, j] > 0
                if collision_left or collision_right:
                    self.face_velocity_x[i, j] = 0
        for i, j in self.face_mass_y:
            if self.face_mass_y[i, j] > 0:  # No need for epsilon here
                self.face_velocity_y[i, j] *= 1 / self.face_mass_y[i, j]
                self.face_velocity_y[i, j] += self.dt * GRAVITY
                collision_top = j > (self.n_grid - 3) and self.face_velocity_y[i, j] > 0
                collision_bottom = j < 3 and self.face_velocity_y[i, j] < 0
                if collision_top or collision_bottom:
                    self.face_velocity_y[i, j] = 0
        for i, j in self.cell_mass:
            if self.cell_mass[i, j] > 0:  # No need for epsilon here
                self.cell_temperature[i, j] *= 1 / self.cell_mass[i, j]
                self.cell_inv_lambda[i, j] *= self.cell_mass[i, j]
                self.cell_capacity[i, j] *= 1 / self.cell_mass[i, j]

    @ti.kernel
    def classify_cells(self):
        # We can extract the offset coordinates from the faces by adding one to the respective axis,
        # e.g. we get the two x-faces with [i, j] and [i + 1, j], where each cell looks like:
        # -  ^  -
        # >  *  >
        # -  ^  -
        for i, j in self.cell_classification:
            # TODO: A cell is marked as colliding if all of its surrounding faces are colliding.
            # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.
            # TODO: Colliding cells are either assigned the temperature of the object it collides with or a user-defined
            # spatially-varying value depending on the setup. If the free surface is being enforced as a Dirichlet
            # temperature condition, the ambient air temperature is recorded for empty cells. No other cells
            # require temperatures to be recorded at this stage.
            is_colliding = False

            # A cell is interior if the cell and all of its surrounding faces have mass.
            is_interior = self.face_mass_x[i, j] > 0
            is_interior &= self.face_mass_y[i, j] > 0
            is_interior &= self.face_mass_x[i + 1, j] > 0
            is_interior &= self.face_mass_y[i, j + 1] > 0

            if is_colliding:
                self.cell_classification[i, j] = Classification.Colliding
            elif is_interior:
                self.cell_classification[i, j] = Classification.Interior
            else:
                self.cell_classification[i, j] = Classification.Empty

    @ti.func
    def sample(self, x, i, j):
        indices = ti.Vector([int(i), int(j)])
        indices = ti.max(0, ti.min(self.n_grid - 1, indices))
        return x[indices]

    @ti.kernel
    def compute_divergence(self):
        for i, j in self.cell_divergence:
            self.cell_divergence[i, j] = 0
            self.cell_divergence[i, j] += self.face_velocity_x[i + 1, j]
            self.cell_divergence[i, j] += self.face_velocity_y[i, j + 1]
            self.cell_divergence[i, j] -= self.face_velocity_x[i, j]
            self.cell_divergence[i, j] -= self.face_velocity_y[i, j]

    @ti.kernel
    def compute_Ap(self, p: ti.template(), Ap: ti.template()):  # pyright: ignore
        inv_rho = 1 / self.rho_0 # TODO: compute rho per cell
        z = self.dt * self.inv_dx * self.inv_dx * inv_rho
        for i, j in ti.ndrange((1, self.n_grid), (1, self.n_grid)):
            if self.cell_classification[i, j] == Classification.Interior:
                l = p[i - 1, j]
                r = p[i + 1, j]
                t = p[i, j + 1]
                b = p[i, j - 1]
                Ap[i, j] = p[i, j] * self.cell_inv_lambda[i, j]
                Ap[i, j] *= self.cell_JP[i, j] * (1 / (self.cell_JE[i, j] * self.dt))
                Ap[i, j] += z * (4.0 * p[i, j] - l - r - t - b)

    @ti.kernel
    def compute_b(self):
        z = self.inv_dx * self.inv_dx
        # for i, j in self.cell_pressure:
        for i, j in ti.ndrange((1, self.n_grid), (1, self.n_grid)):
            if self.cell_classification[i, j] == Classification.Interior:
                self.b[i, j] = -1 * (self.cell_JE[i, j] - 1) / (self.dt * self.cell_JE[i, j])
                self.b[i, j] -= z * (self.face_velocity_x[i + 1, j] - self.face_velocity_x[i, j])
                self.b[i, j] -= z * (self.face_velocity_y[i, j + 1] - self.face_velocity_y[i, j])

            if self.cell_classification[i, j] == Classification.Interior:
                print("-" * 100)
                print("UNDERP ->", self.cell_pressure[i, j])
                print("CELLJE ->", self.cell_JE[i, j])
                print("CELLJP ->", self.cell_JP[i, j])
                print("X_VELO ->", self.face_velocity_x[i, j])
                print("Y_VELO ->", self.face_velocity_y[i, j])
                print("DTTTTT ->", self.dt)
                print("LAMBDA ->", self.cell_inv_lambda[i, j])
                print("BBBBBB ->", self.b[i, j])

    def solve_pressure(self):
        self.compute_b()
        MatrixFreeCG(A=self.Ax, b=self.b, x=self.cell_pressure, maxiter=1000, tol=1e-5, quiet=False)

    @ti.kernel
    def apply_pressure(self):
        inv_rho = 1 / self.rho_0  # TODO: compute inv_rho from face volumes per cell
        z = self.dt * inv_rho * self.inv_dx * self.inv_dx
        for i, j in self.face_mass_x:
            # if self.cell_classification[i - 1, j] == Classification.Interior:
            self.face_velocity_x[i, j] -= z * self.sample(self.cell_pressure, i - 1, j)
            # if self.cell_classification[i, j] == Classification.Interior:
            self.face_velocity_x[i, j] -= z * self.sample(self.cell_pressure, i, j)
            # self.face_velocity_x[i, j] -= z * self.sample(self.cell_pressure, i, j + 1)
            # self.face_velocity_x[i, j] -= z * self.sample(self.cell_pressure, i, j - 1)
            # self.face_velocity_x[i, j] -= 4 * z * self.cell_pressure[i, j]
            # self.face_velocity_x[i, j] -= 4 * z * self.cell_pressure[i, j]
        for i, j in self.face_mass_y:
            # self.face_velocity_y[i, j] -= z * self.sample(self.cell_pressure, i - 1, j)
            # self.face_velocity_y[i, j] -= z * self.sample(self.cell_pressure, i + 1, j)
            # if self.cell_classification[i, j - 1] == Classification.Interior:
            self.face_velocity_y[i, j] -= z * self.sample(self.cell_pressure, i, j - 1)
            # if self.cell_classification[i, j] == Classification.Interior:
            self.face_velocity_y[i, j] -= z * self.sample(self.cell_pressure, i, j)
            # self.face_velocity_y[i, j] -= 4 * z * self.cell_pressure[i, j]
            # self.face_velocity_y[i, j] -= 4 * z * self.cell_pressure[i, j]

    @ti.kernel
    def grid_to_particle(self):
        for p in self.particle_position:
            x_stagger = ti.Vector([self.dx / 2, 0])
            y_stagger = ti.Vector([0, self.dx / 2])
            c_base = (self.particle_position[p] * self.inv_dx - 0.5).cast(int)
            x_base = (self.particle_position[p] * self.inv_dx - (x_stagger + 0.5)).cast(int)
            y_base = (self.particle_position[p] * self.inv_dx - (y_stagger + 0.5)).cast(int)
            c_fx = self.particle_position[p] * self.inv_dx - c_base.cast(float)
            x_fx = self.particle_position[p] * self.inv_dx - x_base.cast(float)
            y_fx = self.particle_position[p] * self.inv_dx - y_base.cast(float)
            # TODO: use the tighter quadratic weights?
            c_w = [0.5 * (1.5 - c_fx) ** 2, 0.75 - (c_fx - 1) ** 2, 0.5 * (c_fx - 0.5) ** 2]
            x_w = [0.5 * (1.5 - x_fx) ** 2, 0.75 - (x_fx - 1) ** 2, 0.5 * (x_fx - 0.5) ** 2]
            y_w = [0.5 * (1.5 - y_fx) ** 2, 0.75 - (y_fx - 1) ** 2, 0.5 * (y_fx - 0.5) ** 2]

            bx = ti.Vector.zero(float, 2)
            by = ti.Vector.zero(float, 2)
            nv = ti.Vector.zero(float, 2)
            nt = 0.0
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                c_weight = c_w[i][0] * c_w[j][1]
                x_weight = x_w[i][0] * x_w[j][1]
                y_weight = y_w[i][0] * y_w[j][1]
                x_dpos = offset.cast(float) - x_fx
                y_dpos = offset.cast(float) - y_fx
                x_velocity = x_weight * self.face_velocity_x[x_base + offset]
                y_velocity = y_weight * self.face_velocity_y[y_base + offset]
                nv += [x_velocity, y_velocity]
                bx += x_velocity * x_dpos
                by += y_velocity * y_dpos
                nt += c_weight * self.cell_temperature[c_base + offset]

            cx = 4 * self.inv_dx * bx  # C = B @ (D^(-1)), 1 / dx cancelled out by dx in dpos.
            cy = 4 * self.inv_dx * by  # C = B @ (D^(-1)), 1 / dx cancelled out by dx in dpos.
            self.particle_C[p] = ti.Matrix([[cx[0], cy[0]], [cx[1], cy[1]]])  # pyright: ignore
            self.particle_color[p] = Color.Water if self.particle_phase[p] == Phase.Water else Color.Ice
            self.particle_position[p] += self.dt * nv
            self.particle_temperature[p] = nt
            self.particle_velocity[p] = nv

    @ti.kernel
    def reset(self):
        for i in range(self.n_particles):
            radius = R * ti.sqrt(ti.random())
            self.particle_position[i] = [
                radius * (ti.sin(THETAS[i]) * 0.5) + 0.5,
                radius * (ti.cos(THETAS[i]) * 0.5) + 0.15,
            ]
            # self.particle_position[i] = [(ti.random() * 0.1) + 0.45, (ti.random() * 0.1) + 0.001]
            # self.particle_position[i] = [(ti.random() * 0.1) + 0.45, (ti.random() * 0.1) + 0.1]
            self.particle_mass[i] = self.particle_vol * self.rho_0
            self.particle_inv_lambda[i] = 1 / self.lambda_0[None]
            self.particle_FE[i] = ti.Matrix([[1, 0], [0, 1]])
            self.particle_C[i] = ti.Matrix.zero(float, 2, 2)
            self.particle_phase[i] = Phase.Water
            self.particle_color[i] = Color.Water
            self.particle_velocity[i] = [0, 0]
            self.particle_JE[i] = 1
            self.particle_JP[i] = 1

    def handle_events(self):
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.reset()
            elif self.window.event.key in [ti.GUI.BACKSPACE, "s"]:
                self.should_write_to_disk = not self.should_write_to_disk
            elif self.window.event.key in [ti.GUI.SPACE, "p"]:
                self.is_paused = not self.is_paused
            elif self.window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                self.window.running = False  # Stop the simulation

    def substep(self):
        if not self.is_paused:
            for _ in range(int(2e-3 // self.dt)):
                self.reset_grids()
                self.particle_to_grid()
                self.momentum_to_velocity()
                self.classify_cells()
                self.solve_pressure()
                # print(self.b)
                # print(self.Ax)
                # self.apply_pressure()
                # self.compute_divergence()
                # print(self.cell_divergence)
                self.grid_to_particle()

    def show_parameters(self, subwindow):
        self.theta_c[None] = subwindow.slider_float("theta_c", self.theta_c[None], 1e-2, 3.5e-2)
        self.theta_s[None] = subwindow.slider_float("theta_s", self.theta_s[None], 5.0e-3, 10e-3)
        self.zeta[None] = subwindow.slider_int("zeta", self.zeta[None], 3, 20)
        self.nu[None] = subwindow.slider_float("nu", self.nu[None], 0.1, 0.4)
        self.E[None] = subwindow.slider_float("E", self.E[None], 4.8e4, 2.8e5)
        self.lambda_0[None] = self.E[None] * self.nu[None] / ((1 + self.nu[None]) * (1 - 2 * self.nu[None]))
        self.mu_0[None] = self.E[None] / (2 * (1 + self.nu[None]))

    def show_buttons(self, subwindow):
        if subwindow.button(" Stop recording  " if self.should_write_to_disk else " Start recording "):
            # This button toggles between saving frames and not saving frames.
            self.should_write_to_disk = not self.should_write_to_disk
            if self.should_write_to_disk:
                # Create directory to dump frames, videos and GIFs.
                date = datetime.now().strftime("%d%m%Y_%H%M")
                output_dir = f"{self.parent_dir}/{date}"
                os.makedirs(output_dir)
                # Create a VideoManager to save frames, videos and GIFs.
                self.video_manager = ti.tools.VideoManager(
                    output_dir=output_dir,
                    framerate=24,
                    automatic_build=False,
                )
            else:
                # Convert stored frames to video and GIF.
                self.video_manager.make_video(gif=True, mp4=True)

        if subwindow.button(" Reset Particles "):
            self.reset()
        if subwindow.button(" Start Simulation"):
            self.is_paused = False

    def show_settings(self):
        if not self.is_paused:
            self.is_showing_settings = False
            return  # don't bother
        self.is_showing_settings = True
        with self.gui.sub_window("Settings", 0.01, 0.01, 0.98, 0.98) as subwindow:
            self.show_parameters(subwindow)
            self.show_buttons(subwindow)

    def render(self):
        self.canvas.set_background_color((0.054, 0.06, 0.09))
        self.canvas.circles(centers=self.particle_position, radius=0.0015, per_vertex_color=self.particle_color)
        if self.should_write_to_disk and not self.is_paused and not self.is_showing_settings:
            self.video_manager.write_frame(self.window.get_image_buffer_as_numpy())
        self.window.show()

    def run(self):
        self.reset()
        while self.window.running:
            self.handle_events()
            self.show_settings()
            self.substep()
            self.render()


def main():
    print("-" * 150)
    print("[Hint] Press R to [R]eset, P|SPACE to [P]ause/un[P]ause and S|BACKSPACE to [S]tart recording!")
    print("-" * 150)

    simulation = ThermodynamicModel(quality=QUALITY, n_particles=N_PARTICLES)
    simulation.run()


if __name__ == "__main__":
    main()
