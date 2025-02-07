from taichi.linalg import SparseMatrixBuilder, SparseSolver
from enums import Classification, Color, Phase, State
from ps import Pressure_MGPCGSolver
import taichi as ti
import numpy as np

from experiments.poisson_solver import PCG

WATER_CONDUCTIVITY = 0.55  # Water: 0.55, Ice: 2.33
ICE_CONDUCTIVITY = 2.33
WATER_HEAT_CAPACITY = 4.186  # j/dC
ICE_HEAT_CAPACITY = 2.093  # j/dC
LATENT_HEAT = 0.334  # J/kg
GRAVITY = -9.81


@ti.data_oriented
class Solver:
    def __init__(self, quality: int, max_particles: int):
        # MPM Parameters that are configuration independent
        self.n_particles = ti.field(dtype=ti.int32, shape=())
        self.current_frame = ti.field(dtype=ti.int32, shape=())
        self.n_grid = 128 * quality
        self.n_cells = self.n_grid * self.n_grid
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = 1e-4 / quality
        self.rho_0 = 4e2
        self.particle_vol = (self.dx * 0.1) ** 2
        self.n_dimensions = 2

        # The width of the simulation boundary in grid nodes.
        self.boundary_width = 3

        # Offset to correct coordinates such that the origin lies within the boundary,
        # added to each position vector when loading a new configuration.
        self.boundary_offset = 1 - ((self.n_grid - self.boundary_width) * self.dx)

        # Parameters to control melting/freezing
        # TODO: these are variables and need toof the particle, be put into fields
        # TODO: these depend not only on phase, but also on temperature,
        #       so ideally they are functions of these two variables
        # self.heat_conductivity = 0.55 # Water: 0.55, Ice: 2.33
        # self.heat_capacity = 4.186 # Water: 4.186, Ice: 2.093 (j/dC)
        # self.latent_heat = 0.334 # in J/kg

        # Properties on MAC-faces.
        self.face_classification_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_classification_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.face_conductivity_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_conductivity_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.face_velocity_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_velocity_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.face_mass_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_mass_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))

        # Properties on MAC-cells.
        self.cell_classification = ti.field(dtype=ti.int8, shape=(self.n_grid, self.n_grid))
        self.cell_temperature = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_divergence = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_capacity = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_mass = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))

        # TODO: are these all needed?
        self.cell_inv_lambda = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_JE = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid + 1))
        self.cell_JP = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid + 1))

        # These fields will be used to solve for pressure, where Ax = b
        self.face_volume_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_volume_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.cell_pressure = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.right_hand_side = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))

        # FIXME: just for testing
        # self.Ap = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.pressure_solver = Pressure_MGPCGSolver(
            self.n_grid,
            self.n_grid,
            self.face_velocity_x,
            self.face_velocity_y,
            self.dt,
            self.cell_JP,
            self.cell_JE,
            self.cell_inv_lambda,
            self.cell_classification,
            multigrid_level=4,
            # pre_and_post_smoothing=pre_and_post_smoothing,
            # bottom_smoothing=bottom_smoothing,
        )

        # Properties on particles.
        self.particle_conductivity = ti.field(dtype=ti.float32, shape=max_particles)
        self.particle_temperature = ti.field(dtype=ti.float32, shape=max_particles)
        self.particle_inv_lambda = ti.field(dtype=ti.float32, shape=max_particles)
        self.particle_position = ti.Vector.field(2, dtype=float, shape=max_particles)
        self.particle_velocity = ti.Vector.field(2, dtype=float, shape=max_particles)
        self.particle_capacity = ti.field(dtype=ti.float32, shape=max_particles)
        self.particle_color = ti.Vector.field(3, dtype=float, shape=max_particles)
        self.particle_phase = ti.field(dtype=ti.float32, shape=max_particles)
        self.particle_mass = ti.field(dtype=ti.float32, shape=max_particles)
        self.particle_FE = ti.Matrix.field(2, 2, dtype=float, shape=max_particles)
        self.particle_C = ti.Matrix.field(2, 2, dtype=float, shape=max_particles)
        self.particle_JE = ti.field(dtype=float, shape=max_particles)
        self.particle_JP = ti.field(dtype=float, shape=max_particles)

        # Fields needed to implement sources (TODO: and sinks), the state will be set to
        # active once the activation threshold (frame) is reached. Active particles in
        # p_active_position will be drawn, all other particles are hidden until active.
        self.p_activation_threshold = ti.field(dtype=int, shape=max_particles)
        self.p_activation_state = ti.field(dtype=int, shape=max_particles)
        self.p_active_position = ti.Vector.field(2, dtype=ti.float32, shape=max_particles)

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

    @ti.kernel
    def reset_grids(self):
        # TODO: not all of these need to be reset
        # TODO: performance can be gained by bundling fields in loops
        # self.cell_classification.fill(Classification.Empty)
        self.face_velocity_x.fill(0)
        self.face_velocity_y.fill(0)
        self.cell_pressure.fill(0)
        self.face_mass_x.fill(0)
        self.face_mass_y.fill(0)
        self.cell_mass.fill(0)
        self.cell_JE.fill(1)
        self.cell_JP.fill(1)

        # TODO: implemented volume computation
        self.face_volume_x.fill(0)
        self.face_volume_y.fill(0)

    @ti.kernel
    def particle_to_grid(self):
        for p in ti.ndrange(self.n_particles[None]):
            # Check whether the particle can be activated.
            if self.p_activation_threshold[p] == self.current_frame[None]:
                self.p_activation_state[p] = State.Active

            # We only update currently active particles.
            if self.p_activation_state[p] == State.Inactive:
                continue

            # Deformation gradient update.
            self.particle_FE[p] = (ti.Matrix.identity(float, 2) + self.dt * self.particle_C[p]) @ self.particle_FE[p]

            # Clamp singular values to simulate plasticity and elasticity.
            U, sigma, V = ti.svd(self.particle_FE[p])
            JE, JP = 1.0, 1.0
            for d in ti.static(range(self.n_dimensions)):
                singular_value = float(sigma[d, d])
                # Clamp singular values to [1 - theta_c, 1 + theta_s]
                if self.particle_phase[p] == Phase.Ice:
                    clamped = singular_value
                    clamped = max(singular_value, 1 - self.theta_c[None])
                    clamped = min(clamped, 1 + self.theta_s[None])
                    sigma[d, d] = clamped
                    JP *= singular_value / clamped
                    JE *= clamped
                else:
                    JP *= singular_value
                    JE *= singular_value

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
            stress = 2 * mu * (self.particle_FE[p] - U @ V.transpose())
            stress = stress @ self.particle_FE[p].transpose()  # pyright: ignore
            stress += ti.Matrix.identity(float, 2) * la * JE * (JE - 1)

            # Compute D^(-1), which equals constant scaling for quadratic/cubic kernels.
            D_inv = 4 * self.inv_dx * self.inv_dx  # Quadratic interpolation
            # D_inv = 3 * self.inv_dx * self.inv_dx  # Cubic interpolation

            # TODO: What happens here exactly? Something with Cauchy-stress?
            stress *= -self.dt * self.particle_vol * D_inv

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            affine = stress + self.particle_mass[p] * self.particle_C[p]
            x_affine = affine @ ti.Vector([1, 0])  # pyright: ignore
            y_affine = affine @ ti.Vector([0, 1])  # pyright: ignore

            # We use an additional offset of 0.5 for element-wise flooring.
            x_stagger = ti.Vector([self.dx / 2, 0])
            y_stagger = ti.Vector([0, self.dx / 2])
            c_base = (self.particle_position[p] * self.inv_dx - 0.5).cast(int)  # pyright: ignore
            x_base = (self.particle_position[p] * self.inv_dx - (x_stagger + 0.5)).cast(int)  # pyright: ignore
            y_base = (self.particle_position[p] * self.inv_dx - (y_stagger + 0.5)).cast(int)  # pyright: ignore
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

                # TODO: implement proper volume computation
                self.face_volume_x[x_base + offset] += x_weight * self.particle_vol
                self.face_volume_y[x_base + offset] += y_weight * self.particle_vol

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
                collision_left = i < self.boundary_width and self.face_velocity_x[i, j] < 0
                collision_right = i > (self.n_grid - self.boundary_width) and self.face_velocity_x[i, j] > 0
                if collision_left or collision_right:
                    self.face_velocity_x[i, j] = 0
        for i, j in self.face_mass_y:
            if self.face_mass_y[i, j] > 0:  # No need for epsilon here
                self.face_velocity_y[i, j] *= 1 / self.face_mass_y[i, j]
                self.face_velocity_y[i, j] += self.dt * GRAVITY
                collision_top = j > (self.n_grid - self.boundary_width) and self.face_velocity_y[i, j] > 0
                collision_bottom = j < self.boundary_width and self.face_velocity_y[i, j] < 0
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
            is_interior = self.cell_mass[i, j] > 0
            is_interior &= self.face_mass_x[i, j] > 0
            is_interior &= self.face_mass_y[i, j] > 0
            is_interior &= self.face_mass_x[i + 1, j] > 0
            is_interior &= self.face_mass_y[i, j + 1] > 0

            if is_colliding:
                self.cell_classification[i, j] = Classification.Colliding
            elif is_interior:
                self.cell_classification[i, j] = Classification.Interior
            else:
                self.cell_classification[i, j] = Classification.Empty

    @ti.kernel
    def fill_linear_system(
        self,
        A: ti.types.sparse_matrix_builder(),  # pyright: ignore
        B: ti.types.sparse_matrix_builder(),  # pyright: ignore
        C: ti.types.sparse_matrix_builder(),  # pyright: ignore
        b: ti.types.ndarray(),  # pyright: ignore
    ):
        # TODO: Laplacian L needs to be filled according to paper.
        # DONE: Solution vector b needs to be filled according to paper.
        # delta = 0.00032  # relaxation
        # inv_rho = 1 / self.rho_0  # TODO: compute rho per cell
        # inv_rho = 1  # TODO: compute rho per cell
        # z = self.dt * self.inv_dx * self.inv_dx * inv_rho
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            # TODO: could be something like:
            # (see https://gitee.com/citadel2020/taichi_demos/blob/master/mgpcgflip/mgpcgflip.py)
            # s = 4.0
            # s -= float(self.is_solid(self.f[l], i - 1, j, _nx, _ny))
            # s -= float(self.is_solid(self.f[l], i + 1, j, _nx, _ny))
            # s -= float(self.is_solid(self.f[l], i, j - 1, _nx, _ny))
            # s -= float(self.is_solid(self.f[l], i, j + 1, _nx, _ny))
            # if self.cell_classification[i, j] == Classification.Interior:

            # Unraveled index.
            row = (i * self.n_grid) + j
            # row = (j * self.n_grid) + i

            # Fill the right hand side of the linear system.
            b[row] = (-1) * (self.cell_JE[i, j] - 1) / (self.dt * self.cell_JE[i, j])
            b[row] += self.inv_dx * (self.face_velocity_x[i + 1, j] - self.face_velocity_x[i, j])
            b[row] += self.inv_dx * (self.face_velocity_y[i, j + 1] - self.face_velocity_y[i, j])
            # b[row] += 4
            #
            # A[row, row] += 2

            # Fill the left hand side of the linear system.
            if self.cell_classification[i, j] == Classification.Interior:
                # if True:
                A[row, row] += self.cell_inv_lambda[i, j] * self.cell_JP[i, j] * (1 / (self.cell_JE[i, j] * self.dt))
                if (i != 0) and (self.cell_classification[i - 1, j] == Classification.Interior):
                    # B[row, row - self.n_grid] += -self.dt * (self.face_volume_x[i, j] / self.face_mass_x[i, j])
                    B[row, row - 1] += -self.dt * (self.face_volume_x[i, j] / self.face_mass_x[i, j])
                    B[row, row] += 1.0
                    C[row, row - self.n_grid] += -1.0
                    C[row, row] += 1.0

                if (i != self.n_grid - 1) and (self.cell_classification[i + 1, j] == Classification.Interior):
                    # B[row, row + self.n_grid] += -self.dt * (self.face_volume_x[i + 1, j] / self.face_mass_x[i + 1, j])
                    B[row, row + 1] += -self.dt * (self.face_volume_x[i + 1, j] / self.face_mass_x[i + 1, j])
                    B[row, row] += 1.0
                    C[row, row + self.n_grid] += -1.0
                    C[row, row] += 1.0

                if (j != 0) and (self.cell_classification[i, j - 1] == Classification.Interior):
                    B[row, row - 1] += -self.dt * (self.face_volume_y[i, j] / self.face_mass_y[i, j])
                    B[row, row] += 1.0
                    C[row, row - 1] += -1.0
                    C[row, row] += 1.0

                if (j != self.n_grid - 1) and (self.cell_classification[i, j + 1] == Classification.Interior):
                    B[row, row + 1] += -self.dt * (self.face_volume_y[i, j + 1] / self.face_mass_y[i, j + 1])
                    B[row, row] += 1.0
                    C[row, row + 1] += -1.0
                    C[row, row] += 1.0

            elif self.cell_classification[i, j] == Classification.Empty:  # Dirichlet
                # TODO: apply Dirichlet boundary condition
                A[row, row] += 1.0
                # B[row, row] += 1.0
                # C[row, row] += 1.0

            elif self.cell_classification[i, j] == Classification.Colliding:  # Neumann
                # TODO: apply Neumann boundary condition
                # TODO: the colliding classification doesn't exist atm
                A[row, row] += 1.0
                # B[row, row] += 1.0
                # C[row, row] += 1.0

        # delta = 0.00032  # relaxation
        # inv_rho = 1 / self.rho_0  # TODO: compute rho per cell
        # z = self.dt * self.inv_dx * self.inv_dx * inv_rho
        # for i, j in ti.ndrange((1, self.n_grid - 1), (1, self.n_grid - 1)):
        #     # for i, j in self.cell_pressure:
        #     Ap[i, j] = 0
        #     if self.cell_classification[i, j] == Classification.Interior:
        #         l = p[i - 1, j]
        #         r = p[i + 1, j]
        #         t = p[i, j + 1]
        #         b = p[i, j - 1]
        #
        #         # TODO: collect face volumes here first
        #
        #         # FIXME: the error is somewhere here:
        #         # Ap[i, j] = delta * p[i, j] * self.cell_inv_lambda[i, j]
        #         # Ap[i, j] *= self.cell_JP[i, j] * (1 / (self.cell_JE[i, j] * self.dt))
        #         Ap[i, j] += z * delta * (4.0 * p[i, j] - l - r - t - b)
        #         self.Ap[i, j] = Ap[i, j]  # FIXME: for testing only

    # @ti.kernel
    # # def fill_right_side(self, b: ti.template()):  # pyright: ignore
    # def fill_right_side(self, b: ti.types.ndarray()):  # pyright: ignore
    #     # for i, j in ti.ndrange((1, self.n_grid), (1, self.n_grid)):
    #     for i, j in self.cell_pressure:
    #         if self.cell_classification[i, j] == Classification.Interior:
    #             row = (i * self.n_grid) + j
    #             b[row] = -1 * (self.cell_JE[i, j] - 1) / (self.dt * self.cell_JE[i, j])
    #             b[row] += self.inv_dx * (self.face_velocity_x[i + 1, j] - self.face_velocity_x[i, j])
    #             b[row] += self.inv_dx * (self.face_velocity_y[i, j + 1] - self.face_velocity_y[i, j])
    #         # if self.cell_classification[i, j] == Classification.Interior:
    #         #     print("-" * 100)
    #         #     print("PRESSU ->", self.cell_pressure[i, j])
    #         #     print("CELLJE ->", self.cell_JE[i, j])
    #         #     print("CELLJP ->", self.cell_JP[i, j])
    #         #     print("X_VELO ->", self.face_velocity_x[i, j])
    #         #     print("Y_VELO ->", self.face_velocity_y[i, j])
    #         #     print("DTTTTT ->", self.dt)
    #         #     print("LAMBDA ->", self.cell_inv_lambda[i, j])
    #         #     print("BBBBBB ->", b[i, j])
    #         #     print("AAAAAA ->", self.Ap[i, j])
    # elif self.cell_classification[i, j - 1] == Classification.Empty:  # Dirichlet
    #     C[row, row] += 1.0
    #     D[row, row] += 1.0
    # elif self.cell_classification[i, j - 1] == Classification.Colliding:  # Neumann
    #     C[row, row] += 1.0
    #     D[row, row] += 1.0
    # elif self.cell_classification[i, j + 1] == Classification.Empty:  # Dirichlet
    #     C[row, row] += 1.0
    #     D[row, row] += 1.0
    # elif self.cell_classification[i, j + 1] == Classification.Colliding:  # Neumann
    #     C[row, row] += 1.0
    #     D[row, row] += 1.0
    # elif self.cell_classification[i - 1, j] == Classification.Empty:  # Dirichlet
    #     C[row, row] += 1.0
    #     D[row, row] += 1.0
    # elif self.cell_classification[i - 1, j] == Classification.Colliding:  # Neumann
    #     C[row, row] += 1.0
    #     D[row, row] += 1.0
    # elif self.cell_classification[i + 1, j] == Classification.Empty:  # Dirichlet
    #     C[row, row] += 1.0
    #     D[row, row] += 1.0
    # elif self.cell_classification[i + 1, j] == Classification.Colliding:  # Neumann
    #     C[row, row] += 1.0
    #     D[row, row] += 1.0

    @ti.kernel
    def compute_volumes(self):
        # TODO: Do this right
        w_4 = [0.041667, 0.45833, 0.45833, 0.041667]
        w_5 = [0.0026042, 0.1979125, 0.59896, 0.1979125, 0.0026042]
        for i, j in self.face_volume_x:
            for x_offset in ti.static(range(4)):
                for y_offset in ti.static(range(5)):
                    k = i - 2 + x_offset
                    l = j - 2 + y_offset
                    is_fluid = k >= 0 and k < self.n_grid and l >= 0 and l < self.n_grid
                    is_fluid &= self.cell_classification[k, l] == Classification.Interior
                    if is_fluid:
                        self.face_volume_x[i, j] += w_4[x_offset] * w_5[y_offset]

        for i, j in self.face_volume_y:
            for x_offset in ti.static(range(5)):
                for y_offset in ti.static(range(4)):
                    k = i - 2 + x_offset
                    l = j - 2 + y_offset
                    is_fluid = k >= 0 and k < self.n_grid and l >= 0 and l < self.n_grid
                    is_fluid &= self.cell_classification[k, l] == Classification.Interior
                    if is_fluid:
                        self.face_volume_y[i, j] += w_5[x_offset] * w_4[y_offset]

    @ti.kernel
    def fill_pressure_field(self, p: ti.types.ndarray()):  # pyright: ignore
        for i, j in self.cell_pressure:
            # TODO: move this to apply_pressure, delete self.cell_pressure
            row = (i * self.n_grid) + j
            self.cell_pressure[i, j] = p[row]
            # print(p[row])
            # print(self.cell_pressure[i, j])

    @ti.kernel
    def apply_pressure(self):
        # inv_rho = 1  # TODO: compute inv_rho from face volumes per cell
        inv_rho = 1 / self.rho_0  # TODO: compute rho per cell
        # z = self.dt * inv_rho * self.inv_dx * self.inv_dx  # TODO: are these even needed?
        z = 1
        for i, j in ti.ndrange((1, self.n_grid - 1), (1, self.n_grid - 1)):
            if self.cell_classification[i - 1, j] == Classification.Interior:
                self.face_velocity_x[i, j] -= z * self.cell_pressure[i - 1, j]
            if self.cell_classification[i + 1, j] == Classification.Interior:
                self.face_velocity_x[i, j] -= z * self.cell_pressure[i + 1, j]
            if self.cell_classification[i, j - 1] == Classification.Interior:
                self.face_velocity_y[i, j] -= z * self.cell_pressure[i, j - 1]
            if self.cell_classification[i, j + 1] == Classification.Interior:
                self.face_velocity_y[i, j] -= z * self.cell_pressure[i, j + 1]

    def correct_pressure(self):
        A = SparseMatrixBuilder(self.n_cells, self.n_cells, max_num_triplets=(self.n_cells * 5))
        B = SparseMatrixBuilder(self.n_cells, self.n_cells, max_num_triplets=(self.n_cells * 5))
        C = SparseMatrixBuilder(self.n_cells, self.n_cells, max_num_triplets=(self.n_cells * 5))
        b = ti.ndarray(ti.f32, shape=self.n_cells)
        self.fill_linear_system(A, B, C, b)
        L, M, N = A.build(), B.build(), C.build()
        R = L + (M @ N)

        # Solve the linear system.
        solver = SparseSolver()
        solver.compute(R)
        p = solver.solve(b)
        # for i in ti.ndrange(p.shape[0]):
        #     pressure = p[i]
        #     if pressure != 0:
        #         print(pressure)
        print("SUCCESS??? ->", solver.info())

        # Apply the pressure to the intermediate velocity field.
        self.fill_pressure_field(p)
        self.apply_pressure()

    @ti.kernel
    def grid_to_particle(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We only update active particles.
            if self.p_activation_state[p] == State.Inactive:
                continue

            x_stagger = ti.Vector([self.dx / 2, 0])
            y_stagger = ti.Vector([0, self.dx / 2])
            c_base = (self.particle_position[p] * self.inv_dx - 0.5).cast(int)  # pyright: ignore
            x_base = (self.particle_position[p] * self.inv_dx - (x_stagger + 0.5)).cast(int)  # pyright: ignore
            y_base = (self.particle_position[p] * self.inv_dx - (y_stagger + 0.5)).cast(int)  # pyright: ignore
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

            # The particle_position holds the positions for all particles, active and inactive,
            # only pushing the position into p_active_position will draw this particle.
            self.p_active_position[p] = self.particle_position[p]

    @ti.kernel
    def compute_divergence(self):
        # TODO: this method is only for debugging and should go to some test file or ???
        for i, j in self.cell_divergence:
            if self.cell_classification[i, j] == Classification.Interior:
                x_divergence = self.face_velocity_x[i + 1, j] - self.face_velocity_x[i, j]
                y_divergence = self.face_velocity_y[i, j + 1] - self.face_velocity_y[i, j]
                self.cell_divergence[i, j] = x_divergence + y_divergence
            else:
                self.cell_divergence[i, j] = 0

    def substep(self) -> None:
        self.current_frame[None] += 1
        for _ in range(int(2e-3 // self.dt)):
            self.reset_grids()
            self.particle_to_grid()
            self.momentum_to_velocity()
            self.classify_cells()

            # print("BEFORE:")
            # self.compute_divergence()
            # print(self.cell_pressure)
            # print(self.cell_divergence)

            # print(self.face_volume_x)
            # print(self.face_mass_x)
            # print("-" * 200)

            self.correct_pressure()

            # scale_A = self.dt / (self.n_grid * self.n_grid)
            # scale_b = 1 / self.n_grid
            # self.pressure_solver.system_init(scale_A, scale_b)
            # self.pressure_solver.solve(500)
            # self.cell_pressure.copy_from(self.pressure_solver.p)
            # self.apply_pressure()

            # print(self.cell_pressure)
            # print(self.cell_classification)

            # print("AFTER:")
            # self.compute_divergence()
            # print(self.cell_pressure)
            # print(self.cell_divergence)
            # print("-" * 200)

            self.grid_to_particle()
