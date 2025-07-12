# TODO: Use enums Water, Ice to hold everything instead 300000 different enums
from src.constants import (
    Capacity,
    Classification,
    ColorRGB,
    Conductivity,
    Density,
    Phase,
    Density,
    LatentHeat,
    State,
    Lambda,
    Mu,
)
from src.solvers import PressureSolver, HeatSolver
from src.parsing import should_use_b_i_computation

import taichi as ti

GRAVITY = -9.81


@ti.data_oriented
class MPM_Solver:
    def __init__(self, quality: int, max_particles: int):
        # MPM Parameters that are configuration independent
        self.n_particles = ti.field(dtype=ti.int32, shape=())
        self.max_particles = max_particles
        self.n_grid = 128 * quality
        self.n_cells = self.n_grid * self.n_grid
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = 1e-4 / quality
        self.vol_0_p = (self.dx * 0.5) ** 2
        self.n_dimensions = 2

        # The width of the simulation boundary in grid nodes and offsets to
        # guarantee that seeded particles always lie within the boundary:
        self.boundary_width = 3
        self.lower = self.boundary_width * self.dx
        self.upper = 1 - self.lower

        # Properties on MAC-faces.
        self.classification_x = ti.field(dtype=ti.i32, shape=(self.n_grid + 1, self.n_grid))
        self.classification_y = ti.field(dtype=ti.i32, shape=(self.n_grid, self.n_grid + 1))
        self.conductivity_x = ti.field(dtype=ti.f32, shape=(self.n_grid + 1, self.n_grid))
        self.conductivity_y = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid + 1))
        self.velocity_x = ti.field(dtype=ti.f32, shape=(self.n_grid + 1, self.n_grid))
        self.velocity_y = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid + 1))
        self.volume_x = ti.field(dtype=ti.f32, shape=(self.n_grid + 1, self.n_grid))
        self.volume_y = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid + 1))
        self.mass_x = ti.field(dtype=ti.f32, shape=(self.n_grid + 1, self.n_grid))
        self.mass_y = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid + 1))

        # Properties on MAC-cells.
        self.classification_c = ti.field(dtype=ti.i32, shape=(self.n_grid, self.n_grid))
        self.temperature_c = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.inv_lambda_c = ti.field(dtype=ti.f64, shape=(self.n_grid, self.n_grid))
        self.capacity_c = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.mass_c = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.JE_c = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.JP_c = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.J_c = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))

        # Properties on particles.
        self.conductivity_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.temperature_p = ti.field(dtype=ti.f32, shape=max_particles)
        # self.inv_lambda_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.position_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.velocity_p = ti.Vector.field(2, dtype=ti.f32, shape=max_particles)
        self.lambda_0_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.capacity_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.color_p = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.state_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.phase_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.mass_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.mu_0_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.FE_p = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)
        self.JE_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.JP_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.J_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.C_p = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)

        # Fields needed for the latent heat and phase change.
        self.heat_p = ti.field(dtype=ti.f32, shape=max_particles)  # U_p
        self.ambient_temperature = ti.field(dtype=ti.f32, shape=())

        # Variables controlled from the GUI, stored in fields to be accessed from compiled kernels.
        self.lambda_0 = ti.field(dtype=float, shape=())
        self.theta_c = ti.field(dtype=ti.f32, shape=())
        self.theta_s = ti.field(dtype=ti.f32, shape=())
        self.mu_0 = ti.field(dtype=float, shape=())
        self.zeta = ti.field(dtype=ti.i32, shape=())
        self.nu = ti.field(dtype=float, shape=())
        self.E = ti.field(dtype=float, shape=())

        # Poisson solvers for pressure and heat.
        self.pressure_solver = PressureSolver(self)
        self.heat_solver = HeatSolver(self)

        # Set the initial boundary:
        self.initialize_boundary()

    @ti.func
    def in_bounds(self, x: float, y: float) -> bool:
        return self.lower < x < self.upper and self.lower < y < self.upper

    @ti.func
    def is_valid(self, i: int, j: int) -> bool:
        return i >= 0 and i <= self.n_grid - 1 and j >= 0 and j <= self.n_grid - 1

    @ti.func
    def is_colliding(self, i: int, j: int) -> bool:
        return self.is_valid(i, j) and self.classification_c[i, j] == Classification.Colliding

    @ti.func
    def is_interior(self, i: int, j: int) -> bool:
        return self.is_valid(i, j) and self.classification_c[i, j] == Classification.Interior

    @ti.kernel
    def initialize_boundary(self):
        for i, j in self.classification_c:
            is_colliding = not (self.boundary_width <= i < self.n_grid - self.boundary_width)
            is_colliding |= not (self.boundary_width <= j < self.n_grid - self.boundary_width)
            if is_colliding:
                self.classification_c[i, j] = Classification.Colliding
            else:
                self.classification_c[i, j] = Classification.Empty

    @ti.kernel
    def reset_grids(self):
        for i, j in self.classification_x:
            self.conductivity_x[i, j] = 0
            self.velocity_x[i, j] = 0
            self.volume_x[i, j] = 0
            self.mass_x[i, j] = 0

        for i, j in self.classification_y:
            self.conductivity_y[i, j] = 0
            self.velocity_y[i, j] = 0
            self.volume_y[i, j] = 0
            self.mass_y[i, j] = 0

        for i, j in self.classification_c:
            self.temperature_c[i, j] = 0
            self.inv_lambda_c[i, j] = 0
            self.capacity_c[i, j] = 0
            self.mass_c[i, j] = 0
            self.JE_c[i, j] = 0
            self.JP_c[i, j] = 0
            self.J_c[i, j] = 0

    @ti.func
    def R(self, M: ti.types.matrix(2, 2, float)) -> ti.types.matrix(2, 2, float):  # pyright: ignore
        # TODO: this might not be needed, as the timestep is so small for explicit MPM anyway
        result = ti.Matrix.identity(float, 2) + M
        while ti.math.determinant(result) < 0:
            result = (ti.Matrix.identity(float, 2) + (0.5 * result)) ** 2
        return result

    @ti.kernel
    def particle_to_grid(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Deformation gradient update.
            self.FE_p[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore
            # TODO: R might not be needed for our small timesteps? then remove R and everything
            # self.FE_p[p] = self.R(self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore
            # TODO: could this just be simplified to: (or would this be unstable?)
            # self.FE_p[p] += (self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore

            # Remove the deviatoric component from each fluid particle:
            if self.phase_p[p] == Phase.Water:
                self.FE_p[p] = ti.sqrt(self.JE_p[p]) * ti.Matrix.identity(ti.f32, 2)

            # # Clamp singular values to simulate plasticity and elasticity:
            U, sigma, V = ti.svd(self.FE_p[p])
            self.JE_p[p] = 1.0
            for d in ti.static(range(self.n_dimensions)):
                singular_value = ti.f32(sigma[d, d])
                clamped = ti.f32(sigma[d, d])
                if self.phase_p[p] == Phase.Ice:
                    # Clamp singular values to [1 - theta_c, 1 + theta_s]
                    clamped = max(clamped, 1 - self.theta_c[None])
                    clamped = min(clamped, 1 + self.theta_s[None])
                self.JP_p[p] *= singular_value / clamped
                self.JE_p[p] *= clamped
                sigma[d, d] = clamped

            # TODO: if elasticity/plasticity is applied in the fluid phase, we also need this corrections:
            # if self.phase_p[p] == Phase.Water:
            #     self.FE_p[p] *= ti.sqrt(self.JP_p[p]) * (U @ sigma @ V.transpose())
            #     self.JE_p[p] = ti.math.determinant(self.FE_p[p])
            #     self.JP_p[p] = 1.0

            la, mu = self.lambda_0_p[p], self.mu_0_p[p]
            cauchy_stress = ti.Matrix.zero(ti.f32, 2, 2)
            if self.phase_p[p] == Phase.Ice:
                # Reconstruct elastic deformation gradient after plasticity
                self.FE_p[p] = U @ sigma @ V.transpose()

                # Apply ice hardening by adjusting Lame parameters: TODO: uncomment this after testing
                hardening = ti.max(0.1, ti.min(20, ti.exp(self.zeta[None] * (1.0 - self.JP_p[p]))))
                la, mu = la * hardening, mu * hardening

                # Compute D^(-1), which equals constant scaling for quadratic/cubic kernels.
                D_inv = 3 * self.inv_dx * self.inv_dx  # Cubic interpolation

                # Compute deviatoric Piola-Kirchhoff stress P(F), (JST16, Eqn. 52):
                # NOTE: further corrections to FE are only needed for fluid particles.
                piola_kirchhoff = 2 * mu * (self.FE_p[p] - U @ V.transpose())
                piola_kirchhoff = piola_kirchhoff @ self.FE_p[p].transpose()  # pyright: ignore

                # TODO: these should be the dilational part and is handled by the pressure projection?!
                # piola_kirchhoff += ti.Matrix.identity(ti.f32, 2) * la * self.JE_p[p] * (self.JE_p[p] - 1)

                # Cauchy stress times dt and D_inv
                cauchy_stress = -self.dt * self.vol_0_p * D_inv * piola_kirchhoff
                # cauchy_stress = -self.dt * self.vol_0_p * D_inv * piola_kirchhoff @ self.FE_p[p].transpose()
                # TODO: the 1 / J is probably cancelled out by V^n and leaves us V^0 (self.vol_0_p)?!
                # cauchy_stress = (1 / J) * (piola_kirchhoff @ self.FE_p[p].transpose())  # pyright: ignore

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            affine = cauchy_stress + self.mass_p[p] * self.C_p[p]
            affine_x = affine @ ti.Vector([1, 0])  # pyright: ignore
            affine_y = affine @ ti.Vector([0, 1])  # pyright: ignore
            # TODO: use cx, cy vectors here directly?
            # affine_x = cauchy_stress + self.mass_p[p] * self.c_x[p]
            # affine_y = cauchy_stress + self.mass_p[p] * self.c_y[p]

            # Lower left corner of the interpolation grid:
            base_x = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.0, 1.5])), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.5, 1.0])), dtype=ti.i32)
            base_c = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([0.5, 0.5])), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.0, 0.5])
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.5, 0.0])
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32)

            # Cubic kernels (JST16 Eqn. 122 with x=fx, abs(fx-1), abs(fx-2), (and abs(fx-3) for faces).
            # Based on https://www.bilibili.com/opus/662560355423092789
            # FIXME: the weights might be wrong?
            w_c = [
                ((-0.166 * dist_c**3) + (dist_c**2) - (2 * dist_c) + 1.33),
                ((0.5 * ti.abs(dist_c - 1.0) ** 3) - ((dist_c - 1.0) ** 2) + 0.66),
                ((0.5 * ti.abs(dist_c - 2.0) ** 3) - ((dist_c - 2.0) ** 2) + 0.66),
            ]
            w_x = [
                ((-0.166 * dist_x**3) + (dist_x**2) - (2 * dist_x) + 1.33),
                ((0.5 * ti.abs(dist_x - 1.0) ** 3) - ((dist_x - 1.0) ** 2) + 0.66),
                ((0.5 * ti.abs(dist_x - 2.0) ** 3) - ((dist_x - 2.0) ** 2) + 0.66),
                ((-0.166 * ti.abs(dist_x - 3.0) ** 3) + ((dist_x - 3.0) ** 2) - (2 * ti.abs(dist_x - 3.0)) + 1.33),
            ]
            w_y = [
                ((-0.166 * dist_y**3) + (dist_y**2) - (2 * dist_y) + 1.33),
                ((0.5 * ti.abs(dist_y - 1.0) ** 3) - ((dist_y - 1.0) ** 2) + 0.66),
                ((0.5 * ti.abs(dist_y - 2.0) ** 3) - ((dist_y - 2.0) ** 2) + 0.66),
                ((-0.166 * ti.abs(dist_y - 3.0) ** 3) + ((dist_y - 3.0) ** 2) - (2 * ti.abs(dist_y - 3.0)) + 1.33),
            ]

            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                weight_c = w_c[i][0] * w_c[j][1]

                # Rasterize temperature to cell centers.
                temperature = self.mass_p[p] * self.temperature_p[p]
                self.temperature_c[base_c + offset] += weight_c * temperature

                # Rasterize capacity to cell centers.
                capacity = self.mass_p[p] * self.capacity_p[p]
                self.capacity_c[base_c + offset] += weight_c * capacity

                # Rasterize mass to cell centers.
                self.mass_c[base_c + offset] += weight_c * self.mass_p[p]

                # Rasterize lambda (inverse) to cell centers.
                inv_lambda = self.mass_p[p] / la
                self.inv_lambda_c[base_c + offset] += weight_c * inv_lambda

                # TODO: remove this:We use JE^n, JP^n from the last timestep for the transfers, the updated
                # values will be assigned to the corresponding field at the end of the loop.
                self.JE_c[base_c + offset] += weight_c * self.mass_p[p] * self.JE_p[p]
                self.JP_c[base_c + offset] += weight_c * self.mass_p[p] * self.JP_p[p]
                # TODO: the paper wants to rasterize JE, J and then set JP = J / JE, but this makes no difference
                # self.J_c[base_c + offset] += weight_c * self.mass_p[p] * self.J_p[p]

            for i, j in ti.static(ti.ndrange(4, 4)):
                offset = ti.Vector([i, j])
                weight_x = w_x[i][0] * w_x[j][1]
                weight_y = w_y[i][0] * w_y[j][1]
                dpos_x = ti.cast(offset - dist_x, ti.f32) * self.dx
                dpos_y = ti.cast(offset - dist_y, ti.f32) * self.dx

                # Rasterize mass to grid faces.
                self.mass_x[base_x + offset] += weight_x * self.mass_p[p]
                self.mass_y[base_y + offset] += weight_y * self.mass_p[p]

                # Rasterize velocity to grid faces.
                velocity_x = self.mass_p[p] * self.velocity_p[p][0] + affine_x @ dpos_x
                velocity_y = self.mass_p[p] * self.velocity_p[p][1] + affine_y @ dpos_y
                self.velocity_x[base_x + offset] += weight_x * velocity_x
                self.velocity_y[base_y + offset] += weight_y * velocity_y

                # Rasterize conductivity to grid faces.
                conductivity = self.mass_p[p] * self.conductivity_p[p]
                self.conductivity_x[base_x + offset] += weight_x * conductivity
                self.conductivity_y[base_y + offset] += weight_y * conductivity

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.velocity_x:
            if (mass_x := self.mass_x[i, j]) > 0:
                self.velocity_x[i, j] /= mass_x
                collision_right = i >= (self.n_grid - self.boundary_width) and self.velocity_x[i, j] > 0
                collision_left = i <= self.boundary_width and self.velocity_x[i, j] < 0
                if collision_left or collision_right:
                    self.velocity_x[i, j] = 0

        for i, j in self.velocity_y:
            if (mass_y := self.mass_y[i, j]) > 0:
                self.velocity_y[i, j] /= mass_y
                self.velocity_y[i, j] += GRAVITY * self.dt
                collision_top = j >= (self.n_grid - self.boundary_width) and self.velocity_y[i, j] > 0
                collision_bottom = j <= self.boundary_width and self.velocity_y[i, j] < 0
                if collision_top or collision_bottom:
                    self.velocity_y[i, j] = 0

        for i, j in self.mass_c:
            if (mass_c := self.mass_c[i, j]) > 0:
                self.temperature_c[i, j] /= mass_c
                self.inv_lambda_c[i, j] /= mass_c
                self.capacity_c[i, j] /= mass_c
                self.JE_c[i, j] /= mass_c
                self.JP_c[i, j] /= mass_c
                # TODO: the paper wants to rasterize JE, J and then set JP = J / JE, but this makes no difference
                # self.J_c[i, j] *= 1 / self.mass_c[i, j]
                # self.JP_c[i, j] = self.J_c[i, j] / self.JE_c[i, j]

    @ti.kernel
    def classify_cells(self):
        # TODO: is it even needed to classify faces?
        # for i, j in self.classification_x:
        #     # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.
        #
        #     # The simulation boundary is always colliding.
        #     x_face_is_colliding = i >= (self.n_grid - self.boundary_width) or i <= self.boundary_width
        #     x_face_is_colliding |= j >= (self.n_grid - self.boundary_width) or j <= self.boundary_width
        #     if x_face_is_colliding:
        #         self.classification_x[i, j] = Classification.Colliding
        #         continue
        #
        #     # For convenience later on: a face is marked interior if it has mass.
        #     if self.mass_x[i, j] > 0:
        #         self.classification_x[i, j] = Classification.Interior
        #         continue
        #
        #     # All remaining faces are empty.
        #     self.classification_x[i, j] = Classification.Empty
        #
        # for i, j in self.classification_y:
        #     # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.
        #
        #     # The simulation boundary is always colliding.
        #     y_face_is_colliding = i >= (self.n_grid - self.boundary_width) or i <= self.boundary_width
        #     y_face_is_colliding |= j >= (self.n_grid - self.boundary_width) or j <= self.boundary_width
        #     if y_face_is_colliding:
        #         self.classification_y[i, j] = Classification.Colliding
        #         continue
        #
        #     # For convenience later on: a face is marked interior if it has mass.
        #     if self.mass_y[i, j] > 0:
        #         self.classification_y[i, j] = Classification.Interior
        #         continue
        #
        #     # All remaining faces are empty.
        #     self.classification_y[i, j] = Classification.Empty

        for i, j in self.classification_c:
            # TODO: Colliding cells are either assigned the temperature of the object it collides with
            # or a user-defined spatially-varying value depending on the setup.

            # NOTE: currently this is only set in the beginning, as the colliding boundary is fixed:
            # TODO: decide if this should be done here for better integration of colliding objects
            if self.is_colliding(i, j):
                continue

            # A cell is marked as colliding if all of its surrounding faces are colliding.
            # cell_is_colliding = self.classification_x[i, j] == Classification.Colliding
            # cell_is_colliding &= self.classification_x[i + 1, j] == Classification.Colliding
            # cell_is_colliding &= self.classification_y[i, j] == Classification.Colliding
            # cell_is_colliding &= self.classification_y[i, j + 1] == Classification.Colliding
            # if cell_is_colliding:
            #     # self.cell_temperature[i, j] = self.ambient_temperature[None]
            #     self.classification_c[i, j] = Classification.Colliding
            #     continue

            # A cell is interior if the cell and all of its surrounding faces have mass.
            cell_is_interior = self.mass_c[i, j] > 0
            cell_is_interior &= self.mass_x[i, j] > 0 and self.mass_x[i + 1, j] > 0
            cell_is_interior &= self.mass_y[i, j] > 0 and self.mass_y[i, j + 1] > 0
            if cell_is_interior:
                self.classification_c[i, j] = Classification.Interior
                continue

            # All remaining cells are empty.
            self.classification_c[i, j] = Classification.Empty

            # If the free surface is being enforced as a Dirichlet temperature condition,
            # the ambient air temperature is recorded for empty cells.
            self.temperature_c[i, j] = self.ambient_temperature[None]

    @ti.kernel
    def compute_volumes(self):
        # TODO: this seems to be wrong, the paper has a sum over CDFs
        control_volume = 0.5 * self.dx * self.dx
        for i, j in self.classification_c:
            if self.classification_c[i, j] == Classification.Interior:
                self.volume_x[i + 1, j] += control_volume
                self.volume_y[i, j + 1] += control_volume
                self.volume_x[i, j] += control_volume
                self.volume_y[i, j] += control_volume

    @ti.kernel
    def grid_to_particle(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Lower left corner of the interpolation grid:
            base_x = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([0.5, 1.0])), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.0, 0.5])), dtype=ti.i32)
            base_c = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([0.5, 0.5])), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.0, 0.5])
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.5, 0.0])
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32)

            # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2)
            # Based on https://www.bilibili.com/opus/662560355423092789
            w_c = [0.5 * (1.5 - dist_c) ** 2, 0.75 - (dist_c - 1) ** 2, 0.5 * (dist_c - 0.5) ** 2]
            w_x = [0.5 * (1.5 - dist_x) ** 2, 0.75 - (dist_x - 1) ** 2, 0.5 * (dist_x - 0.5) ** 2]
            w_y = [0.5 * (1.5 - dist_y) ** 2, 0.75 - (dist_y - 1) ** 2, 0.5 * (dist_y - 0.5) ** 2]

            # NOTE: Computing b_i and setting c_i <- D^{-1} * b_i
            b_x = ti.Vector.zero(ti.f32, 2)
            b_y = ti.Vector.zero(ti.f32, 2)

            # NOTE: Computing c_i directly:
            c_x = ti.Vector.zero(ti.f32, 2)
            c_y = ti.Vector.zero(ti.f32, 2)
            g_x = [dist_x - 1.5, (-2) * (dist_x - 1), dist_x - 0.5]
            g_y = [dist_y - 1.5, (-2) * (dist_y - 1), dist_y - 0.5]

            next_velocity = ti.Vector.zero(ti.f32, 2)
            next_temperature = 0.0
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                c_weight = w_c[i][0] * w_c[j][1]
                x_weight = w_x[i][0] * w_x[j][1]
                y_weight = w_y[i][0] * w_y[j][1]
                next_temperature += c_weight * self.temperature_c[base_c + offset]
                x_velocity = x_weight * self.velocity_x[base_x + offset]
                y_velocity = y_weight * self.velocity_y[base_y + offset]
                next_velocity += [x_velocity, y_velocity]

                if ti.static(should_use_b_i_computation):
                    # NOTE: Computing b_i and setting c_i <- D^{-1} * b_i
                    x_dpos = ti.cast(offset, ti.f32) - dist_x
                    y_dpos = ti.cast(offset, ti.f32) - dist_y
                    b_x += x_velocity * x_dpos
                    b_y += y_velocity * y_dpos
                else:
                    # NOTE: Computing c_i directly:
                    grad_x = ti.Vector([g_x[i][0] * w_x[j][1], w_x[i][0] * g_x[j][1]])
                    grad_y = ti.Vector([g_y[i][0] * w_y[j][1], w_y[i][0] * g_y[j][1]])
                    c_x += self.velocity_x[base_x + offset] * grad_x
                    c_y += self.velocity_y[base_y + offset] * grad_y

            if ti.static(should_use_b_i_computation):
                c_x = 3 * self.inv_dx * b_x  # C = B @ (D^(-1)), inv_dx cancelled out by dx in dpos
                c_y = 3 * self.inv_dx * b_y  # C = B @ (D^(-1)), inv_dx cancelled out by dx in dpos

            self.C_p[p] = ti.Matrix([[c_x[0], c_y[0]], [c_x[1], c_y[1]]])  # pyright: ignore
            self.position_p[p] += self.dt * next_velocity
            self.velocity_p[p] = next_velocity

            # # DONE: set temperature for empty cells
            # # DONE: set temperature for particles, ideally per geometry
            # # DONE: set heat capacity per particle depending on phase
            # # DONE: set heat conductivity per particle depending on phase
            # # DONE: set particle mass per phase
            # # DONE: set E and nu for each particle depending on phase
            # # DONE: apply latent heat
            # # TODO: move this to a ti.func? (or keep this here but assign values in func and use when adding particles)
            # # TODO: set theta_c, theta_s per phase? Water probably wants very small values, ice depends on temperature
            # # TODO: in theory all of the constitutive parameters must be functions of temperature
            # #       in the ice phase to range from solid ice to slushy ice?

            # # Initially, we allow each particle to freely change its temperature according to the heat equation.
            # # But whenever the freezing point is reached, any additional temperature change is multiplied by
            # # conductivity and mass and added to the buffer, with the particle temperature kept unchanged.
            # if (self.phase_p[p] == Phase.Ice) and (next_temperature >= 0):
            #     # Ice reached the melting point, additional temperature change is added to heat buffer.
            #     difference = next_temperature - self.temperature_p[p]
            #     self.heat_p[p] += self.conductivity_p[p] * self.mass_p[p] * difference

            #     # If the heat buffer is full the particle changes its phase to water,
            #     # everything is then reset according to the new phase.
            #     if self.heat_p[p] >= LatentHeat.Water:
            #         self.lambda_0_p[p] = Lambda.Water
            #         self.mu_0_p[p] = Mu.Water
            #         self.capacity_p[p] = Capacity.Water
            #         self.conductivity_p[p] = Conductivity.Water
            #         self.color_p[p] = ColorRGB.Water
            #         self.temperature_p[p] = 0.0
            #         self.phase_p[p] = Phase.Water
            #         self.mass_p[p] = self.vol_0_p * Density.Water
            #         self.heat_p[p] = LatentHeat.Water

            # elif (self.phase_p[p] == Phase.Water) and (next_temperature < 0):
            #     # Water particle reached the freezing point, additional temperature change is added to heat buffer.
            #     difference = next_temperature - self.temperature_p[p]
            #     self.heat_p[p] += self.conductivity_p[p] * self.mass_p[p] * difference

            #     # If the heat buffer is empty the particle changes its phase to ice,
            #     # everything is then reset according to the new phase.
            #     if self.heat_p[p] <= LatentHeat.Ice:
            #         self.lambda_0_p[p] = Lambda.Ice
            #         self.mu_0_p[p] = Mu.Ice
            #         self.capacity_p[p] = Capacity.Ice
            #         self.color_p[p] = ColorRGB.Ice
            #         self.conductivity_p[p] = Conductivity.Ice
            #         self.temperature_p[p] = 0.0
            #         self.phase_p[p] = Phase.Ice
            #         self.mass_p[p] = self.vol_0_p * Density.Ice
            #         self.heat_p[p] = LatentHeat.Ice

            # else:
            #     # Freely change temperature according to heat equation.
            #     self.temperature_p[p] = next_temperature
