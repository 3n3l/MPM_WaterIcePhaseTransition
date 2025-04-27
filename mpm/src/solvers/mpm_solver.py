# TODO: Use enums Water, Ice to hold everything instead 300000 different enums
from src.constants import (
    Capacity,
    Classification,
    Color,
    Conductivity,
    Density,
    Phase,
    Density,
    LatentHeat,
    State,
    PoissonsRatio,
    YoungsModulus,
)
from src.solvers import PressureSolver, HeatSolver

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
        # self.dt = 2e-3 / quality # TODO: better dt for fluid, test this for solid
        self.dt = 1e-3 / quality
        self.rho_0 = 1000  # TODO: this is kg/m^3 for water, what about ice?
        self.particle_vol = (self.dx * 0.5) ** 2
        self.n_dimensions = 2

        # The width of the simulation boundary in grid nodes and offsets to
        # guarantee that seeded particles always lie within the boundary:
        self.boundary_width = 3
        self.lower = self.boundary_width * self.dx
        self.upper = 1 - self.lower

        # Properties on MAC-faces.
        self.classification_x = ti.field(dtype=int, shape=(self.n_grid + 1, self.n_grid))
        self.classification_y = ti.field(dtype=int, shape=(self.n_grid, self.n_grid + 1))
        self.conductivity_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.conductivity_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.velocity_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.velocity_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.volume_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.volume_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.mass_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.mass_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))

        # Properties on MAC-cells.
        self.classification_c = ti.field(dtype=ti.int8, shape=(self.n_grid, self.n_grid))
        self.temperature_c = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.inv_lambda_c = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.capacity_c = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.pressure_c = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.mass_c = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.JE_c = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.JP_c = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.J_c = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))

        # Properties on particles.
        self.conductivity_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.temperature_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.inv_lambda_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.position_p = ti.Vector.field(2, dtype=float, shape=max_particles)
        self.velocity_p = ti.Vector.field(2, dtype=float, shape=max_particles)
        self.lambda_0_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.capacity_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.color_p = ti.Vector.field(3, dtype=float, shape=max_particles)
        self.state_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.phase_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.mass_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.mu_0_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.FE_p = ti.Matrix.field(2, 2, dtype=float, shape=max_particles)
        self.JE_p = ti.field(dtype=float, shape=max_particles)
        self.JP_p = ti.field(dtype=float, shape=max_particles)
        self.C_p = ti.Matrix.field(2, 2, dtype=float, shape=max_particles)

        # Fields needed for the latent heat and phase change.
        self.heat_p = ti.field(dtype=ti.float32, shape=max_particles)  # U_p
        self.ambient_temperature = ti.field(dtype=ti.float32, shape=())

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
            self.pressure_c[i, j] = 0
            self.capacity_c[i, j] = 0
            self.mass_c[i, j] = 0
            self.JE_c[i, j] = 0
            self.JP_c[i, j] = 0
            # self.J_c[i, j] = 0

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
            self.FE_p[p] = self.R(self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore

            la, mu = self.lambda_0_p[p], self.mu_0_p[p]
            U, sigma, V = ti.svd(self.FE_p[p])
            JE_p, JP_p, J_p = 1.0, 1.0, 1.0

            # Clamp singular values to simulate plasticity and elasticity.
            for d in ti.static(range(self.n_dimensions)):
                singular_value = float(sigma[d, d])
                clamped = singular_value
                if self.phase_p[p] == Phase.Ice:
                    # Clamp singular values to [1 - theta_c, 1 + theta_s]
                    clamped = max(clamped, 1 - self.theta_c[None])
                    clamped = min(clamped, 1 + self.theta_s[None])
                    sigma[d, d] = clamped
                JP_p *= singular_value / clamped
                JE_p *= clamped

            if self.phase_p[p] == Phase.Ice:
                # Reconstruct elastic deformation gradient after plasticity
                self.FE_p[p] = U @ sigma @ V.transpose()

                # Apply ice hardening by adjusting Lame parameters
                hardening = ti.max(0.1, ti.min(20, ti.exp(self.zeta[None] * (1.0 - JP_p))))
                la, mu = la * hardening, mu * hardening

                # NOTE: further corrections to FE, FP, JE, JP would go here:
                # FE <- JP^(1/d) * FE
                # FP <- JP^(-1/d) * FP
                # NOTE: this would make FP purely deviatoric and set JP = 1,
                #       while keeping a balance with FE, JE, that are then used
                #       to compute the deviatoric stress.
            else:
                # Reset elastic deformation gradient to avoid numerical instability.
                # NOTE: this makes FE purely dilational, clearing its deviatoric component
                # NOTE: that this update to FE is different than the update in the solid phase
                self.FE_p[p] = ti.Matrix.identity(float, self.n_dimensions) * JE_p ** (1 / self.n_dimensions)
                # self.F_p[p] = ti.sqrt(JE_p) * ti.Matrix.identity(float, self.n_dimensions)
                # Set mu to zero for water TODO: this could just be done in mu_0_p?
                mu = 0

            # Compute Piola-Kirchhoff stress P(F), (JST16, Eqn. 52)
            stress = 2 * mu * (self.FE_p[p] - U @ V.transpose()) @ self.FE_p[p].transpose()  # pyright: ignore

            # FIXME: discretize the dilational component with the pressure correction?!
            # NOTE: not incorporating this dilational stress is equivalent to setting
            #       stress to zero in the fluid phase.
            # stress += ti.Matrix.identity(float, 2) * la * JE_p * (JE_p - 1)

            # Compute D^(-1), which equals constant scaling for quadratic/cubic kernels.
            D_inv = 3 * self.inv_dx * self.inv_dx  # Cubic interpolation

            # TODO: What happens here exactly? Something with Cauchy-stress?
            stress *= -self.dt * self.particle_vol * D_inv

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            # TODO: use cx, cy vectors here directly?
            affine = stress + self.mass_p[p] * self.C_p[p]
            affine_x = affine @ ti.Vector([1, 0])  # pyright: ignore
            affine_y = affine @ ti.Vector([0, 1])  # pyright: ignore

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
                self.inv_lambda_c[base_c + offset] += weight_c * (1.0 / self.lambda_0_p[p])
                # TODO: this should be la, because of incorporated hardening?
                # TODO: store the inverse on particles to save computations:
                # self.inv_lambda_c[base_c + offset] += weight_c * self.inv_lambda_0_p[p]

                # We use JE^n, JP^n from the last timestep for the transfers, the updated
                # values will be assigned to the corresponding field at the end of the loop.
                self.JE_c[base_c + offset] += weight_c * self.mass_p[p] * JE_p
                self.JP_c[base_c + offset] += weight_c * self.mass_p[p] * JP_p
                # self.J_c[base_c + offset] += weight_c * self.mass_p[p] * J_p

            for i, j in ti.static(ti.ndrange(4, 4)):
                offset = ti.Vector([i, j])
                weight_x = w_x[i][0] * w_x[j][1]
                weight_y = w_y[i][0] * w_y[j][1]
                dpos_x = (ti.cast(offset, ti.f32) - dist_x) * self.dx
                dpos_y = (ti.cast(offset, ti.f32) - dist_y) * self.dx

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
            if (mass := self.mass_x[i, j]) > 0:
                self.velocity_x[i, j] /= mass
                collision_right = i >= (self.n_grid - self.boundary_width) and self.velocity_x[i, j] > 0
                collision_left = i <= self.boundary_width and self.velocity_x[i, j] < 0
                if collision_left or collision_right:
                    self.velocity_x[i, j] = 0

        for i, j in self.velocity_y:
            if (mass := self.mass_y[i, j]) > 0:
                self.velocity_y[i, j] /= mass
                self.velocity_y[i, j] += GRAVITY * self.dt
                collision_top = j >= (self.n_grid - self.boundary_width) and self.velocity_y[i, j] > 0
                collision_bottom = j <= self.boundary_width and self.velocity_y[i, j] < 0
                if collision_top or collision_bottom:
                    self.velocity_y[i, j] = 0

        for i, j in self.mass_c:
            if self.mass_c[i, j] > 0:  # No need for epsilon here
                self.temperature_c[i, j] *= 1 / self.mass_c[i, j]
                self.inv_lambda_c[i, j] *= 1 / self.mass_c[i, j]
                self.capacity_c[i, j] *= 1 / self.mass_c[i, j]
                self.JE_c[i, j] *= 1 / self.mass_c[i, j]
                # self.J_c[i, j] *= 1 / self.mass_c[i, j]
                self.JP_c[i, j] *= 1 / self.mass_c[i, j]

    @ti.kernel
    def _classify_cells(self):
        # FIXME: this is not used at the moment and replaced by
        # FIXME: the cell classification is offset to the left, resulting in asymmetry
        for i, j in self.classification_x:
            # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.

            # The simulation boundary is always colliding.
            x_face_is_colliding = i >= (self.n_grid - self.boundary_width) or i <= self.boundary_width
            x_face_is_colliding |= j >= (self.n_grid - self.boundary_width) or j <= self.boundary_width
            if x_face_is_colliding:
                self.classification_x[i, j] = Classification.Colliding
                continue

            # For convenience later on: a face is marked interior if it has mass.
            if self.mass_x[i, j] > 0:
                self.classification_x[i, j] = Classification.Interior
                continue

            # All remaining faces are empty.
            self.classification_x[i, j] = Classification.Empty

        for i, j in self.classification_y:
            # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.

            # The simulation boundary is always colliding.
            y_face_is_colliding = i >= (self.n_grid - self.boundary_width) or i <= self.boundary_width
            y_face_is_colliding |= j >= (self.n_grid - self.boundary_width) or j <= self.boundary_width
            if y_face_is_colliding:
                self.classification_y[i, j] = Classification.Colliding
                continue

            # For convenience later on: a face is marked interior if it has mass.
            if self.mass_y[i, j] > 0:
                self.classification_y[i, j] = Classification.Interior
                continue

            # All remaining faces are empty.
            self.classification_y[i, j] = Classification.Empty

        for i, j in self.classification_c:
            # TODO: Colliding cells are either assigned the temperature of the object it collides with or a user-defined
            # spatially-varying value depending on the setup. If the free surface is being enforced as a Dirichlet
            # temperature condition, the ambient air temperature is recorded for empty cells. No other cells
            # require temperatures to be recorded at this stage.

            # A cell is marked as colliding if all of its surrounding faces are colliding.
            cell_is_colliding = self.classification_x[i, j] == Classification.Colliding
            cell_is_colliding &= self.classification_x[i + 1, j] == Classification.Colliding
            cell_is_colliding &= self.classification_y[i, j] == Classification.Colliding
            cell_is_colliding &= self.classification_y[i, j + 1] == Classification.Colliding
            if cell_is_colliding:
                # self.cell_temperature[i, j] = self.ambient_temperature[None]
                self.classification_c[i, j] = Classification.Colliding
                continue

            # A cell is interior if the cell and all of its surrounding faces have mass.
            cell_is_interior = self.mass_c[i, j] > 0
            cell_is_interior &= self.mass_x[i, j] > 0
            cell_is_interior &= self.mass_x[i + 1, j] > 0
            cell_is_interior &= self.mass_y[i, j] > 0
            cell_is_interior &= self.mass_y[i, j + 1] > 0
            if cell_is_interior:
                self.classification_c[i, j] = Classification.Interior
                continue

            # All remaining cells are empty.
            self.classification_c[i, j] = Classification.Empty

            # The ambient air temperature is recorded for empty cells.
            self.temperature_c[i, j] = self.ambient_temperature[None]

    @ti.kernel
    def classify_cells(self):
        for i, j in self.classification_c:
            if not self.is_colliding(i, j):
                self.classification_c[i, j] = Classification.Empty

        for p in self.velocity_p:
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Find the nearest cell and set it to interior:
            i, j = ti.cast(self.position_p[p] * self.inv_dx, int)  # pyright: ignore
            if not self.is_colliding(i, j):  # pyright: ignore
                self.classification_c[i, j] = Classification.Interior

    @ti.kernel
    def compute_volumes(self):
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

            g_x = [dist_x - 1.5, (-2) * (dist_x - 1), dist_x - 0.5]
            g_y = [dist_y - 1.5, (-2) * (dist_y - 1), dist_y - 0.5]

            next_velocity = ti.Vector.zero(float, 2)
            # b_x = ti.Vector.zero(float, 2)
            # b_y = ti.Vector.zero(float, 2)
            c_x = ti.Vector.zero(ti.f32, 2)
            c_y = ti.Vector.zero(ti.f32, 2)
            next_temperature = 0.0
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                c_weight = w_c[i][0] * w_c[j][1]
                x_weight = w_x[i][0] * w_x[j][1]
                y_weight = w_y[i][0] * w_y[j][1]
                # x_dpos = ti.cast(offset, ti.f32) - dist_x
                # y_dpos = ti.cast(offset, ti.f32) - dist_y
                grad_x = ti.Vector([g_x[i][0] * w_x[j][1], w_x[i][0] * g_x[j][1]])
                grad_y = ti.Vector([g_y[i][0] * w_y[j][1], w_y[i][0] * g_y[j][1]])

                next_temperature += c_weight * self.temperature_c[base_c + offset]
                x_velocity = x_weight * self.velocity_x[base_x + offset]
                y_velocity = y_weight * self.velocity_y[base_y + offset]
                next_velocity += [x_velocity, y_velocity]
                # b_x += x_velocity * x_dpos
                # b_y += y_velocity * y_dpos
                c_x += self.velocity_x[base_x + offset] * grad_x
                c_y += self.velocity_y[base_y + offset] * grad_y

            # c_x = 3 * self.inv_dx * b_x  # C = B @ (D^(-1)), 1 / dx cancelled out by dx in dpos, Cubic kernels in P2G
            # c_y = 3 * self.inv_dx * b_y  # C = B @ (D^(-1)), 1 / dx cancelled out by dx in dpos, Cubic kernels in P2G
            self.C_p[p] = ti.Matrix([[c_x[0], c_y[0]], [c_x[1], c_y[1]]])  # pyright: ignore
            self.position_p[p] += self.dt * next_velocity
            self.velocity_p[p] = next_velocity

            # DONE: set temperature for empty cells
            # DONE: set temperature for particles, ideally per geometry
            # DONE: set heat capacity per particle depending on phase
            # DONE: set heat conductivity per particle depending on phase
            # DONE: set particle mass per phase
            # DONE: set E and nu for each particle depending on phase
            # DONE: apply latent heat
            # TODO: move this to a ti.func? (or keep this here but assign values in func and use when adding particles)
            # TODO: set theta_c, theta_s per phase? Water probably wants very small values, ice depends on temperature
            # TODO: in theory all of the constitutive parameters must be functions of temperature
            #       in the ice phase to range from solid ice to slushy ice?

            # # Initially, we allow each particle to freely change its temperature according to the heat equation.
            # # But whenever the freezing point is reached, any additional temperature change is multiplied by
            # # conductivity and mass and added to the buffer, with the particle temperature kept unchanged.
            # if (self.phase_p[p] == Phase.Ice) and (next_temperature >= 0):
            #     # Ice reached the melting point, additional temperature change is added to heat buffer.
            #     difference = next_temperature - self.temperature_p[p]
            #     self.heat_p[p] += self.conductivity_p[p] * self.mass_p[p] * difference
            #
            #     # If the heat buffer is full the particle changes its phase to water,
            #     # everything is then reset according to the new phase.
            #     if self.heat_p[p] >= LatentHeat.Water:
            #         # TODO: Shouldn't this just be set to ~inf, 0?
            #         E, nu = YoungsModulus.Water, PoissonsRatio.Water
            #         self.lambda_0_p[p] = E * nu / ((1 + nu) * (1 - 2 * nu))
            #         self.mu_0_p[p] = E / (2 * (1 + nu))
            #         self.capacity_p[p] = Capacity.Water
            #         self.conductivity_p[p] = Conductivity.Water
            #         self.color_p[p] = Color.Water
            #         self.temperature_p[p] = 0.0
            #         self.phase_p[p] = Phase.Water
            #         self.mass_p[p] = self.particle_vol * Density.Water
            #         self.heat_p[p] = LatentHeat.Water
            #
            # elif (self.phase_p[p] == Phase.Water) and (next_temperature < 0):
            #     # Water particle reached the freezing point, additional temperature change is added to heat buffer.
            #     difference = next_temperature - self.temperature_p[p]
            #     self.heat_p[p] += self.conductivity_p[p] * self.mass_p[p] * difference
            #
            #     # If the heat buffer is empty the particle changes its phase to ice,
            #     # everything is then reset according to the new phase.
            #     if self.heat_p[p] <= LatentHeat.Ice:
            #         E, nu = YoungsModulus.Ice, PoissonsRatio.Ice
            #         self.lambda_0_p[p] = E * nu / ((1 + nu) * (1 - 2 * nu))
            #         self.mu_0_p[p] = E / (2 * (1 + nu))
            #         self.capacity_p[p] = Capacity.Ice
            #         self.color_p[p] = Color.Ice
            #         self.conductivity_p[p] = Conductivity.Ice
            #         self.temperature_p[p] = 0.0
            #         self.phase_p[p] = Phase.Ice
            #         self.mass_p[p] = self.particle_vol * Density.Ice
            #         self.heat_p[p] = LatentHeat.Ice
            #
            # else:
            #     # Freely change temperature according to heat equation.
            #     self.temperature_p[p] = next_temperature
