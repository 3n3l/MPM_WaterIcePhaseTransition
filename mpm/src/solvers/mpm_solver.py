from src.constants import Capacity, Classification, Color, Conductivity, Density, Phase, Density, LatentHeat, State
from src.solvers import PressureSolver, HeatSolver

import taichi as ti

# TODO: move to configuration?
GRAVITY = -9.81


@ti.data_oriented
class MPM_Solver:
    def __init__(self, quality: int, max_particles: int, should_use_direct_solver: bool = True):
        # MPM Parameters that are configuration independent
        self.n_particles = ti.field(dtype=ti.int32, shape=())
        self.max_particles = max_particles
        self.n_grid = 128 * quality
        self.n_cells = self.n_grid * self.n_grid
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = 1e-4 / quality
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

        # Properties on particles.
        self.conductivity_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.temperature_p = ti.field(dtype=ti.float32, shape=max_particles)
        # self.particle_lambda_0 = ti.field(dtype=ti.float32, shape=max_particles)
        # self.particle_mu_0 = ti.field(dtype=ti.float32, shape=max_particles)
        self.inv_lambda_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.position_p = ti.Vector.field(2, dtype=float, shape=max_particles)
        self.velocity_p = ti.Vector.field(2, dtype=float, shape=max_particles)
        self.capacity_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.color_p = ti.Vector.field(3, dtype=float, shape=max_particles)
        self.state_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.phase_p = ti.field(dtype=ti.float32, shape=max_particles)
        self.mass_p = ti.field(dtype=ti.float32, shape=max_particles)
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

    @ti.func
    def in_bounds(self, x: float, y: float) -> bool:
        return self.lower < x < self.upper and self.lower < y < self.upper

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

    @ti.kernel
    def particle_to_grid(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Deformation gradient update. TODO: is this F and not FE???
            self.FE_p[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore

            # TODO: set lambda an mu per particle
            la, mu = self.lambda_0[None], self.mu_0[None]
            U, sigma, V = ti.svd(self.FE_p[p])
            JE_p, JP_p = 1.0, 1.0

            # Clamp singular values to simulate plasticity and elasticity.
            for d in ti.static(range(self.n_dimensions)):
                singular_value = float(sigma[d, d])
                # Clamp singular values to [1 - theta_c, 1 + theta_s]
                clamped = max(singular_value, 1 - self.theta_c[None])
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
            else:
                # Reset elastic deformation gradient to avoid numerical instability.
                self.FE_p[p] = ti.Matrix.identity(float, self.n_dimensions) * JE_p ** (1 / self.n_dimensions)
                # FIXME: JE_p is only close to 1.0, but should be exactly 1.0?!
                # print(JE_p, JP_p)
                mu = 0

            # Compute Piola-Kirchhoff stress P(F), (JST16, Eqn. 52)
            stress = 2 * mu * (self.FE_p[p] - U @ V.transpose())
            stress = stress @ self.FE_p[p].transpose()  # pyright: ignore
            stress += ti.Matrix.identity(float, 2) * la * JE_p * (JE_p - 1)

            # Compute D^(-1), which equals constant scaling for quadratic/cubic kernels.
            # D_inv = 4 * self.inv_dx * self.inv_dx  # Quadratic interpolation
            D_inv = 3 * self.inv_dx * self.inv_dx  # Cubic interpolation

            # TODO: What happens here exactly? Something with Cauchy-stress?
            stress *= -self.dt * self.particle_vol * D_inv

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            affine = stress + self.mass_p[p] * self.C_p[p]
            affine_x = affine @ ti.Vector([1, 0])  # pyright: ignore
            affine_y = affine @ ti.Vector([0, 1])  # pyright: ignore

            # Additional stagger for the grid and additional 0.5 to force flooring, used for the weight computations.
            x_stagger = ti.Vector([1.0, 1.5])
            y_stagger = ti.Vector([1.5, 1.0])
            c_stagger = ti.Vector([0.5, 0.5])

            # Additional offsets for the grid, used for the distance (fx) computations.
            x_offset = ti.Vector([0.0, 0.5])
            y_offset = ti.Vector([0.5, 0.0])

            # We use an additional offset of 0.5 for element-wise flooring.
            base_c = ti.floor((self.position_p[p] * self.inv_dx - c_stagger), dtype=ti.i32)
            base_x = ti.floor((self.position_p[p] * self.inv_dx - x_stagger), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - y_stagger), dtype=ti.i32)
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32)
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - x_offset
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - y_offset

            # Cubic kernels (JST16 Eqn. 122 with x=fx, abs(fx-1), abs(fx-2), (and abs(fx-3) for faces).
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
                # TODO: use particle_inv_lambda, set different lambda for each phase?
                # self.inv_lambda_c[base_c + offset] += weight_c * self.inv_lambda_p[p]
                self.inv_lambda_c[base_c + offset] += weight_c * self.lambda_0[None]

                # We use JE^n, JP^n from the last timestep for the transfers, the updated
                # values will be assigned to the corresponding field at the end of P2G.
                self.JE_c[base_c + offset] += weight_c * self.mass_p[p] * self.JE_p[p]
                self.JP_c[base_c + offset] += weight_c * self.mass_p[p] * self.JP_p[p]

            for i, j in ti.static(ti.ndrange(4, 4)):
                offset = ti.Vector([i, j])
                weight_x = w_x[i][0] * w_x[j][1]
                weight_y = w_y[i][0] * w_y[j][1]
                dpos_x = (offset.cast(float) - dist_x) * self.dx
                dpos_y = (offset.cast(float) - dist_y) * self.dx

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

            # Update field values with JE^(n+1), JP^(n+1).
            self.JE_p[p] = JE_p
            self.JP_p[p] = JP_p

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.mass_x:
            if self.mass_x[i, j] > 0:  # No need for epsilon here
                self.velocity_x[i, j] *= 1 / self.mass_x[i, j]
                # TODO: as the boundary is classified as colliding later on, this could done while applying pressure?
                collision_left = i < self.boundary_width and self.velocity_x[i, j] < 0
                collision_right = i > (self.n_grid - self.boundary_width) and self.velocity_x[i, j] > 0
                if collision_left or collision_right:
                    self.velocity_x[i, j] = 0
        for i, j in self.mass_y:
            if self.mass_y[i, j] > 0:  # No need for epsilon here
                self.velocity_y[i, j] *= 1 / self.mass_y[i, j]
                self.velocity_y[i, j] += self.dt * GRAVITY
                # TODO: as the boundary is classified as colliding later on, this could done while applying pressure?
                collision_top = j > (self.n_grid - self.boundary_width) and self.velocity_y[i, j] > 0
                collision_bottom = j < self.boundary_width and self.velocity_y[i, j] < 0
                if collision_top or collision_bottom:
                    self.velocity_y[i, j] = 0
        for i, j in self.mass_c:
            if self.mass_c[i, j] > 0:  # No need for epsilon here
                self.temperature_c[i, j] *= 1 / self.mass_c[i, j]
                self.inv_lambda_c[i, j] *= 1 / self.mass_c[i, j]
                self.capacity_c[i, j] *= 1 / self.mass_c[i, j]
                self.JE_c[i, j] *= 1 / self.mass_c[i, j]
                self.JP_c[i, j] *= 1 / self.mass_c[i, j]

    @ti.kernel
    def classify_cells(self):
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
    def compute_volumes(self):
        for i, j in self.classification_c:
            if self.classification_c[i, j] == Classification.Interior:
                control_volume = 0.5 * self.dx * self.dx
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

            # Additional stagger for the grid and additional 0.5 to force flooring, used for the weight computations.
            c_stagger = ti.Vector([0.5, 0.5])
            x_stagger = ti.Vector([0.5, 1.0])
            y_stagger = ti.Vector([1.0, 0.5])

            # Additional offsets for the grid, used for the distance (fx) computations.
            x_offset = ti.Vector([0.0, 0.5])
            y_offset = ti.Vector([0.5, 0.0])

            base_c = ti.floor((self.position_p[p] * self.inv_dx - c_stagger), dtype=ti.i32)
            base_x = ti.floor((self.position_p[p] * self.inv_dx - x_stagger), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - y_stagger), dtype=ti.i32)
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32)
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - x_offset
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - y_offset
            # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2)
            w_c = [0.5 * (1.5 - dist_c) ** 2, 0.75 - (dist_c - 1) ** 2, 0.5 * (dist_c - 0.5) ** 2]
            w_x = [0.5 * (1.5 - dist_x) ** 2, 0.75 - (dist_x - 1) ** 2, 0.5 * (dist_x - 0.5) ** 2]
            w_y = [0.5 * (1.5 - dist_y) ** 2, 0.75 - (dist_y - 1) ** 2, 0.5 * (dist_y - 0.5) ** 2]

            bx = ti.Vector.zero(float, 2)
            by = ti.Vector.zero(float, 2)
            nv = ti.Vector.zero(float, 2)
            nt = 0.0
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])

                c_weight = w_c[i][0] * w_c[j][1]
                nt += c_weight * self.temperature_c[base_c + offset]

                x_weight = w_x[i][0] * w_x[j][1]
                y_weight = w_y[i][0] * w_y[j][1]
                x_dpos = ti.cast(offset, ti.f32) - dist_x
                y_dpos = ti.cast(offset, ti.f32) - dist_y
                x_velocity = x_weight * self.velocity_x[base_x + offset]
                y_velocity = y_weight * self.velocity_y[base_y + offset]
                nv += [x_velocity, y_velocity]
                bx += x_velocity * x_dpos
                by += y_velocity * y_dpos

            # cx = 4 * self.inv_dx * bx  # C = B @ (D^(-1)), 1 / dx cancelled out by dx in dpos, Quadratic kernels in P2G
            # cy = 4 * self.inv_dx * by  # C = B @ (D^(-1)), 1 / dx cancelled out by dx in dpos, Quadratic kernels in P2G
            cx = 3 * self.inv_dx * bx  # C = B @ (D^(-1)), 1 / dx cancelled out by dx in dpos, Cubic kernels in P2G
            cy = 3 * self.inv_dx * by  # C = B @ (D^(-1)), 1 / dx cancelled out by dx in dpos, Cubic kernels in P2G
            self.C_p[p] = ti.Matrix([[cx[0], cy[0]], [cx[1], cy[1]]])  # pyright: ignore
            self.position_p[p] += self.dt * nv
            self.velocity_p[p] = nv

            # DONE: set temperature for empty cells
            # DONE: set temperature for particles, ideally per geometry
            # DONE: set heat capacity per particle depending on phase
            # DONE: set heat conductivity per particle depending on phase
            # DONE: set particle mass per phase
            # TODO: set E and nu for each particle depending on phase
            # DONE: apply latent heat
            # TODO: move this to a ti.func?

            # Initially, we allow each particle to freely change its temperature according to the heat equation.
            # But whenever the freezing point is reached, any additional temperature change is multiplied by
            # conductivity and mass and added to the buffer, with the particle temperature kept unchanged.
            if (self.phase_p[p] == Phase.Ice) and (nt >= 0):
                # Ice reached the melting point, additional temperature change is added to heat buffer.
                self.heat_p[p] += self.conductivity_p[p] * self.mass_p[p] * (nt - self.temperature_p[p])

                # If the heat buffer is full the particle changes its phase to water,
                # everything is then reset according to the new phase.
                if self.heat_p[p] > LatentHeat.Water:
                    # TODO: Lame parameters for each phase, something like:
                    # E = 3 * 1e-4
                    # nu = 0.45
                    # self.particle_mu_0[p] = ...
                    # self.particle_lambda_0[p] = ...
                    self.capacity_p[p] = Capacity.Water
                    self.conductivity_p[p] = Conductivity.Water
                    self.color_p[p] = Color.Water
                    self.temperature_p[p] = 0.0
                    self.phase_p[p] = Phase.Water
                    self.mass_p[p] = self.particle_vol * Density.Water
                    self.heat_p[p] = LatentHeat.Water

            elif (self.phase_p[p] == Phase.Water) and (nt < 0):
                # Water particle reached the freezing point, additional temperature change is added to heat buffer.
                self.heat_p[p] += self.conductivity_p[p] * self.mass_p[p] * (nt - self.temperature_p[p])

                # If the heat buffer is empty the particle changes its phase to ice,
                # everything is then reset according to the new phase.
                if self.heat_p[p] < LatentHeat.Ice:
                    # TODO: Lame parameters for each phase, something like:
                    # E = 7 * 1e-4
                    # nu = 0.3
                    # self.particle_mu_0[p] = ...
                    # self.particle_lambda_0[p] = ...
                    self.capacity_p[p] = Capacity.Ice
                    self.color_p[p] = Color.Ice
                    self.conductivity_p[p] = Conductivity.Ice
                    self.temperature_p[p] = 0.0
                    self.phase_p[p] = Phase.Ice
                    self.mass_p[p] = self.particle_vol * Density.Ice
                    self.heat_p[p] = LatentHeat.Ice

            else:
                # Freely change temperature according to heat equation.
                self.temperature_p[p] = nt
