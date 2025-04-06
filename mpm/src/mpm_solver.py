from src.enums import Capacity, Classification, Color, Conductivity, Density, Phase, State, Density
from src.pressure_solver import PressureSolver
from src.heat_solver import HeatSolver

import taichi as ti

WATER_CONDUCTIVITY = 0.55  # Water: 0.55, Ice: 2.33
ICE_CONDUCTIVITY = 2.33
LATENT_HEAT = 334.4  # kJ/kg, L_p
GRAVITY = -9.81


@ti.data_oriented
class MPM_Solver:
    def __init__(self, quality: int, max_particles: int):
        # MPM Parameters that are configuration independent
        self.n_particles = ti.field(dtype=ti.int32, shape=())
        self.current_frame = ti.field(dtype=ti.int32, shape=())
        self.n_grid = 128 * quality
        self.n_cells = self.n_grid * self.n_grid
        self.dx = 1 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.dt = 1e-4 / quality
        self.rho_0 = 1000  # TODO: this is kg/m^3 for water, what about ice?
        self.particle_vol = (self.dx * 0.5) ** 2
        self.n_dimensions = 2

        # The width of the simulation boundary in grid nodes.
        self.boundary_width = 3

        # Offset to correct coordinates such that the origin lies within the boundary,
        # added to each position vector when loading a new configuration.
        self.boundary_offset = 1 - ((self.n_grid - self.boundary_width) * self.dx)

        # Properties on MAC-faces.
        self.face_classification_x = ti.field(dtype=int, shape=(self.n_grid + 1, self.n_grid))
        self.face_classification_y = ti.field(dtype=int, shape=(self.n_grid, self.n_grid + 1))
        self.face_conductivity_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_conductivity_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.face_velocity_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_velocity_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.face_volume_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_volume_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))
        self.face_mass_x = ti.field(dtype=ti.float32, shape=(self.n_grid + 1, self.n_grid))
        self.face_mass_y = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid + 1))

        # Properties on MAC-cells.
        self.cell_classification = ti.field(dtype=ti.int8, shape=(self.n_grid, self.n_grid))
        self.cell_temperature = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_inv_lambda = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_capacity = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_pressure = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_mass = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_JE = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))
        self.cell_JP = ti.field(dtype=ti.float32, shape=(self.n_grid, self.n_grid))

        # Properties on particles.
        self.p_conductivity = ti.field(dtype=ti.float32, shape=max_particles)
        self.p_temperature = ti.field(dtype=ti.float32, shape=max_particles)
        # self.particle_lambda_0 = ti.field(dtype=ti.float32, shape=max_particles)
        # self.particle_mu_0 = ti.field(dtype=ti.float32, shape=max_particles)
        self.particle_inv_lambda = ti.field(dtype=ti.float32, shape=max_particles)
        self.particle_position = ti.Vector.field(2, dtype=float, shape=max_particles)
        self.particle_velocity = ti.Vector.field(2, dtype=float, shape=max_particles)
        self.particle_capacity = ti.field(dtype=ti.float32, shape=max_particles)
        self.particle_color = ti.Vector.field(3, dtype=float, shape=max_particles)
        self.p_phase = ti.field(dtype=ti.float32, shape=max_particles)
        self.p_mass = ti.field(dtype=ti.float32, shape=max_particles)
        self.particle_FE = ti.Matrix.field(2, 2, dtype=float, shape=max_particles)
        self.particle_JE = ti.field(dtype=float, shape=max_particles)
        self.particle_JP = ti.field(dtype=float, shape=max_particles)
        self.particle_C = ti.Matrix.field(2, 2, dtype=float, shape=max_particles)

        # Fields needed to implement sources (TODO: and sinks), the state will be set to
        # active once the activation threshold (frame) is reached. Active particles in
        # p_active_position will be drawn, all other particles are hidden until active.
        self.p_activation_threshold = ti.field(dtype=int, shape=max_particles)
        self.p_activation_state = ti.field(dtype=int, shape=max_particles)
        self.p_active_position = ti.Vector.field(2, dtype=ti.float32, shape=max_particles)

        # Fields needed for the latent heat and phase change.
        self.p_heat = ti.field(dtype=ti.float32, shape=max_particles)  # U_p
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

        # The solver for the Poisson pressure equation.
        self.pressure_solver = PressureSolver(self)
        self.heat_solver = HeatSolver(self)

        # Additional stagger for the grid and additional 0.5 to force flooring, used for the weight computations.
        self.x_stagger = ti.Vector([(self.dx * 0.5) + 0.5, 0.5])
        self.y_stagger = ti.Vector([0.5, (self.dx * 0.5) + 0.5])
        self.c_stagger = ti.Vector([0.5, 0.5])

        # Additional offsets for the grid, used for the distance (fx) computations.
        self.x_offset = ti.Vector([(self.dx * 0.5), 0.0])
        self.y_offset = ti.Vector([0.0, (self.dx * 0.5)])

    @ti.kernel
    def reset_grids(self):
        for i, j in self.face_velocity_x:
            self.face_conductivity_x[i, j] = 0
            self.face_velocity_x[i, j] = 0
            self.face_volume_x[i, j] = 0
            self.face_mass_x[i, j] = 0

        for i, j in self.face_velocity_y:
            self.face_conductivity_y[i, j] = 0
            self.face_velocity_y[i, j] = 0
            self.face_volume_y[i, j] = 0
            self.face_mass_y[i, j] = 0

        for i, j in self.cell_classification:
            self.cell_temperature[i, j] = 0
            self.cell_inv_lambda[i, j] = 0
            self.cell_pressure[i, j] = 0
            self.cell_capacity[i, j] = 0
            self.cell_mass[i, j] = 0
            self.cell_JE[i, j] = 0
            self.cell_JP[i, j] = 0

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
                if self.p_phase[p] == Phase.Ice:
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
            if self.p_phase[p] == Phase.Water:
                # TODO: Apply correction for dilational/deviatoric stresses?
                # Reset elastic deformation gradient to avoid numerical instability.
                self.particle_FE[p] = ti.Matrix.identity(float, self.n_dimensions) * JE ** (1 / self.n_dimensions)
                # Set the viscosity to zero.
                mu = 0
            elif self.p_phase[p] == Phase.Ice:
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

            # TODO: What happens here exactly? Something with Cauchy-stress?
            stress *= -self.dt * self.particle_vol * D_inv

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            affine = stress + self.p_mass[p] * self.particle_C[p]
            x_affine = affine @ ti.Vector([1, 0])  # pyright: ignore
            y_affine = affine @ ti.Vector([0, 1])  # pyright: ignore

            # We use an additional offset of 0.5 for element-wise flooring.
            c_base = ti.floor((self.particle_position[p] * self.inv_dx - self.c_stagger), dtype=ti.i32)
            x_base = ti.floor((self.particle_position[p] * self.inv_dx - self.x_stagger), dtype=ti.i32)
            y_base = ti.floor((self.particle_position[p] * self.inv_dx - self.y_stagger), dtype=ti.i32)
            c_fx = self.particle_position[p] * self.inv_dx - ti.cast(c_base, ti.f32)
            x_fx = self.particle_position[p] * self.inv_dx - ti.cast(x_base, ti.f32) - self.x_offset
            y_fx = self.particle_position[p] * self.inv_dx - ti.cast(y_base, ti.f32) - self.y_offset
            # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2)
            c_w = [0.5 * (1.5 - c_fx) ** 2, 0.75 - (c_fx - 1) ** 2, 0.5 * (c_fx - 0.5) ** 2]
            x_w = [0.5 * (1.5 - x_fx) ** 2, 0.75 - (x_fx - 1) ** 2, 0.5 * (x_fx - 0.5) ** 2]
            y_w = [0.5 * (1.5 - y_fx) ** 2, 0.75 - (y_fx - 1) ** 2, 0.5 * (y_fx - 0.5) ** 2]

            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                c_weight = c_w[i][0] * c_w[j][1]
                x_weight = x_w[i][0] * x_w[j][1]
                y_weight = y_w[i][0] * y_w[j][1]
                x_dpos = (offset.cast(float) - x_fx) * self.dx
                y_dpos = (offset.cast(float) - y_fx) * self.dx

                # Rasterize mass to grid faces.
                self.face_mass_x[x_base + offset] += x_weight * self.p_mass[p]
                self.face_mass_y[y_base + offset] += y_weight * self.p_mass[p]

                # Rasterize velocity to grid faces.
                x_velocity = self.p_mass[p] * self.particle_velocity[p][0] + x_affine @ x_dpos
                y_velocity = self.p_mass[p] * self.particle_velocity[p][1] + y_affine @ y_dpos
                self.face_velocity_x[x_base + offset] += x_weight * x_velocity
                self.face_velocity_y[y_base + offset] += y_weight * y_velocity

                # Rasterize conductivity to grid faces.
                conductivity = self.p_mass[p] * self.p_conductivity[p]
                self.face_conductivity_x[x_base + offset] += x_weight * conductivity
                self.face_conductivity_y[y_base + offset] += y_weight * conductivity

                # Rasterize to cell centers.
                temperature = self.p_mass[p] * self.p_temperature[p]
                self.cell_temperature[c_base + offset] += c_weight * temperature
                capacity = self.p_mass[p] * self.particle_capacity[p]
                self.cell_capacity[c_base + offset] += c_weight * capacity
                self.cell_mass[c_base + offset] += c_weight * self.p_mass[p]

                # t = self.cell_temperature[c_base + offset]
                # if not (t < 0 or t > 0 or t == 0):
                #     print("AAAAAAAAAAAAAAAAAAAAH")
                #     print("INDICES ->", c_base + offset)
                #     print("TEMP -> ", self.particle_temperature[p])
                #     self.cell_temperature[c_base ** 100] = 0

                # TODO: use particle_inv_lambda, set different lambda for each phase?
                self.cell_inv_lambda[c_base + offset] += c_weight * self.particle_inv_lambda[p]

                # NOTE: the old JE, JP values are used here to compute the cell values.
                self.cell_JE[c_base + offset] += c_weight * self.p_mass[p] * self.particle_JE[p]
                self.cell_JP[c_base + offset] += c_weight * self.p_mass[p] * self.particle_JP[p]
                # FIXME: or do we need to use the new ones?
                # self.cell_JE[c_base + offset] += c_weight * JE
                # self.cell_JP[c_base + offset] += c_weight * JP
            self.particle_JE[p] = JE
            self.particle_JP[p] = JP

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.face_mass_x:
            if self.face_mass_x[i, j] > 0:  # No need for epsilon here
                self.face_velocity_x[i, j] *= 1 / self.face_mass_x[i, j]
                # TODO: as the boundary is classified as colliding later on, this could done while applying pressure?
                collision_left = i < self.boundary_width and self.face_velocity_x[i, j] < 0
                collision_right = i > (self.n_grid - self.boundary_width) and self.face_velocity_x[i, j] > 0
                if collision_left or collision_right:
                    self.face_velocity_x[i, j] = 0
        for i, j in self.face_mass_y:
            if self.face_mass_y[i, j] > 0:  # No need for epsilon here
                self.face_velocity_y[i, j] *= 1 / self.face_mass_y[i, j]
                self.face_velocity_y[i, j] += self.dt * GRAVITY
                # TODO: as the boundary is classified as colliding later on, this could done while applying pressure?
                collision_top = j > (self.n_grid - self.boundary_width) and self.face_velocity_y[i, j] > 0
                collision_bottom = j < self.boundary_width and self.face_velocity_y[i, j] < 0
                if collision_top or collision_bottom:
                    self.face_velocity_y[i, j] = 0
        for i, j in self.cell_mass:
            if self.cell_mass[i, j] > 0:  # No need for epsilon here
                self.cell_temperature[i, j] *= 1 / self.cell_mass[i, j]
                self.cell_inv_lambda[i, j] *= 1 / self.cell_mass[i, j]
                self.cell_capacity[i, j] *= 1 / self.cell_mass[i, j]
                self.cell_JE[i, j] *= 1 / self.cell_mass[i, j]
                self.cell_JP[i, j] *= 1 / self.cell_mass[i, j]

    @ti.kernel
    def classify_cells(self):
        for i, j in self.face_classification_x:
            # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.

            # The simulation boundary is always colliding.
            x_face_is_colliding = i > (self.n_grid - self.boundary_width) or i < self.boundary_width
            if x_face_is_colliding:
                self.face_classification_x[i, j] = Classification.Colliding
                continue

            # For convenience later on: a face is marked interior if it has mass.
            if self.face_mass_x[i, j] > 0:
                self.face_classification_x[i, j] = Classification.Interior
                continue

            # All remaining faces are empty.
            self.face_classification_x[i, j] = Classification.Empty

        for i, j in self.face_classification_y:
            # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.

            # The simulation boundary is always colliding.
            y_face_is_colliding = j > (self.n_grid - self.boundary_width) or j < self.boundary_width
            if y_face_is_colliding:
                self.face_classification_y[i, j] = Classification.Colliding
                continue

            # For convenience later on: a face is marked interior if it has mass.
            if self.face_mass_y[i, j] > 0:
                self.face_classification_y[i, j] = Classification.Interior
                continue

            # All remaining faces are empty.
            self.face_classification_y[i, j] = Classification.Empty

        for i, j in self.cell_classification:
            # TODO: Colliding cells are either assigned the temperature of the object it collides with or a user-defined
            # spatially-varying value depending on the setup. If the free surface is being enforced as a Dirichlet
            # temperature condition, the ambient air temperature is recorded for empty cells. No other cells
            # require temperatures to be recorded at this stage.

            # A cell is marked as colliding if all of its surrounding faces are colliding.
            cell_is_colliding = self.face_classification_x[i, j] == Classification.Colliding
            cell_is_colliding &= self.face_classification_x[i + 1, j] == Classification.Colliding
            cell_is_colliding &= self.face_classification_y[i, j] == Classification.Colliding
            cell_is_colliding &= self.face_classification_y[i, j + 1] == Classification.Colliding
            if cell_is_colliding:
                # self.cell_temperature[i, j] = self.ambient_temperature[None]
                self.cell_classification[i, j] = Classification.Colliding
                continue

            # A cell is interior if the cell and all of its surrounding faces have mass.
            cell_is_interior = self.cell_mass[i, j] > 0
            cell_is_interior &= self.face_mass_x[i, j] > 0
            cell_is_interior &= self.face_mass_x[i + 1, j] > 0
            cell_is_interior &= self.face_mass_y[i, j] > 0
            cell_is_interior &= self.face_mass_y[i, j + 1] > 0
            if cell_is_interior:
                self.cell_classification[i, j] = Classification.Interior
                continue

            # All remaining cells are empty.
            self.cell_classification[i, j] = Classification.Empty

            # The ambient air temperature is recorded for empty cells.
            self.cell_temperature[i, j] = self.ambient_temperature[None]

    @ti.kernel
    def compute_volumes(self):
        for i, j in self.cell_classification:
            if self.cell_classification[i, j] == Classification.Interior:
                control_volume = 0.5 * self.dx * self.dx
                self.face_volume_x[i + 1, j] += control_volume
                self.face_volume_y[i, j + 1] += control_volume
                self.face_volume_x[i, j] += control_volume
                self.face_volume_y[i, j] += control_volume

    @ti.kernel
    def grid_to_particle(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We only update active particles.
            if self.p_activation_state[p] == State.Inactive:
                continue

            c_base = ti.floor((self.particle_position[p] * self.inv_dx - self.c_stagger), dtype=ti.i32)
            x_base = ti.floor((self.particle_position[p] * self.inv_dx - self.x_stagger), dtype=ti.i32)
            y_base = ti.floor((self.particle_position[p] * self.inv_dx - self.y_stagger), dtype=ti.i32)
            c_fx = self.particle_position[p] * self.inv_dx - ti.cast(c_base, ti.f32)
            x_fx = self.particle_position[p] * self.inv_dx - ti.cast(x_base, ti.f32) - self.x_offset
            y_fx = self.particle_position[p] * self.inv_dx - ti.cast(y_base, ti.f32) - self.y_offset
            # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2)
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
                x_dpos = ti.cast(offset, ti.f32) - x_fx
                y_dpos = ti.cast(offset, ti.f32) - y_fx
                x_velocity = x_weight * self.face_velocity_x[x_base + offset]
                y_velocity = y_weight * self.face_velocity_y[y_base + offset]
                nv += [x_velocity, y_velocity]
                bx += x_velocity * x_dpos
                by += y_velocity * y_dpos

                # FIXME: is this not incorporating values from the left and bottom cells?
                nt += c_weight * self.cell_temperature[c_base + offset]
                # FIXME: This will incorporate values from left and bottom cell,
                # but not from top and right cells
                # nt += c_weight * self.cell_temperature[c_base + offset - 1]

            cx = 4 * self.inv_dx * bx  # C = B @ (D^(-1)), 1 / dx cancelled out by dx in dpos, Quadratic kernels
            cy = 4 * self.inv_dx * by  # C = B @ (D^(-1)), 1 / dx cancelled out by dx in dpos, Quadratic kernels
            self.particle_C[p] = ti.Matrix([[cx[0], cy[0]], [cx[1], cy[1]]])  # pyright: ignore
            self.particle_position[p] += self.dt * nv
            self.particle_velocity[p] = nv

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
            if (self.p_phase[p] == Phase.Ice) and (nt >= 0):
                # Ice reached the melting point, additional temperature change is added to heat buffer.
                self.p_heat[p] += self.p_conductivity[p] * self.p_mass[p] * (nt - self.p_temperature[p])

                # If the heat buffer is full the particle changes its phase to water,
                # everything is then reset according to the new phase.
                if self.p_heat[p] > LATENT_HEAT:
                    # TODO: Lame parameters for each phase, something like:
                    # E = 3 * 1e-4
                    # nu = 0.45
                    # self.particle_mu_0[p] = ...
                    # self.particle_lambda_0[p] = ...
                    self.particle_capacity[p] = Capacity.Water
                    self.p_conductivity[p] = Conductivity.Water
                    self.particle_color[p] = Color.Water
                    self.p_temperature[p] = 0.0
                    self.p_phase[p] = Phase.Water
                    self.p_mass[p] = self.particle_vol * Density.Water
                    self.p_heat[p] = LATENT_HEAT

            elif (self.p_phase[p] == Phase.Water) and (nt < 0):
                # Water particle reached the freezing point, additional temperature change is added to heat buffer.
                self.p_heat[p] += self.p_conductivity[p] * self.p_mass[p] * (nt - self.p_temperature[p])

                # If the heat buffer is empty the particle changes its phase to ice,
                # everything is then reset according to the new phase.
                if self.p_heat[p] < 0:
                    # TODO: Lame parameters for each phase, something like:
                    # E = 7 * 1e-4
                    # nu = 0.3
                    # self.particle_mu_0[p] = ...
                    # self.particle_lambda_0[p] = ...
                    self.particle_capacity[p] = Capacity.Ice
                    self.particle_color[p] = Color.Ice
                    self.p_conductivity[p] = Conductivity.Ice
                    self.p_temperature[p] = 0.0
                    self.p_phase[p] = Phase.Ice
                    self.p_mass[p] = self.particle_vol * Density.Ice
                    self.p_heat[p] = 0.0

            else:
                # Freely change temperature according to heat equation.
                self.p_temperature[p] = nt

            # The particle_position holds the positions for all particles, active and inactive,
            # only pushing the position into p_active_position will draw this particle.
            self.p_active_position[p] = self.particle_position[p]

    def substep(self) -> None:
        self.current_frame[None] += 1
        for _ in range(int(2e-3 // self.dt)):
            self.reset_grids()
            self.particle_to_grid()
            self.momentum_to_velocity()
            self.classify_cells()
            self.compute_volumes()
            # self.pressure_solver.solve()
            self.heat_solver.solve()
            self.grid_to_particle()
