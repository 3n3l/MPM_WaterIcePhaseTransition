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
        self.dt = 1e-4 / quality  # FIXME: dt for solid, fluid behaves wrong
        # self.dt = 1e-3 / quality  # FIXME: dt for fluid, solid explodes
        # self.dt = 3e-4 / quality # FIXME: this should be working for both phases, but isn't
        self.vol_0_p = (self.dx * 0.5) ** 2
        self.n_dimensions = 2

        # The width of the simulation boundary in grid nodes and offsets to
        # guarantee that seeded particles always lie within the boundary:
        self.boundary_width = 3
        self.lower = self.boundary_width * self.dx
        self.upper = 1 - self.lower

        # Properties on MAC-faces.
        # self.classification_x = ti.field(dtype=ti.i32, shape=(self.n_grid + 1, self.n_grid))
        # self.classification_y = ti.field(dtype=ti.i32, shape=(self.n_grid, self.n_grid + 1))
        # self.conductivity_x = ti.field(dtype=ti.f32, shape=(self.n_grid + 1, self.n_grid))
        # self.conductivity_y = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid + 1))
        # self.velocity_x = ti.field(dtype=ti.f32, shape=(self.n_grid + 1, self.n_grid))
        # self.velocity_y = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid + 1))
        # self.volume_x = ti.field(dtype=ti.f32, shape=(self.n_grid + 1, self.n_grid))
        # self.volume_y = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid + 1))
        # self.mass_x = ti.field(dtype=ti.f32, shape=(self.n_grid + 1, self.n_grid))
        # self.mass_y = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid + 1))

        # Properties on MAC-cells.
        self.classification_c = ti.field(dtype=ti.i32, shape=(self.n_grid, self.n_grid))
        self.velocity_c = ti.Vector.field(2, dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        # self.velocity_c = ti.Vector.field(2, dtype=ti.f32)
        self.temperature_c = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.inv_lambda_c = ti.field(dtype=ti.f64, shape=(self.n_grid, self.n_grid))
        self.volume_c = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.capacity_c = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.mass_c = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        # self.mass_c = ti.field(dtype=ti.f32)
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
        self.lambda_0 = ti.field(dtype=ti.f32, shape=())
        self.theta_c = ti.field(dtype=ti.f32, shape=())
        self.theta_s = ti.field(dtype=ti.f32, shape=())
        self.mu_0 = ti.field(dtype=ti.f32, shape=())
        self.zeta = ti.field(dtype=ti.i32, shape=())
        self.nu = ti.field(dtype=ti.f32, shape=())
        self.E = ti.field(dtype=ti.f32, shape=())

        # Poisson solvers for pressure and heat.
        self.pressure_solver = PressureSolver(self)
        # self.heat_solver = HeatSolver(self)

        ############################################################################################
        ### NEW VARIABLES
        ############################################################################################
        self.newton_max_iterations = 10
        self.newton_tolerance = 1e-6
        self.line_search = True  # FIXME: this just freezes under collision with line search and higher tols?
        self.line_search_max_iterations = 10
        self.linear_solve_tolerance_scale = 1
        self.linear_solve_max_iterations = 500
        self.linear_solve_tolerance = 1e-6
        self.cfl = 0.4
        self.ppc = 16
        self.debug_mode = False
        self.ignore_collision = False

        self.neighbour = (3,) * self.n_dimensions
        self.bound = 3
        self.n_nodes = self.n_grid**self.n_dimensions

        self.ignore_collision = False

        self.prevFE_p = ti.Matrix.field(self.n_dimensions, self.n_dimensions, dtype=ti.f32, shape=max_particles)  # for backup/restore F

        # These should be updated everytime a new SVD is performed to F
        # if ti.static(dim == 2):
        self.psi0 = ti.field(dtype=ti.f32, shape=max_particles)  # d_PsiHat_d_sigma0
        self.psi1 = ti.field(dtype=ti.f32, shape=max_particles)  # d_PsiHat_d_sigma1
        self.psi00 = ti.field(dtype=ti.f32, shape=max_particles)  # d^2_PsiHat_d_sigma0_d_sigma0
        self.psi01 = ti.field(dtype=ti.f32, shape=max_particles)  # d^2_PsiHat_d_sigma0_d_sigma1
        self.psi11 = ti.field(dtype=ti.f32, shape=max_particles)  # d^2_PsiHat_d_sigma1_d_sigma1
        self.m01 = ti.field(dtype=ti.f32, shape=max_particles)  # (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
        self.p01 = ti.field(dtype=ti.f32, shape=max_particles)  # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
        self.Aij = ti.Matrix.field(self.n_dimensions, self.n_dimensions, dtype=ti.f32, shape=max_particles)
        self.B01 = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)

        # if ti.static(dim == 3):
        self.psi0 = ti.field(dtype=ti.f32, shape=max_particles)  # d_PsiHat_d_sigma0
        self.psi1 = ti.field(dtype=ti.f32, shape=max_particles)  # d_PsiHat_d_sigma1
        self.psi2 = ti.field(dtype=ti.f32, shape=max_particles)  # d_PsiHat_d_sigma2
        self.psi00 = ti.field(dtype=ti.f32, shape=max_particles)  # d^2_PsiHat_d_sigma0_d_sigma0
        self.psi11 = ti.field(dtype=ti.f32, shape=max_particles)  # d^2_PsiHat_d_sigma1_d_sigma1
        self.psi22 = ti.field(dtype=ti.f32, shape=max_particles)  # d^2_PsiHat_d_sigma2_d_sigma2
        self.psi01 = ti.field(dtype=ti.f32, shape=max_particles)  # d^2_PsiHat_d_sigma0_d_sigma1
        self.psi02 = ti.field(dtype=ti.f32, shape=max_particles)  # d^2_PsiHat_d_sigma0_d_sigma2
        self.psi12 = ti.field(dtype=ti.f32, shape=max_particles)  # d^2_PsiHat_d_sigma1_d_sigma2

        self.m01 = ti.field(dtype=ti.f32, shape=max_particles)
        self.p01 = ti.field(dtype=ti.f32, shape=max_particles)  # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
        self.m02 = ti.field(dtype=ti.f32, shape=max_particles)  # (psi0-psi2)/(sigma0-sigma2), usually can be computed robustly
        self.p02 = ti.field(dtype=ti.f32, shape=max_particles)  # (psi0+psi2)/(sigma0+sigma2), need to clamp bottom with 1e-6
        self.m12 = ti.field(dtype=ti.f32, shape=max_particles)  # (psi1-psi2)/(sigma1-sigma2), usually can be computed robustly
        self.p12 = ti.field(dtype=ti.f32, shape=max_particles)  # (psi1+psi2)/(sigma1+sigma2), need to clamp bottom with 1e-6
        self.Aij = ti.Matrix.field(self.n_dimensions, self.n_dimensions, dtype=ti.f32, shape=max_particles)
        self.B01 = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)
        self.B12 = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)
        self.B20 = ti.Matrix.field(2, 2, dtype=ti.f32, shape=max_particles)

        # block_size = 16
        # indices = ti.ijk if self.n_dimensions == 3 else ti.ij
        # self.grid = ti.root.pointer(indices, [self.n_grid // block_size])
        # self.grid.dense(indices, block_size).place(self.velocity_c, self.mass_c)

        # data of Newton's method
        self.mass_matrix = ti.field(dtype=ti.f32)
        self.dv = ti.Vector.field(self.n_dimensions, dtype=ti.f32)  # dv = v(n+1) - v(n), Newton is formed from g(dv)=0
        self.residual = ti.Vector.field(self.n_dimensions, dtype=ti.f32)
        if ti.static(self.line_search):
            self.dv0 = ti.Vector.field(self.n_dimensions, dtype=ti.f32)  # dv of last iteration, for line search only

        # data of Linear Solver, i.e. Conjugate Gradient
        # All notations adopted from Wikipedia, q denotes A*p in general
        self.r = ti.Vector.field(self.n_dimensions, dtype=ti.f32)
        self.p = ti.Vector.field(self.n_dimensions, dtype=ti.f32)
        self.q = ti.Vector.field(self.n_dimensions, dtype=ti.f32)
        self.temp = ti.Vector.field(self.n_dimensions, dtype=ti.f32)
        self.step_direction = ti.Vector.field(self.n_dimensions, dtype=ti.f32)

        # scratch data for calculate differential of F
        self.scratch_xp = ti.Vector.field(self.n_dimensions, dtype=ti.f32, shape=max_particles)
        self.scratch_vp = ti.Vector.field(self.n_dimensions, dtype=ti.f32, shape=max_particles)
        self.scratch_gradV = ti.Matrix.field(self.n_dimensions, self.n_dimensions, dtype=ti.f32, shape=max_particles)
        self.scratch_stress = ti.Matrix.field(self.n_dimensions, self.n_dimensions, dtype=ti.f32, shape=max_particles)

        chip_size = 16
        self.newton_data = ti.root.pointer(ti.i, [self.n_nodes // chip_size])
        self.newton_data.dense(ti.i, chip_size).place(self.mass_matrix, self.dv, self.residual)
        if ti.static(self.line_search):
            self.newton_data.dense(ti.i, chip_size).place(self.dv0)

        self.linear_solver_data = ti.root.pointer(ti.i, [self.n_nodes // chip_size])
        self.linear_solver_data.dense(ti.i, chip_size).place(self.r, self.p, self.q, self.temp, self.step_direction)
        ############################################################################################
        ### /NEW VARIABLES
        ############################################################################################

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
        for i, j in self.classification_c:
            self.temperature_c[i, j] = 0
            self.inv_lambda_c[i, j] = 0
            self.capacity_c[i, j] = 0
            self.velocity_c[i, j] = 0
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

            # NOTE: this is now at the end of the timestep
            # Deformation gradient update. TODO: R might not be needed for our small timesteps
            # # self.FE_p[p] = self.R(self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore
            # self.FE_p[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore

            # TODO: could this just be: (or would this be unstable?)
            # self.FE_p[p] += (self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore

            # Remove the deviatoric component from each fluid particle:
            if self.phase_p[p] == Phase.Water:
                self.FE_p[p] = ti.sqrt(self.JE_p[p]) * ti.Matrix.identity(ti.f32, 2)
                self.updateIsotropicHelper(p, self.FE_p[p])
                # TODO: update helpers???

            # # Clamp singular values to simulate plasticity and elasticity:
            U, sigma, V = ti.svd(self.FE_p[p])
            # self.JE_p[p], self.JP_p[p] = 1.0, 1.0
            self.JE_p[p] = 1.0
            for d in ti.static(range(self.n_dimensions)):
                singular_value = float(sigma[d, d])
                clamped = float(sigma[d, d])
                if self.phase_p[p] == Phase.Ice:
                    # Clamp singular values to [1 - theta_c, 1 + theta_s]
                    clamped = max(clamped, 1 - self.theta_c[None])
                    clamped = min(clamped, 1 + self.theta_s[None])
                self.JP_p[p] *= singular_value / clamped
                self.JE_p[p] *= clamped
                sigma[d, d] = clamped
                # J *= singular_value

            # WARNING: if elasticity/plasticity is applied in the fluid phase, we also need this corrections:
            # if self.phase_p[p] == Phase.Water:
            #     self.FE_p[p] *= ti.sqrt(self.JP_p[p]) * (U @ sigma @ V.transpose())
            #     self.JE_p[p] = ti.math.determinant(self.FE_p[p])
            #     self.JP_p[p] = 1.0

            # TODO: explicit stress update wants small timestep, implicit pressure solve want bigger timestep

            # la, mu = 1.0, 1.0
            # FIXME: this is just for testing purposes, to change values from the gui, remove this later
            #        (or make gui sliders for water and ice values separate)
            # if self.phase_p[p] == Phase.Ice:
            #     la, mu = self.lambda_0[None], self.mu_0[None]
            # else:
            # la, mu = self.lambda_0[None], self.mu_0[None]
            la, mu = self.lambda_0_p[p], self.mu_0_p[p]
            cauchy_stress = ti.Matrix.zero(ti.f32, 2, 2)
            if self.phase_p[p] == Phase.Ice:
                # Reconstruct elastic deformation gradient after plasticity
                self.FE_p[p] = U @ sigma @ V.transpose()
                self.updateIsotropicHelper(p, self.FE_p[p])
                # TODO: update helpers???

                # Apply ice hardening by adjusting Lame parameters: TODO: uncomment this after testing
                hardening = ti.max(0.1, ti.min(20, ti.exp(self.zeta[None] * (1.0 - self.JP_p[p]))))
                la, mu = la * hardening, mu * hardening

                # Compute D^(-1), which equals constant scaling for quadratic/cubic kernels.
                D_inv = 3 * self.inv_dx * self.inv_dx  # Cubic interpolation

                # Compute Piola-Kirchhoff stress P(F), (JST16, Eqn. 52)
                # NOTE: this is the stress update to be used with the pressure correction
                F_dev = (self.JE_p[p] ** (-1 / 2)) * self.FE_p[p]
                # TODO: could just be this for d == 2?:
                # F_dev = self.FE_p[p] / ti.sqrt(self.JE_p[p])
                U_dev, _, V_dev = ti.svd(F_dev)  # TODO: can we just correct U?
                piola_kirchhoff = 2 * mu * (F_dev - U_dev @ V_dev.transpose())
                piola_kirchhoff = piola_kirchhoff @ self.FE_p[p].transpose()  # pyright: ignore
                # piola_kirchhoff += ti.Matrix.identity(ti.f32, 2) * la * self.JE_p[p] * (self.JE_p[p] - 1)
                # Cauchy stress times dt and D_inv
                cauchy_stress = -self.dt * self.vol_0_p * D_inv * piola_kirchhoff
                # TODO: the 1 / J is probably cancelled out by V^n and leaves us V^0 (self.vol_0_p)?!
                # cauchy_stress = (1 / J) * (piola_kirchhoff @ self.FE_p[p].transpose())  # pyright: ignore

                # # Compute Piola-Kirchhoff stress P(F), (JST16, Eqn. 52)
                # # NOTE: this is the usual MPM stress update
                # piola_kirchhoff = 2 * mu * (self.FE_p[p] - U @ V.transpose())
                # piola_kirchhoff = piola_kirchhoff @ self.FE_p[p].transpose()  # pyright: ignore
                # piola_kirchhoff += ti.Matrix.identity(ti.f32, 2) * la * self.JE_p[p] * (self.JE_p[p] - 1)
                # # Cauchy stress times dt and D_inv
                # # cauchy_stress = piola_kirchhoff @ self.FE_p[p].transpose()
                # cauchy_stress = -self.dt * self.vol_0_p * D_inv * piola_kirchhoff

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            # TODO: use cx, cy vectors here directly?
            affine = cauchy_stress + self.mass_p[p] * self.C_p[p]

            # Lower left corner of the interpolation grid:
            base_c = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([0.5, 0.5])), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32)

            # Cubic kernels (JST16 Eqn. 122 with x=fx, abs(fx-1), abs(fx-2), (and abs(fx-3) for faces).
            # Based on https://www.bilibili.com/opus/662560355423092789
            # FIXME: the weights might be wrong?
            w_c = [
                ((-0.166 * dist_c**3) + (dist_c**2) - (2 * dist_c) + 1.33),
                ((0.5 * ti.abs(dist_c - 1.0) ** 3) - ((dist_c - 1.0) ** 2) + 0.66),
                ((0.5 * ti.abs(dist_c - 2.0) ** 3) - ((dist_c - 2.0) ** 2) + 0.66),
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

                dpos_c = ti.cast(offset - dist_c, ti.f32) * self.dx

                # Rasterize velocity to grid faces.
                velocity_c = self.mass_p[p] * self.velocity_p[p] + affine @ dpos_c  # pyright: ignore
                self.velocity_c[base_c + offset] += weight_c * velocity_c

                # Rasterize conductivity to grid faces.
                # conductivity = self.mass_p[p] * self.conductivity_p[p]
                # self.conductivity_x[base_x + offset] += weight_x * conductivity
                # self.conductivity_y[base_y + offset] += weight_y * conductivity

        # Momentum to Velocity:
        for i, j in self.mass_c:
            if self.mass_c[i, j] > 0:
                self.velocity_c[i, j] /= self.mass_c[i, j]

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j in self.velocity_c:
            if (mass := self.mass_c[i, j]) > 0:
                # Normalize:
                self.temperature_c[i, j] /= mass
                self.inv_lambda_c[i, j] /= mass
                self.capacity_c[i, j] /= mass
                self.velocity_c[i, j] /= mass
                self.JE_c[i, j] /= mass
                self.JP_c[i, j] /= mass

                # Apply gravity:
                self.velocity_c[i, j] += [0, GRAVITY * self.dt]

                # Slip boundary condition:
                collision_right = i >= (self.n_grid - self.boundary_width) and self.velocity_c[i, j][0] > 0
                collision_left = i <= self.boundary_width and self.velocity_c[i, j][0] < 0
                if collision_left or collision_right:
                    self.velocity_c[i, j][0] = 0
                collision_top = j >= (self.n_grid - self.boundary_width) and self.velocity_c[i, j][1] > 0
                collision_bottom = j <= self.boundary_width and self.velocity_c[i, j][1] < 0
                if collision_top or collision_bottom:
                    self.velocity_c[i, j][1] = 0

    @ti.kernel
    def classify_cells(self):
        for i, j in self.classification_c:
            # Reset all the cells that don't belong to the colliding boundary:
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

    # # NOTE: this classification doesn't work on a collocated grid
    # @ti.kernel
    # def _classify_cells(self):
    #     for i, j in self.classification_c:
    #         # TODO: Colliding cells are either assigned the temperature of the object it collides with
    #         # or a user-defined spatially-varying value depending on the setup.
    #
    #         # NOTE: currently this is only set in the beginning, as the colliding boundary is fixed:
    #         # TODO: decide if this should be done here for better integration of colliding objects
    #         if self.is_colliding(i, j):
    #             continue
    #
    #         # A cell is interior if the cell and all of its surrounding faces have mass.
    #         cell_is_interior = self.mass_c[i, j] > 0
    #         # cell_is_interior &= self.mass_x[i, j] > 0 and self.mass_x[i + 1, j] > 0
    #         # cell_is_interior &= self.mass_y[i, j] > 0 and self.mass_y[i, j + 1] > 0
    #         if cell_is_interior:
    #             self.classification_c[i, j] = Classification.Interior
    #             continue
    #
    #         # All remaining cells are empty.
    #         self.classification_c[i, j] = Classification.Empty
    #
    #         # If the free surface is being enforced as a Dirichlet temperature condition,
    #         # the ambient air temperature is recorded for empty cells.
    #         self.temperature_c[i, j] = self.ambient_temperature[None]

    # @ti.kernel
    # def compute_volumes(self):
    #     # TODO: this seems to be wrong, the paper has a sum over CDFs
    #     control_volume = 0.5 * self.dx * self.dx
    #     for i, j in self.classification_c:
    #         if self.classification_c[i, j] == Classification.Interior:
    #             self.volume_x[i + 1, j] += control_volume
    #             self.volume_y[i, j + 1] += control_volume
    #             self.volume_x[i, j] += control_volume
    #             self.volume_y[i, j] += control_volume

    @ti.kernel
    def grid_to_particle(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Lower left corner of the interpolation grid:
            base_c = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([0.5, 0.5])), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32)

            # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2)
            # Based on https://www.bilibili.com/opus/662560355423092789
            w_c = [0.5 * (1.5 - dist_c) ** 2, 0.75 - (dist_c - 1) ** 2, 0.5 * (dist_c - 0.5) ** 2]

            next_velocity = ti.Vector.zero(ti.f32, 2)
            B = ti.Matrix.zero(ti.f32, 2, 2)
            next_temperature = 0.0
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                weight_c = w_c[i][0] * w_c[j][1]
                dpos_c = ti.cast(offset, ti.f32) - dist_c
                next_velocity += weight_c * self.velocity_c[base_c + offset]
                next_temperature += weight_c * self.temperature_c[base_c + offset]
                B += weight_c * self.velocity_c[base_c + offset].outer_product(dpos_c)

            self.C_p[p] = 3 * self.inv_dx * B
            # self.position_p[p] += self.dt * next_velocity
            self.velocity_p[p] = next_velocity

            # TODO: why is this at the end of the timestep?
            self.FE_p[p] = (ti.Matrix.identity(ti.f32, self.n_dimensions) + self.dt * self.C_p[p]) @ self.FE_p[p]  # pyright: ignore
            self.updateIsotropicHelper(p, self.FE_p[p])

            # TODO: why is this down here?
            self.position_p[p] += self.dt * self.velocity_p[p]

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
            #
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
            #         self.lambda_0[None]_p[p] = Lambda.Water
            #         self.mu_0[None]_p[p] = Mu.Water
            #         self.capacity_p[p] = Capacity.Water
            #         self.conductivity_p[p] = Conductivity.Water
            #         self.color_p[p] = Color.Water
            #         self.temperature_p[p] = 0.0
            #         self.phase_p[p] = Phase.Water
            #         self.mass_p[p] = self.vol_0_p * Density.Water
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
            #         self.lambda_0[None]_p[p] = Lambda.Ice
            #         self.mu_0[None]_p[p] = Mu.Ice
            #         self.capacity_p[p] = Capacity.Ice
            #         self.color_p[p] = Color.Ice
            #         self.conductivity_p[p] = Conductivity.Ice
            #         self.temperature_p[p] = 0.0
            #         self.phase_p[p] = Phase.Ice
            #         self.mass_p[p] = self.vol_0_p * Density.Ice
            #         self.heat_p[p] = LatentHeat.Ice
            #
            # else:
            #     # Freely change temperature according to heat equation.
            #     self.temperature_p[p] = next_temperature

    #################################################################################################
    ### NEW METHODS
    #################################################################################################
    @ti.func
    def clamp_small_magnitude(self, x, eps):
        result = 0.0
        if x < -eps:
            result = x
        elif x < 0:
            result = -eps
        elif x < eps:
            result = eps
        else:
            result = x
        return result

    @ti.func
    def makePD(self, M: ti.template()):  # pyright: ignore
        _, sigma, _ = ti.svd(M)
        if sigma[0, 0] < 0:
            D = ti.Matrix.zero(ti.f32, self.n_dimensions, self.n_dimensions)
            for i in ti.static(range(self.n_dimensions)):
                if sigma[i, i] < 0:
                    D[i, i] = 0
                else:
                    D[i, i] = sigma[i, i]

            M = sigma @ D @ sigma.transpose()

    @ti.func
    def makePD2d(self, M: ti.template()):  # pyright: ignore
        a = M[0, 0]
        b = (M[0, 1] + M[1, 0]) / 2
        d = M[1, 1]

        b2 = b * b
        D = a * d - b2
        T_div_2 = (a + d) / 2
        sqrtTT4D = ti.sqrt(ti.abs(T_div_2 * T_div_2 - D))
        L2 = T_div_2 - sqrtTT4D
        if L2 < 0.0:
            L1 = T_div_2 + sqrtTT4D
            if L1 <= 0.0:
                M = ti.zero(M)
            else:
                if b2 == 0:
                    M = ti.Matrix([[L1, 0], [0, 0]])
                else:
                    L1md = L1 - d
                    L1md_div_L1 = L1md / L1
                    M = ti.Matrix([[L1md_div_L1 * L1md, b * L1md_div_L1], [b * L1md_div_L1, b2 / L1]])

    # [i, j]/[i, j, k] -> id
    @ti.func
    def idx(self, I):
        ret = 0
        for i in range(self.n_dimensions):
            ret += I[i] * self.n_grid**i
        return ret
        # return sum([I[i] * self.n_grid ** i for i in range(self.dim)])
        # return sum([I[i] * self.n_grid ** i for i in range(self.dim)])

    # id -> [i, j]/[i, j, k]
    @ti.func
    def node(self, p):
        return ti.Vector([(p % (self.n_grid ** (i + 1))) // (self.n_grid**i) for i in range(self.n_dimensions)])

    # target = source
    @ti.kernel
    def copy(self, target: ti.template(), source: ti.template()):  # pyright: ignore
        for I in ti.grouped(source):
            target[I] = source[I]

    # target = source + scale * scaled
    @ti.kernel
    def scaledCopy(self, target: ti.template(), source: ti.template(), scale: ti.f32, scaled: ti.template()):  # pyright: ignore
        for I in ti.grouped(source):
            target[I] = source[I] + scale * scaled[I]

    # TODO: abstract as general stress classes
    @ti.func
    def psi(self, F):  # strain energy density function Ψ(F)
        U, _, V = ti.svd(F)

        # fixed corotated model, you can replace it with any constitutive model
        return self.mu_0[None] * (F - U @ V.transpose()).norm() ** 2 + self.lambda_0[None] / 2 * (F.determinant() - 1) ** 2

    @ti.func
    def dpsi_dF(self, F):  # first Piola-Kirchoff stress P(F), i.e. ∂Ψ/∂F
        U, _, V = ti.svd(F)
        J = F.determinant()
        R = U @ V.transpose()
        return 2 * self.mu_0[None] * (F - R) + self.lambda_0[None] * (J - 1) * J * F.inverse().transpose()

    # B = dPdF(Sigma) : A
    @ti.func
    def dPdFOfSigmaContractProjected(self, p, A, B: ti.template()):  # pyright: ignore
        if ti.static(self.n_dimensions == 2):
            B[0, 0] = self.Aij[p][0, 0] * A[0, 0] + self.Aij[p][0, 1] * A[1, 1]
            B[1, 1] = self.Aij[p][1, 0] * A[0, 0] + self.Aij[p][1, 1] * A[1, 1]
            B[0, 1] = self.B01[p][0, 0] * A[0, 1] + self.B01[p][0, 1] * A[1, 0]
            B[1, 0] = self.B01[p][1, 0] * A[0, 1] + self.B01[p][1, 1] * A[1, 0]
        if ti.static(self.n_dimensions == 3):
            B[0, 0] = self.Aij[p][0, 0] * A[0, 0] + self.Aij[p][0, 1] * A[1, 1] + self.Aij[p][0, 2] * A[2, 2]
            B[1, 1] = self.Aij[p][1, 0] * A[0, 0] + self.Aij[p][1, 1] * A[1, 1] + self.Aij[p][1, 2] * A[2, 2]
            B[2, 2] = self.Aij[p][2, 0] * A[0, 0] + self.Aij[p][2, 1] * A[1, 1] + self.Aij[p][2, 2] * A[2, 2]
            B[0, 1] = self.B01[p][0, 0] * A[0, 1] + self.B01[p][0, 1] * A[1, 0]
            B[1, 0] = self.B01[p][1, 0] * A[0, 1] + self.B01[p][1, 1] * A[1, 0]
            B[0, 2] = self.B20[p][0, 0] * A[0, 2] + self.B20[p][0, 1] * A[2, 0]
            B[2, 0] = self.B20[p][1, 0] * A[0, 2] + self.B20[p][1, 1] * A[2, 0]
            B[1, 2] = self.B12[p][0, 0] * A[1, 2] + self.B12[p][0, 1] * A[2, 1]
            B[2, 1] = self.B12[p][1, 0] * A[1, 2] + self.B12[p][1, 1] * A[2, 1]

    @ti.func
    def firstPiolaDifferential(self, p, F, dF):
        U, _, V = ti.svd(F)
        D = U.transpose() @ dF @ V
        K = ti.Matrix.zero(ti.f32, self.n_dimensions, self.n_dimensions)
        self.dPdFOfSigmaContractProjected(p, D, K)
        return U @ K @ V.transpose()

    @ti.func
    def reinitializeIsotropicHelper(self, p):
        if ti.static(self.n_dimensions == 2):
            self.psi0[p] = 0  # d_PsiHat_d_sigma0
            self.psi1[p] = 0  # d_PsiHat_d_sigma1
            self.psi00[p] = 0  # d^2_PsiHat_d_sigma0_d_sigma0
            self.psi01[p] = 0  # d^2_PsiHat_d_sigma0_d_sigma1
            self.psi11[p] = 0  # d^2_PsiHat_d_sigma1_d_sigma1
            self.m01[p] = 0  # (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
            self.p01[p] = 0  # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
            self.Aij[p] = ti.zero(self.Aij[p])
            self.B01[p] = ti.zero(self.B01[p])
        if ti.static(self.n_dimensions == 3):
            self.psi0[p] = 0  # d_PsiHat_d_sigma0
            self.psi1[p] = 0  # d_PsiHat_d_sigma1
            self.psi2[p] = 0  # d_PsiHat_d_sigma2
            self.psi00[p] = 0  # d^2_PsiHat_d_sigma0_d_sigma0
            self.psi11[p] = 0  # d^2_PsiHat_d_sigma1_d_sigma1
            self.psi22[p] = 0  # d^2_PsiHat_d_sigma2_d_sigma2
            self.psi01[p] = 0  # d^2_PsiHat_d_sigma0_d_sigma1
            self.psi02[p] = 0  # d^2_PsiHat_d_sigma0_d_sigma2
            self.psi12[p] = 0  # d^2_PsiHat_d_sigma1_d_sigma2

            self.m01[p] = 0  # (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
            self.p01[p] = 0  # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
            self.m02[p] = 0  # (psi0-psi2)/(sigma0-sigma2), usually can be computed robustly
            self.p02[p] = 0  # (psi0+psi2)/(sigma0+sigma2), need to clamp bottom with 1e-6
            self.m12[p] = 0  # (psi1-psi2)/(sigma1-sigma2), usually can be computed robustly
            self.p12[p] = 0  # (psi1+psi2)/(sigma1+sigma2), need to clamp bottom with 1e-6
            self.Aij[p] = ti.zero(self.Aij[p])
            self.B01[p] = ti.zero(self.B01[p])
            self.B12[p] = ti.zero(self.B12[p])
            self.B20[p] = ti.zero(self.B20[p])

    @ti.func
    def updateIsotropicHelper(self, p, F):
        self.reinitializeIsotropicHelper(p)
        if ti.static(self.n_dimensions == 2):
            _, sigma, _ = ti.svd(F)
            J = sigma[0, 0] * sigma[1, 1]
            _2mu = self.mu_0[None] * 2
            _lambda = self.lambda_0[None] * (J - 1)
            Sprod = ti.Vector([sigma[1, 1], sigma[0, 0]])
            self.psi0[p] = _2mu * (sigma[0, 0] - 1) + _lambda * Sprod[0]
            self.psi1[p] = _2mu * (sigma[1, 1] - 1) + _lambda * Sprod[1]
            self.psi00[p] = _2mu + self.lambda_0[None] * Sprod[0] * Sprod[0]
            self.psi11[p] = _2mu + self.lambda_0[None] * Sprod[1] * Sprod[1]
            self.psi01[p] = _lambda + self.lambda_0[None] * Sprod[0] * Sprod[1]

            # (psi0-psi1)/(sigma0-sigma1)
            self.m01[p] = _2mu - _lambda

            # (psi0+psi1)/(sigma0+sigma1)
            self.p01[p] = (self.psi0[p] + self.psi1[p]) / self.clamp_small_magnitude(sigma[0, 0] + sigma[1, 1], 1e-6)

            self.Aij[p] = ti.Matrix([[self.psi00[p], self.psi01[p]], [self.psi01[p], self.psi11[p]]])
            self.B01[p] = ti.Matrix(
                [
                    [(self.m01[p] + self.p01[p]) * 0.5, (self.m01[p] - self.p01[p]) * 0.5],
                    [(self.m01[p] - self.p01[p]) * 0.5, (self.m01[p] + self.p01[p]) * 0.5],
                ]
            )

            # proj A
            self.makePD(self.Aij[p])
            # proj B
            self.makePD2d(self.B01[p])

    def reinitialize(self):
        # FIXME: deactivate this
        # self.grid.deactivate_all()
        self.newton_data.deactivate_all()

    @ti.kernel
    def buildMassMatrix(self):
        for I in ti.grouped(self.mass_c):
            mass = self.mass_c[I]
            if mass > 0:
                self.mass_matrix[self.idx(I)] = mass

    @ti.kernel
    def buildInitialDvForNewton(self):
        for I in ti.grouped(self.mass_c):
            if self.mass_c[I] > 0:
                node_id = self.idx(I)
                if ti.static(not self.ignore_collision):
                    cond = (I < self.bound and self.velocity_c[I] < 0) or (
                        I > self.n_grid - self.bound and self.velocity_c[I] > 0
                    )
                    self.dv[node_id] = ti.select(cond, 0, GRAVITY * self.dt)
                else:
                    self.dv[node_id] = GRAVITY * self.dt  # Newton initial guess for non-collided nodes

    @ti.kernel
    def backupStrain(self):
        for p in self.FE_p:
            self.prevFE_p[p] = self.FE_p[p]

    @ti.kernel
    def restoreStrain(self):
        for p in self.FE_p:
            self.FE_p[p] = self.prevFE_p[p]

    @ti.kernel
    def constructNewVelocityFromNewtonResult(self):
        for I in ti.grouped(self.mass_c):
            if self.mass_c[I] > 0:
                self.velocity_c[I] += self.dv[self.idx(I)]
                cond = (I < self.bound and self.velocity_c[I] < 0) or (I > self.n_grid - self.bound and self.velocity_c[I] > 0)
                self.velocity_c[I] = ti.select(cond, 0, self.velocity_c[I])

    @ti.kernel
    def totalEnergy(self) -> ti.f32:  # pyright: ignore
        result = ti.cast(0.0, ti.f32)
        for p in self.FE_p:
            result += self.psi(self.FE_p[p]) * self.vol_0_p  # gathered from particles, psi defined in the rest space

        # inertia part
        for I in self.dv:
            m = self.mass_matrix[I]
            dv = self.dv[I]
            result += m * dv.dot(dv) / 2

        # gravity part
        for I in self.dv:
            m = self.mass_matrix[I]
            dv = self.dv[I]
            result -= self.dt * m * dv.dot(dv) / 2

        return result

    @ti.kernel
    def computeResidual(self):
        for I in self.dv:
            self.residual[I] = self.dt * self.mass_matrix[I] * [0, GRAVITY]

        for I in self.dv:
            self.residual[I] -= self.mass_matrix[I] * self.dv[I]

        # ti.loop_config(block_dim=self.n_grid)
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            Xp = self.position_p[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_C = ti.zero(self.C_p[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = ti.cast(1.0, ti.f32)
                for i in ti.static(range(self.n_dimensions)):
                    weight *= w[offset[i]][i]

                g_v = self.velocity_c[base + offset] + self.dv[self.idx(base + offset)]
                new_C += 3 * self.inv_dx * weight * g_v.outer_product(dpos)  # pyright: ignore

            F = (ti.Matrix.identity(ti.f32, self.n_dimensions) + self.dt * new_C) @ self.prevFE_p[p]
            # FIXME: this is D^-1 for quadratic kernels? Check this also elsewhere
            stress = (-self.vol_0_p * 3 * self.inv_dx * self.inv_dx) * self.dpsi_dF(F) @ F.transpose()

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = ti.cast(1.0, ti.f32)
                for i in ti.static(range(self.n_dimensions)):
                    weight *= w[offset[i]][i]

                force = weight * stress @ dpos
                self.residual[self.idx(base + offset)] += self.dt * force

        self.project(self.residual)

    @ti.kernel
    def computeNorm(self) -> ti.f32:  # pyright: ignore
        norm_sq = ti.cast(0.0, ti.f32)
        for I in self.dv:
            mass = self.mass_matrix[I]
            residual = self.residual[I]
            if mass > 0:
                norm_sq += residual.dot(residual) / mass
        return ti.sqrt(norm_sq)

    @ti.kernel
    def updateState(self):
        # ti.loop_config(block_dim=self.n_grid)
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            Xp = self.position_p[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_C = ti.zero(self.C_p[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = ti.cast(1.0, ti.f32)
                for i in ti.static(range(self.n_dimensions)):
                    weight *= w[offset[i]][i]

                g_v = self.velocity_c[base + offset] + self.dv[self.idx(base + offset)]
                new_C += 3 * self.inv_dx * weight * g_v.outer_product(dpos)

            self.FE_p[p] = (ti.Matrix.identity(ti.f32, self.n_dimensions) + self.dt * new_C) @ self.prevFE_p[p]
            self.updateIsotropicHelper(p, self.FE_p[p])
            self.scratch_xp[p] = self.position_p[p] + self.dt * self.scratch_vp[p]

    @ti.func
    def computeDvAndGradDv(self, dv: ti.template()):  # pyright: ignore
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            Xp = self.position_p[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            vp = ti.zero(self.scratch_vp[p])
            gradV = ti.zero(self.scratch_gradV[p])

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = ti.cast(1.0, ti.f32)
                for i in ti.static(range(self.n_dimensions)):
                    weight *= w[offset[i]][i]

                dv0 = dv[self.idx(base + offset)]
                vp += weight * dv0
                gradV += 3 * self.inv_dx * weight * dv0.outer_product(dpos)

            self.scratch_vp[p] = vp
            self.scratch_gradV[p] = gradV

    @ti.func
    def computeStressDifferential(self, p, gradDv: ti.template(), dstress: ti.template(), dvp: ti.template()):  # pyright: ignore
        Fn_local = self.prevFE_p[p]
        dP = self.firstPiolaDifferential(p, Fn_local, gradDv @ Fn_local)
        dstress += self.vol_0_p * dP @ Fn_local.transpose()  # pyright: ignore

    @ti.kernel
    def multiply(self, x: ti.template(), b: ti.template()):  # pyright: ignore
        for I in b:
            b[I] = ti.zero(b[I])

        # Note the relationship H dx = - df, where H is the stiffness matrix
        # inertia part
        for I in x:
            b[I] += self.mass_matrix[I] * x[I]

        self.computeDvAndGradDv(x)

        # scratch_gradV is now temporaraly used for storing gradDV (evaluated at particles)
        # scratch_vp is now temporaraly used for storing DV (evaluated at particles)

        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            self.scratch_stress[p] = ti.zero(self.scratch_stress[p])

        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            self.computeStressDifferential(p, self.scratch_gradV[p], self.scratch_stress[p], self.scratch_vp[p])
            # scratch_stress is now V_p^0 dP (F_p^n)^T (dP is Ap in snow paper)

        # ti.loop_config(block_dim=self.n_grid)
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            Xp = self.position_p[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]  # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            stress = self.scratch_stress[p]
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.n_dimensions)):
                    weight *= w[offset[i]][i]

                b[self.idx(base + offset)] += (
                    self.dt * self.dt * (weight * stress @ dpos)
                )  # fi -= \sum_p (Ap (xi-xp)  - fp )w_ip Dp_inv

    @ti.func
    def project(self, x: ti.template()):  # pyright: ignore
        for p in x:
            I = self.node(p)
            cond = any(I < self.bound and self.velocity_c[I] < 0) or any(
                I > self.n_grid - self.bound and self.velocity_c[I] > 0
            )
            if cond:
                x[p] = ti.zero(x[p])

    @ti.kernel
    def kernelProject(self, x: ti.template()):  # pyright: ignore
        self.project(x)

    @ti.kernel
    def precondition(self, _in: ti.template(), _out: ti.template()):  # pyright: ignore
        for I in _in:
            _out[I] = _in[I] / self.mass_matrix[I] if self.mass_matrix[I] > 0 else _in[I]

    @ti.kernel
    def dotProduct(self, a: ti.template(), b: ti.template()) -> ti.f32:  # pyright: ignore
        result = ti.cast(0.0, ti.f32)
        for I in a:
            result += a[I].dot(b[I])

        return result

    @ti.kernel
    def linearSolverReinitialize(self):
        for I in self.mass_matrix:
            self.r[I] = ti.zero(self.r[I])
            self.p[I] = ti.zero(self.p[I])
            self.q[I] = ti.zero(self.q[I])
            self.temp[I] = ti.zero(self.temp[I])
            self.step_direction[I] = ti.zero(self.step_direction[I])

    # solve Ax = b, where A build implicitly, x := step_direction, b := redisual
    def linearSolve(self, x, b, relative_tolerance):
        self.linear_solver_data.deactivate_all()
        self.linearSolverReinitialize()

        # NOTE: requires that the input x has been projected
        # self.multiply(x, self.temp)
        self.scaledCopy(self.r, b, -1, self.temp)
        self.kernelProject(self.r)
        self.precondition(self.r, self.q)  # NOTE: requires that preconditioning matrix is projected
        self.copy(self.p, self.q)

        zTrk = self.dotProduct(self.r, self.q)

        # print('\033[1;36mzTrk = ', zTrk, '\033[0m')
        residual_preconditioned_norm = ti.sqrt(zTrk)
        local_tolerance = ti.min(relative_tolerance * residual_preconditioned_norm, self.linear_solve_tolerance)
        for cnt in range(self.linear_solve_max_iterations):
            if ti.static(self.debug_mode):
                print(
                    "\033[1;33mlinear_iter = ",
                    cnt,
                    ", residual_preconditioned_norm = ",
                    residual_preconditioned_norm,
                    "\033[0m",
                )
            if residual_preconditioned_norm <= local_tolerance:
                return cnt

            self.multiply(self.p, self.temp)
            self.kernelProject(self.temp)
            alpha = zTrk / self.dotProduct(self.temp, self.p)
            if ti.static(self.debug_mode):
                print("\033[1;36malpha = ", alpha, "\033[0m")
            self.scaledCopy(x, x, alpha, self.p)  # i.e. x += p * alpha
            self.scaledCopy(self.r, self.r, -alpha, self.temp)  # i.e. r -= temp * alpha
            self.precondition(self.r, self.q)  # NOTE: requires that preconditioning matrix is projected

            zTrk_last = zTrk
            zTrk = self.dotProduct(self.q, self.r)
            if ti.static(self.debug_mode):
                print("\033[1;36mzTrk = ", zTrk, "\033[0m")
            beta = zTrk / zTrk_last
            if ti.static(self.debug_mode):
                print("\033[1;36mbeta = ", beta, "\033[0m")

            self.scaledCopy(self.p, self.q, beta, self.p)  # i.e. p = q + beta * p

            residual_preconditioned_norm = ti.sqrt(zTrk)

        return self.linear_solve_max_iterations

    def backwardEulerStep(self):  # on the assumption that collision is ignored
        self.buildMassMatrix()
        self.buildInitialDvForNewton()
        # Which should be called at the beginning of newton.
        self.backupStrain()

        self.newtonSolve()

        self.restoreStrain()
        self.constructNewVelocityFromNewtonResult()

    def newtonSolve(self):
        self.updateState()
        E0 = 0.0  # totalEnergy of last iteration, for line search only
        if ti.static(self.line_search):
            E0 = self.totalEnergy()
            self.copy(self.dv0, self.dv)

        for it in range(self.newton_max_iterations):
            # Mv^(n) - Mv^(n+1) + dt * f(x_n + dt v^(n+1)) + dt * Mg
            # -Mdv + dt * f(x_n + dt(v^n + dv)) + dt * Mg
            self.computeResidual()
            residual_norm = self.computeNorm()
            if ti.static(self.debug_mode):
                print("\033[1;31mnewton_iter = ", it, ", residual_norm = ", residual_norm, "\033[0m")
            if residual_norm < self.newton_tolerance:
                break

            linear_solve_relative_tolerance = ti.min(
                0.5, self.linear_solve_tolerance_scale * ti.sqrt(ti.max(residual_norm, self.newton_tolerance))
            )
            self.linearSolve(self.step_direction, self.residual, linear_solve_relative_tolerance)

            if ti.static(self.line_search):
                step_size, E = 1.0, 0.0
                for ls_cnt in range(self.line_search_max_iterations):
                    self.scaledCopy(self.dv, self.dv0, step_size, self.step_direction)
                    self.updateState()
                    E = self.totalEnergy()
                    if ti.static(self.debug_mode):
                        print("\033[1;32m[line search]", "E = ", E, "E0 = ", E0, "\033[0m")
                    step_size /= 2
                    if E <= E0:
                        break
                E0 = E
                self.copy(self.dv0, self.dv)
            else:
                self.scaledCopy(self.dv, self.dv, 1, self.step_direction)
                self.updateState()
