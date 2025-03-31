from src.enums import Classification
import taichi as ti


@ti.data_oriented
class Pressure_MGPCGSolver:
    def __init__(self, mpm_solver):
        # TODO: only one variable needed here
        self.n_grid = mpm_solver.n_grid
        self.m = mpm_solver.n_grid
        self.n = mpm_solver.n_grid

        self.x_velocity = mpm_solver.face_velocity_x
        self.y_velocity = mpm_solver.face_velocity_y

        self.dx = mpm_solver.dx
        self.inv_dx = mpm_solver.inv_dx

        self.dt = mpm_solver.dt
        self.c_JP = mpm_solver.cell_JP
        self.c_JE = mpm_solver.cell_JE
        self.c_inv_lambda = mpm_solver.cell_inv_lambda
        self.c_classification = mpm_solver.cell_classification

        self.multigrid_level = 4
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 10

        # rhs of linear system
        self.b = ti.field(dtype=ti.f32, shape=(self.m, self.n))

        # TODO: make lambda
        def grid_shape(l):
            return (self.m // 2**l, self.n // 2**l)

        # lhs of linear system and its corresponding form in coarse grids
        self.Adiag = [ti.field(dtype=ti.f32, shape=grid_shape(l)) for l in range(self.multigrid_level)]
        self.Ax = [ti.field(dtype=ti.f32, shape=grid_shape(l)) for l in range(self.multigrid_level)]
        self.Ay = [ti.field(dtype=ti.f32, shape=grid_shape(l)) for l in range(self.multigrid_level)]

        # grid type
        self.grid_type = [ti.field(dtype=ti.i32, shape=grid_shape(l)) for l in range(self.multigrid_level)]

        # pcg var
        self.r = [ti.field(dtype=ti.f32, shape=grid_shape(l)) for l in range(self.multigrid_level)]
        self.z = [ti.field(dtype=ti.f32, shape=grid_shape(l)) for l in range(self.multigrid_level)]

        self.pressure = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.s = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.As = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.sum = ti.field(dtype=ti.f32, shape=())
        self.alpha = ti.field(dtype=ti.f32, shape=())
        self.beta = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def system_init_kernel(self) -> None:
        # define right hand side of linear system
        # assume that scale_b = 1 / grid_x
        for i, j in ti.ndrange(self.m, self.n):
            if self.c_classification[i, j] == Classification.Interior:
                self.b[i, j] = -((self.c_JE[i, j] - 1) / (self.dt * self.c_JE[i, j]))
                # if i != self.n_grid - 1:
                    # b[idx] -= 2 * self.inv_dx * (self.x_velocity[i + 1, j] - self.x_velocity[i, j])
                self.b[i, j] -= self.inv_dx * (self.x_velocity[i + 1, j] - self.x_velocity[i, j])
                # if j != self.n_grid - 1:
                    # b[idx] -= 2 * self.inv_dx * (self.y_velocity[i, j + 1] - self.y_velocity[i, j])
                self.b[i, j] -= self.inv_dx * (self.y_velocity[i, j + 1] - self.y_velocity[i, j])

        # modify right hand side of linear system to account for solid velocities
        # currently hard code solid velocities to zero
        # for i, j in ti.ndrange(self.m, self.n):
        #     if self.c_classification[i, j] == Classification.Interior:
        #         if self.c_classification[i - 1, j] == Classification.Colliding:
        #             self.b[i, j] -= scale_b * (self.x_velocity[i, j] - 0)
        #         if self.c_classification[i + 1, j] == Classification.Colliding:
        #             self.b[i, j] += scale_b * (self.x_velocity[i + 1, j] - 0)
        #
        #         if self.c_classification[i, j - 1] == Classification.Colliding:
        #             self.b[i, j] -= scale_b * (self.y_velocity[i, j] - 0)
        #         if self.c_classification[i, j + 1] == Classification.Colliding:
        #             self.b[i, j] += scale_b * (self.y_velocity[i, j + 1] - 0)

        # define left handside of linear system
        scale_A = self.dt * self.inv_dx * self.inv_dx
        for i, j in ti.ndrange(self.m, self.n):
            self.Adiag[0][i, j] += (self.c_JP[i, j] / (self.c_JE[i, j] * self.dt)) * self.c_inv_lambda[i, j]
            if self.c_classification[i, j] == Classification.Interior:
                if self.c_classification[i - 1, j] == Classification.Interior:
                    self.Adiag[0][i, j] += scale_A
                if self.c_classification[i + 1, j] == Classification.Interior:
                    self.Adiag[0][i, j] += scale_A
                    self.Ax[0][i, j] = -scale_A
                elif self.c_classification[i + 1, j] == Classification.Empty:
                    self.Adiag[0][i, j] += scale_A

                if self.c_classification[i, j - 1] == Classification.Interior:
                    self.Adiag[0][i, j] += scale_A
                if self.c_classification[i, j + 1] == Classification.Interior:
                    self.Adiag[0][i, j] += scale_A
                    self.Ay[0][i, j] = -scale_A
                elif self.c_classification[i, j + 1] == Classification.Empty:
                    self.Adiag[0][i, j] += scale_A

    @ti.kernel
    def gridtype_init(self, l: ti.template()):  # pyright: ignore
        for i, j in self.grid_type[l]:
            # if i == 0 or i == self.m // (2**l) - 1 or j == 0 or j == self.n // (2 ** l) - 1:
            #     self.grid_type[l][i, j] = Classification.Colliding

            i2 = i * 2
            j2 = j * 2

            if (
                self.grid_type[l - 1][i2, j2] == Classification.Empty
                or self.grid_type[l - 1][i2, j2 + 1] == Classification.Empty
                or self.grid_type[l - 1][i2 + 1, j2] == Classification.Empty
                or self.grid_type[l - 1][i2 + 1, j2 + 1] == Classification.Empty
            ):
                self.grid_type[l][i, j] = Classification.Empty
            else:
                if (
                    self.grid_type[l - 1][i2, j2] == Classification.Interior
                    or self.grid_type[l - 1][i2, j2 + 1] == Classification.Interior
                    or self.grid_type[l - 1][i2 + 1, j2] == Classification.Interior
                    or self.grid_type[l - 1][i2 + 1, j2 + 1] == Classification.Interior
                ):
                    self.grid_type[l][i, j] = Classification.Interior
                else:
                    self.grid_type[l][i, j] = Classification.Colliding

    @ti.kernel
    def preconditioner_init(self, l: ti.template()):  # pyright: ignore
        scale = self.dt * self.inv_dx * self.inv_dx
        s = scale / (2**l * 2**l)

        for i, j in self.grid_type[l]:
            self.Adiag[l][i, j] += ((self.c_JP[i, j] / (self.c_JE[i, j] * self.dt)) * self.c_inv_lambda[i, j]) / (
                2**l * 2**l
            )
            if self.grid_type[l][i, j] == Classification.Interior:
                if self.grid_type[l][i - 1, j] == Classification.Interior:
                    self.Adiag[l][i, j] += s
                if self.grid_type[l][i + 1, j] == Classification.Interior:
                    self.Adiag[l][i, j] += s
                    self.Ax[l][i, j] = -s
                elif self.grid_type[l][i + 1, j] == Classification.Empty:
                    self.Adiag[l][i, j] += s

                if self.grid_type[l][i, j - 1] == Classification.Interior:
                    self.Adiag[l][i, j] += s
                if self.grid_type[l][i, j + 1] == Classification.Interior:
                    self.Adiag[l][i, j] += s
                    self.Ay[l][i, j] = -s
                elif self.grid_type[l][i, j + 1] == Classification.Empty:
                    self.Adiag[l][i, j] += s

    def system_init(self):
        self.b.fill(0.0)

        for l in range(self.multigrid_level):
            self.Adiag[l].fill(0.0)
            self.Ax[l].fill(0.0)
            self.Ay[l].fill(0.0)

        self.system_init_kernel()
        self.grid_type[0].copy_from(self.c_classification)

        for l in range(1, self.multigrid_level):
            self.gridtype_init(l)
            self.preconditioner_init(l)

    @ti.func
    def neighbor_sum(self, Ax, Ay, z, nx, ny, i, j):
        Az = (
            Ax[(i - 1 + nx) % nx, j] * z[(i - 1 + nx) % nx, j]
            + Ax[i, j] * z[(i + 1) % nx, j]
            + Ay[i, (j - 1 + ny) % ny] * z[i, (j - 1 + ny) % ny]
            + Ay[i, j] * z[i, (j + 1) % ny]
        )

        return Az

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.i32):  # pyright: ignore
        # phase: red/black Gauss-Seidel phase
        for i, j in self.r[l]:
            if self.grid_type[l][i, j] == Classification.Interior and (i + j) & 1 == phase:
                self.z[l][i, j] = (
                    self.r[l][i, j]
                    - self.neighbor_sum(self.Ax[l], self.Ay[l], self.z[l], self.m // (2**l), self.n // (2**l), i, j)
                ) / self.Adiag[l][i, j]

    @ti.kernel
    def restrict(self, l: ti.template()):  # pyright: ignore
        for i, j in self.r[l]:
            if self.grid_type[l][i, j] == Classification.Interior:
                Az = self.Adiag[l][i, j] * self.z[l][i, j]
                Az += self.neighbor_sum(self.Ax[l], self.Ay[l], self.z[l], self.m // (2**l), self.n // (2**l), i, j)
                res = self.r[l][i, j] - Az

                self.r[l + 1][i // 2, j // 2] += 0.25 * res

    @ti.kernel
    def prolongate(self, l: ti.template()):  # pyright: ignore
        for i, j in self.z[l]:
            self.z[l][i, j] += self.z[l + 1][i // 2, j // 2]

    def v_cycle(self):
        self.z[0].fill(0.0)
        for l in range(self.multigrid_level - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)

            self.r[l + 1].fill(0.0)
            self.z[l + 1].fill(0.0)
            self.restrict(l)

        # solve Az = r on the coarse grid
        for i in range(self.bottom_smoothing):
            self.smooth(self.multigrid_level - 1, 0)
            self.smooth(self.multigrid_level - 1, 1)

        for l in reversed(range(self.multigrid_level - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self, max_iters=500):
        tol = 1e-12

        self.pressure.fill(0.0)
        self.As.fill(0.0)
        self.s.fill(0.0)
        self.r[0].copy_from(self.b)

        self.reduce(self.r[0], self.r[0])
        init_rTr = self.sum[None]

        print("init rTr = {}".format(init_rTr))

        if init_rTr < tol:
            print("Converged: init rtr = {}".format(init_rTr))
        else:
            # p0 = 0
            # r0 = b - Ap0 = b
            # z0 = M^-1r0
            # self.z.fill(0.0)
            self.v_cycle()

            # s0 = z0
            self.s.copy_from(self.z[0])

            # zTr
            self.reduce(self.z[0], self.r[0])
            old_zTr = self.sum[None]

            iteration = 0

            for i in range(max_iters):
                # alpha = zTr / sAs
                self.compute_As()
                self.reduce(self.s, self.As)
                sAs = self.sum[None]
                self.alpha[None] = old_zTr / sAs

                # p = p + alpha * s
                self.update_p()

                # r = r - alpha * As
                self.update_r()

                # check for convergence
                self.reduce(self.r[0], self.r[0])
                rTr = self.sum[None]
                if rTr < init_rTr * tol:
                    break

                # z = M^-1r
                self.v_cycle()

                self.reduce(self.z[0], self.r[0])
                new_zTr = self.sum[None]

                # beta = zTrnew / zTrold
                self.beta[None] = new_zTr / old_zTr

                # s = z + beta * s
                self.update_s()
                old_zTr = new_zTr
                iteration = i

                # if iteration % 100 == 0:
                #     print("iter {}, res = {}".format(iteration, rTr))

            print("Converged to {} in {} iterations".format(rTr, iteration))
        print("Pressure result: ")
        print(self.pressure)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):  # pyright: ignore
        self.sum[None] = 0.0
        for i, j in ti.ndrange(self.m, self.n):
            if self.c_classification[i, j] == Classification.Interior:
                self.sum[None] += p[i, j] * q[i, j]

    @ti.kernel
    def compute_As(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.c_classification[i, j] == Classification.Interior:
                self.As[i, j] = (
                    self.Adiag[0][i, j] * self.s[i, j]
                    + self.Ax[0][i - 1, j] * self.s[i - 1, j]
                    + self.Ax[0][i, j] * self.s[i + 1, j]
                    + self.Ay[0][i, j - 1] * self.s[i, j - 1]
                    + self.Ay[0][i, j] * self.s[i, j + 1]
                )

    @ti.kernel
    def update_p(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.c_classification[i, j] == Classification.Interior:
                self.pressure[i, j] = self.pressure[i, j] + self.alpha[None] * self.s[i, j]

    @ti.kernel
    def update_r(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.c_classification[i, j] == Classification.Interior:
                self.r[0][i, j] = self.r[0][i, j] - self.alpha[None] * self.As[i, j]

    @ti.kernel
    def update_s(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.c_classification[i, j] == Classification.Interior:
                self.s[i, j] = self.z[0][i, j] + self.beta[None] * self.s[i, j]
