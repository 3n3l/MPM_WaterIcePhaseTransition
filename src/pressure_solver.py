from taichi.linalg import SparseMatrixBuilder, SparseSolver
from enums import Classification

import taichi as ti


@ti.data_oriented
class PressureSolver:
    def __init__(self, mpm_solver) -> None:
        self.n_cells = mpm_solver.n_grid * mpm_solver.n_grid
        self.n_grid = mpm_solver.n_grid
        self.inv_dx = mpm_solver.inv_dx
        self.rho_0 = mpm_solver.rho_0
        self.dt = mpm_solver.dt

        self.cell_classification = mpm_solver.cell_classification
        self.cell_inv_lambda = mpm_solver.cell_inv_lambda
        self.cell_pressure = mpm_solver.cell_pressure
        self.cell_JE = mpm_solver.cell_JE
        self.cell_JP = mpm_solver.cell_JP

        self.face_velocity_x = mpm_solver.face_velocity_x
        self.face_velocity_y = mpm_solver.face_velocity_y
        self.face_volume_x = mpm_solver.face_volume_x
        self.face_volume_y = mpm_solver.face_volume_y
        self.face_mass_x = mpm_solver.face_mass_x
        self.face_mass_y = mpm_solver.face_mass_y

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
                    B[row, row - self.n_grid] += -self.dt * (self.face_volume_x[i, j] / self.face_mass_x[i, j])
                    # B[row, row - 1] += -self.dt * (self.face_volume_x[i, j] / self.face_mass_x[i, j])
                    B[row, row] += 1.0
                    C[row, row - self.n_grid] += -1.0
                    C[row, row] += 1.0

                if (i != self.n_grid - 1) and (self.cell_classification[i + 1, j] == Classification.Interior):
                    B[row, row + self.n_grid] += -self.dt * (self.face_volume_x[i + 1, j] / self.face_mass_x[i + 1, j])
                    # B[row, row + 1] += -self.dt * (self.face_volume_x[i + 1, j] / self.face_mass_x[i + 1, j])
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
                B[row, row] += 1.0
                C[row, row] += 1.0

            elif self.cell_classification[i, j] == Classification.Colliding:  # Neumann
                # TODO: apply Neumann boundary condition
                # TODO: the colliding classification doesn't exist atm
                A[row, row] += 1.0
                B[row, row] += 1.0
                C[row, row] += 1.0

    @ti.kernel
    def fill_pressure_field(self, p: ti.types.ndarray()):  # pyright: ignore
        for i, j in self.cell_pressure:
            # TODO: move this to apply_pressure, delete self.cell_pressure (IF POSSIBLE)
            row = (i * self.n_grid) + j
            self.cell_pressure[i, j] = p[row]

    @ti.kernel
    def apply_pressure(self):
        # inv_rho = 1  # TODO: compute inv_rho from face volumes per cell
        inv_rho = 1 / self.rho_0  # TODO: compute rho per cell
        z = self.dt * inv_rho * self.inv_dx * self.inv_dx  # TODO: are these even needed?
        for i, j in ti.ndrange((1, self.n_grid - 1), (1, self.n_grid - 1)):
            if self.cell_classification[i, j] == Classification.Interior:
                if self.cell_classification[i - 1, j] == Classification.Interior:
                    self.face_velocity_x[i, j] -= z * self.cell_pressure[i - 1, j]
                if self.cell_classification[i + 1, j] == Classification.Interior:
                    self.face_velocity_x[i, j] -= z * self.cell_pressure[i + 1, j]
                if self.cell_classification[i, j - 1] == Classification.Interior:
                    self.face_velocity_y[i, j] -= z * self.cell_pressure[i, j - 1]
                if self.cell_classification[i, j + 1] == Classification.Interior:
                    self.face_velocity_y[i, j] -= z * self.cell_pressure[i, j + 1]

    def solve(self):
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
        #     print(p[i])
        # print("SUCCESS??? ->", solver.info())

        # Apply the pressure to the intermediate velocity field.
        self.fill_pressure_field(p)
        self.apply_pressure()

        # print("BEFORE:")
        # self.compute_divergence()
        # print(self.cell_pressure)
        # print(self.cell_divergence)

        # print(self.face_volume_x)
        # print(self.face_mass_x)
        # print("-" * 200)

        # As = self.dt * self.inv_dx * self.inv_dx
        # bs = self.inv_dx

    # @ti.kernel
    # def compute_divergence(self):
    #     # TODO: this method is only for debugging and should go to some test file or ???
    #     for i, j in self.cell_divergence:
    #         if self.cell_classification[i, j] == Classification.Interior:
    #             x_divergence = self.face_velocity_x[i + 1, j] - self.face_velocity_x[i, j]
    #             y_divergence = self.face_velocity_y[i, j + 1] - self.face_velocity_y[i, j]
    #             self.cell_divergence[i, j] = x_divergence + y_divergence
    #         else:
    #             self.cell_divergence[i, j] = 0
