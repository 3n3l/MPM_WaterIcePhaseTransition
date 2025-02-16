from taichi.linalg import SparseMatrixBuilder, SparseSolver
from src.enums import Classification

import taichi as ti


@ti.data_oriented
class PressureSolver:
    def __init__(self, mpm_solver) -> None:
        self.n_cells = mpm_solver.n_grid * mpm_solver.n_grid
        self.n_grid = mpm_solver.n_grid
        self.inv_dx = mpm_solver.inv_dx
        self.dx = mpm_solver.dx
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

    @ti.func
    def sample(self, some_field, i_index, j_index):
        I = ti.Vector([int(i_index), int(j_index)])
        I = ti.max(0, ti.min(self.n_grid - 1, I))
        return some_field[I]

    @ti.kernel
    def fill_linear_system(
        self,
        A: ti.types.sparse_matrix_builder(),  # pyright: ignore
        B: ti.types.sparse_matrix_builder(),  # pyright: ignore
        C: ti.types.sparse_matrix_builder(),  # pyright: ignore
        b: ti.types.ndarray(),  # pyright: ignore
    ):
        # TODO: Laplacian L needs to be filled according to paper.
        # TODO: Solution vector b needs to be filled according to paper.
        # TODO: do we need relaxation here?
        # delta = 0.00032  # relaxation
        # z = self.dt * self.inv_dx * self.inv_dx * inv_rho

        # TODO: could be something like:
        # (see https://gitee.com/citadel2020/taichi_demos/blob/master/mgpcgflip/mgpcgflip.py)
        # s = 4.0
        # s -= float(self.is_solid(self.f[l], i - 1, j, _nx, _ny))
        # s -= float(self.is_solid(self.f[l], i + 1, j, _nx, _ny))
        # s -= float(self.is_solid(self.f[l], i, j - 1, _nx, _ny))
        # s -= float(self.is_solid(self.f[l], i, j + 1, _nx, _ny))

        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            # TODO: b: use central difference for face values (TODO: also for everything else???) or maybe not?
            #       MAC would use forward differences there, as this is supposed to simulate divergence the
            #       forward differences make a lot more sense???

            # Unraveled index.
            idx = (i * self.n_grid) + j

            # if self.cell_classification[i, j] != Classification.Interior:
            #     continue  # don't bother

            # Fill the left hand side of the linear system.
            A[idx, idx] += self.cell_inv_lambda[i, j] * self.cell_JP[i, j] * (1 / (self.cell_JE[i, j] * self.dt))

            # Fill the right hand side of the linear system.
            b[idx] = (-1) * (self.cell_JE[i, j] - 1) / (self.dt * self.cell_JE[i, j])

            # TODO: b[i]: ??? BC might just be setting the velocities between interior and colliding cells to zero
            # TODO: L|M[i, i]: is set to the amount of NON COLLIDING neighbors
            # We apply the ??? boundary condition to solution vector b by treating the velocities between
            # interior and colliding cells as zero.
            if (i != 0) and (self.cell_classification[i - 1, j] != Classification.Colliding):
                b[idx] -= self.inv_dx * self.face_velocity_x[i, j]
                B[idx, idx] += self.face_volume_x[i, j] / self.face_mass_x[i, j]
                C[idx, idx] += 1.0
            if (i != self.n_grid - 1) and (self.cell_classification[i + 1, j] != Classification.Colliding):
                b[idx] += self.inv_dx * self.face_velocity_x[i + 1, j]
                B[idx, idx] += self.face_volume_x[i, j] / self.face_mass_x[i, j]
                C[idx, idx] += 1.0
            if (j != 0) and (self.cell_classification[i, j - 1] != Classification.Colliding):
                b[idx] -= self.inv_dx * self.face_velocity_y[i, j]
                B[idx, idx] += self.face_volume_y[i, j] / self.face_mass_y[i, j]
                C[idx, idx] += 1.0
            if (j != self.n_grid - 1) and (self.cell_classification[i, j + 1] != Classification.Colliding):
                b[idx] += self.inv_dx * self.face_velocity_y[i, j + 1]
                B[idx, idx] += self.face_volume_y[i, j] / self.face_mass_y[i, j]
                C[idx, idx] += 1.0

            # TODO: b: Dirichlet BC might just be subtracting the pressure of empty cells?
            #       But the ambient/atmopsheric pressure is set to zero atm, so this can just be skipped?

            # TODO: B|D[i, ?] is set to one if [i, ?] IS INTERIOR
            if (i != 0) and (self.cell_classification[i - 1, j] == Classification.Interior):
                B[idx, idx - self.n_grid] -= self.face_volume_x[i, j] / self.face_mass_x[i, j]
                C[idx, idx - self.n_grid] -= 1.0
            if (i != self.n_grid - 1) and (self.cell_classification[i + 1, j] == Classification.Interior):
                B[idx, idx + self.n_grid] -= self.face_volume_x[i + 1, j] / self.face_mass_x[i + 1, j]
                C[idx, idx + self.n_grid] -= 1.0
            if (j != 0) and (self.cell_classification[i, j - 1] == Classification.Interior):
                B[idx, idx - 1] -= self.face_volume_y[i, j] / self.face_mass_y[i, j]
                C[idx, idx - 1] -= 1.0
            if (j != self.n_grid - 1) and (self.cell_classification[i, j + 1] == Classification.Interior):
                B[idx, idx + 1] -= self.face_volume_y[i, j + 1] / self.face_mass_y[i, j + 1]
                C[idx, idx + 1] -= 1.0

    @ti.kernel
    def fill_pressure_field(self, p: ti.types.ndarray()):  # pyright: ignore
        # TODO: move this to apply_pressure, delete self.cell_pressure (IF POSSIBLE)
        for i, j in self.cell_pressure:
            row = (i * self.n_grid) + j
            self.cell_pressure[i, j] = p[row]

    @ti.kernel
    def apply_pressure(self):
        left_boundary = self.boundary_width
        right_boundary = self.n_grid - self.boundary_width
        z = self.dt * self.inv_dx * self.inv_dx  # Central Difference Coefficients

        for i, j in ti.ndrange((left_boundary, right_boundary + 1), (left_boundary, right_boundary)):
            # TODO: this could just be done for the classification step and then saved into a field?
            x_face_is_not_interior = self.cell_classification[i - 1, j] != Classification.Interior
            x_face_is_not_interior |= self.cell_classification[i, j] != Classification.Interior
            if x_face_is_not_interior:
                continue  # don't bother

            x_face_is_colliding = self.cell_classification[i - 1, j] == Classification.Colliding
            x_face_is_colliding |= self.cell_classification[i, j] == Classification.Colliding
            if x_face_is_colliding:
                self.x_velocity[i, j] = 0
                continue

            self.x_velocity[i, j] += 4 * z * (self.x_volume[i, j] / self.x_mass[i, j]) * self.c_pressure[i, j]
            self.x_velocity[i, j] -= z * (self.x_volume[i + 1, j] / self.x_mass[i + 1, j]) * self.c_pressure[i + 1, j]
            self.x_velocity[i, j] -= z * (self.x_volume[i - 1, j] / self.x_mass[i - 1, j]) * self.c_pressure[i - 1, j]
            self.x_velocity[i, j] -= z * (self.x_volume[i, j + 1] / self.x_mass[i, j + 1]) * self.c_pressure[i, j + 1]
            self.x_velocity[i, j] -= z * (self.x_volume[i, j - 1] / self.x_mass[i, j - 1]) * self.c_pressure[i, j - 1]

        for i, j in ti.ndrange((left_boundary, right_boundary), (left_boundary, right_boundary + 1)):
            y_face_is_not_interior = self.cell_classification[i, j - 1] != Classification.Interior
            y_face_is_not_interior |= self.cell_classification[i, j] != Classification.Interior
            if y_face_is_not_interior:
                continue  # don't bother

            y_face_is_colliding = self.cell_classification[i - 1, j] == Classification.Colliding
            y_face_is_colliding |= self.cell_classification[i, j] == Classification.Colliding
            if y_face_is_colliding:
                self.y_velocity[i, j] = 0
                continue

            self.y_velocity[i, j] += 4 * z * (self.y_volume[i, j] / self.y_mass[i, j]) * self.c_pressure[i, j]
            self.y_velocity[i, j] -= z * (self.y_volume[i + 1, j] / self.y_mass[i + 1, j]) * self.c_pressure[i + 1, j]
            self.y_velocity[i, j] -= z * (self.y_volume[i - 1, j] / self.y_mass[i - 1, j]) * self.c_pressure[i - 1, j]
            self.y_velocity[i, j] -= z * (self.y_volume[i, j + 1] / self.y_mass[i, j + 1]) * self.c_pressure[i, j + 1]
            self.y_velocity[i, j] -= z * (self.y_volume[i, j - 1] / self.y_mass[i, j - 1]) * self.c_pressure[i, j - 1]

    def solve(self):
        A = SparseMatrixBuilder(self.n_cells, self.n_cells, max_num_triplets=(self.n_cells * 5))
        B = SparseMatrixBuilder(self.n_cells, self.n_cells, max_num_triplets=(self.n_cells * 5))
        C = SparseMatrixBuilder(self.n_cells, self.n_cells, max_num_triplets=(self.n_cells * 5))
        b = ti.ndarray(ti.f32, shape=self.n_cells)
        self.fill_linear_system(A, B, C, b)
        L, M, N = A.build(), B.build(), C.build()
        R = L + (self.dt * self.inv_dx * self.inv_dx * (M @ N))
        # R = L + (M @ N)

        # Solve the linear system.
        solver = SparseSolver()
        solver.compute(R)
        p = solver.solve(b)

        solver_succeeded = solver.info()
        assert solver_succeeded, "SOLVER DID NOT FIND A SOLUTION!"

        # Apply the pressure to the intermediate velocity field.
        self.fill_pressure_field(p)
        self.apply_pressure()
