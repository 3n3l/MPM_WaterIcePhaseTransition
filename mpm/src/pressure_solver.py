from taichi.linalg import SparseMatrixBuilder, SparseSolver
from src.enums import Classification

import taichi as ti
import numpy as np


@ti.data_oriented
class PressureSolver:
    def __init__(self, mpm_solver) -> None:
        self.n_cells = mpm_solver.n_grid * mpm_solver.n_grid
        self.boundary_width = mpm_solver.boundary_width
        self.inv_dx = mpm_solver.inv_dx
        self.n_grid = mpm_solver.n_grid
        self.dx = mpm_solver.dx
        self.dt = mpm_solver.dt

        self.c_classification = mpm_solver.cell_classification
        self.cell_inv_lambda = mpm_solver.cell_inv_lambda
        self.c_pressure = mpm_solver.cell_pressure
        self.cell_JE = mpm_solver.cell_JE
        self.cell_JP = mpm_solver.cell_JP

        self.x_classification = mpm_solver.face_classification_x
        self.y_classification = mpm_solver.face_classification_y
        self.x_velocity = mpm_solver.face_velocity_x
        self.y_velocity = mpm_solver.face_velocity_y
        self.x_volume = mpm_solver.face_volume_x
        self.y_volume = mpm_solver.face_volume_y
        self.x_mass = mpm_solver.face_mass_x
        self.y_mass = mpm_solver.face_mass_y
    
    @ti.func
    def flatten_gpos_row_major(self, i, j): # pyright: ignore
        return i * self.n_grid + j
    
    @ti.func
    def fold_index_row_major(self, idx): # pyright: ignore
        return (idx // self.n_grid, idx % self.n_grid)

    @ti.kernel
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
        Gic = self.inv_dx * self.inv_dx

        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            # Unraveled index.
            idx = self.flatten_gpos_row_major(i, j)

            # b[idx] = (1 - self.cell_JE[i, j]) / (self.dt * self.cell_JE[i, j])
            b[idx] = -((self.cell_JE[i, j] - 1) / (self.dt * self.cell_JE[i, j]))

            if i != self.n_grid - 1:
                # b[idx] -= 2 * self.inv_dx * (self.x_velocity[i + 1, j] - self.x_velocity[i, j])
                b[idx] -= self.inv_dx * (self.x_velocity[i + 1, j] - self.x_velocity[i, j])
            if j != self.n_grid - 1:
                # b[idx] -= 2 * self.inv_dx * (self.y_velocity[i, j + 1] - self.y_velocity[i, j])
                b[idx] -= self.inv_dx * (self.y_velocity[i, j + 1] - self.y_velocity[i, j])

            # FIXME: these variables are just used to print everything and can be removed after debugging
            A_t = 0.0
            A_l = 0.0
            A_c = 0.0
            A_r = 0.0
            A_b = 0.0

            # FIXME: this should be on non empty cells, but then the colliding
            #        simulation boundary results in underdetermined linear system
            # if self.c_classification[i, j] != Classification.Empty:
            if self.c_classification[i, j] == Classification.Interior:
                # TODO: save lambda in field instead of inverse (but compute inverse for stability)
                cell_lambda = 1. / self.cell_inv_lambda[i, j]
                # A[idx, idx] += delta * self.cell_JP[i, j] / (self.cell_JE[i, j] * cell_lambda * self.dt)
                A_c += self.cell_JP[i, j] / (self.cell_JE[i, j] * cell_lambda * self.dt)

                # We will apply a Neumann boundary condition on the colliding faces,
                # to guarantee zero flux into colliding cells, by just not adding these
                # face values in the Laplacian for the off-diagonal values.
                if (
                    i >= self.boundary_width
                    and self.x_classification[i - 1, j] != Classification.Colliding
                    # FIXME: that the adjacent cell is empty shouldn't matter,
                    #        but without this the solver won't converge
                    and self.c_classification[i - 1, j] != Classification.Empty
                ):
                    # inv_rho = self.x_volume[i - 1, j] / self.x_mass[i - 1, j]
                    inv_rho = self.x_volume[i, j] / self.x_mass[i, j]
                    # A[idx, idx - self.n_grid] -= self.dt * Gic * inv_rho
                    A[idx, self.flatten_gpos_row_major(i-1, j)] -= self.dt * Gic * inv_rho
                    A_l -= self.dt * Gic * inv_rho
                    # A[idx, idx] += self.dt * Gic * inv_rho
                    A_c += self.dt * Gic * inv_rho

                if (
                    i <= self.n_grid - self.boundary_width
                    and self.x_classification[i + 1, j] != Classification.Colliding
                    # FIXME: that the adjacent cell is empty shouldn't matter,
                    #        but without this the solver won't converge
                    and self.c_classification[i + 1, j] != Classification.Empty
                ):
                    inv_rho = self.x_volume[i + 1, j] / self.x_mass[i + 1, j]
                    # A[idx, idx + self.n_grid] -= self.dt * Gic * inv_rho
                    A[idx, self.flatten_gpos_row_major(i+1, j)] -= self.dt * Gic * inv_rho
                    A_r -= self.dt * Gic * inv_rho
                    # A[idx, idx] += self.dt * Gic * inv_rho
                    A_c += self.dt * Gic * inv_rho

                if (
                    j >= self.boundary_width
                    and self.y_classification[i, j - 1] != Classification.Colliding
                    # FIXME: that the adjacent cell is empty shouldn't matter,
                    #        but without this the solver won't converge
                    and self.c_classification[i, j - 1] != Classification.Empty
                ):
                    # inv_rho = self.y_volume[i, j - 1] / self.y_mass[i, j - 1]
                    inv_rho = self.y_volume[i, j] / self.y_mass[i, j]
                    # A[idx, idx - 1] -= self.dt * Gic * inv_rho
                    A[idx, self.flatten_gpos_row_major(i, j-1)] -= self.dt * Gic * inv_rho
                    A_b -= self.dt * Gic * inv_rho
                    # A[idx, idx] += self.dt * Gic * inv_rho
                    A_c += self.dt * Gic * inv_rho

                if (
                    j <= self.n_grid - self.boundary_width
                    and self.y_classification[i, j + 1] != Classification.Colliding
                    # FIXME: that the adjacent cell is empty shouldn't matter,
                    #        but without this the solver won't converge
                    and self.c_classification[i, j + 1] != Classification.Empty
                ):
                    inv_rho = self.y_volume[i, j + 1] / self.y_mass[i, j + 1]
                    # A[idx, idx + 1] -= self.dt * Gic * inv_rho
                    A[idx, self.flatten_gpos_row_major(i, j+1)] -= self.dt * Gic * inv_rho
                    A_t -= self.dt * Gic * inv_rho
                    # A[idx, idx] += self.dt * Gic * inv_rho
                    A_c += self.dt * Gic * inv_rho

                A[idx, idx] += A_c

            else:  # Homogeneous Dirichlet boundary condition.
                A[idx, idx] += 1.0
                b[idx] = 0.0
                # A_c += 1.0

            continue
            if self.c_classification[i, j] != Classification.Interior:
                continue
            print("~" * 100)
            print()
            if self.c_classification[i, j] == Classification.Interior:
                print(f">>> INTERIOR, idx = {idx}, i = {i}, j = {j}")
            elif self.c_classification[i, j] == Classification.Colliding:
                print(f">>> COLLIDING, idx = {idx}, i = {i}, j = {j}")
            else:
                print(f">>> EMPTY, idx = {idx}, i = {i}, j = {j}")

            print()
            print(f"A[{idx}, {idx} + 1]   ->", A_t)
            print(f"A[{idx} - 1, {idx}]   ->", A_l)
            print(f"A[{idx}, {idx}]       ->", A_c)
            print(f"A[{idx} + 1, {idx}]   ->", A_r)
            print(f"A[{idx}, {idx} - 1]   ->", A_b)

            print()
            print(f"x_velocity[i, j]      ->", self.x_velocity[i, j])
            # print(f"x_velocity[i + 1, j]  ->", self.x_velocity[i + 1, j])
            # print(f"x_velocity[i - 1, j]  ->", self.x_velocity[i - 1, j])

            print()
            print(f"y_velocity[i, j]      ->", self.y_velocity[i, j])
            # print(f"y_velocity[i, j - 1]  ->", self.y_velocity[i, j - 1])
            # print(f"y_velocity[i, j + 1]  ->", self.y_velocity[i, j + 1])

            print()
            print(f"c_JE[i, j]            ->", self.cell_JE[i, j])
            print(f"c_JP[i, j]            ->", self.cell_JP[i, j])
            print(f"c_inv_lambda[i, j]    ->", self.cell_inv_lambda[i, j])

            print()
            print(f"b[{idx}]                ->", b[idx])
            print()

    @ti.kernel
    def fill_pressure_field(self, p: ti.types.ndarray()):  # pyright: ignore
        # TODO: move this to apply_pressure, delete self.cell_pressure (IF POSSIBLE)
        for i, j in self.c_pressure:
            # row = (i * self.n_grid) + j
            idx = self.flatten_gpos_row_major(i, j)
            self.c_pressure[i, j] = p[idx]

    @ti.kernel
    def apply_pressure(self):
        z = self.inv_dx * self.dt
        for i, j in ti.ndrange((1, self.n_grid), (0, self.n_grid - 1)):
            # TODO: this could just be done for the classification step and then saved into a field?
            x_face_is_not_interior = self.c_classification[i - 1, j] != Classification.Interior
            x_face_is_not_interior &= self.c_classification[i, j] != Classification.Interior
            if x_face_is_not_interior:
                continue  # don't bother

            x_face_is_colliding = self.c_classification[i - 1, j] == Classification.Colliding
            x_face_is_colliding |= self.c_classification[i, j] == Classification.Colliding
            if x_face_is_colliding:
                self.x_velocity[i, j] = 0
                continue

            # Backward difference between the two adjacent cells.
            inv_rho = self.x_volume[i, j] / self.x_mass[i, j]
            self.x_velocity[i, j] -= z * inv_rho * (self.c_pressure[i, j] - self.c_pressure[i - 1, j])

        for i, j in ti.ndrange((0, self.n_grid - 1), (1, self.n_grid)):
            y_face_is_not_interior = self.c_classification[i, j - 1] != Classification.Interior
            y_face_is_not_interior &= self.c_classification[i, j] != Classification.Interior
            if y_face_is_not_interior:
                continue  # don't bother

            y_face_is_colliding = self.c_classification[i, j - 1] == Classification.Colliding
            y_face_is_colliding |= self.c_classification[i, j] == Classification.Colliding
            if y_face_is_colliding:
                self.y_velocity[i, j] = 0
                continue
            
            # Backward difference between the two adjacent cells.
            inv_rho = self.y_volume[i, j] / self.y_mass[i, j]
            self.y_velocity[i, j] -= z * inv_rho * (self.c_pressure[i, j] - self.c_pressure[i, j - 1])

    def solve(self):
        # TODO: max_num_triplets could be optimized to N * 5?
        A = SparseMatrixBuilder(
            max_num_triplets=(self.n_cells * self.n_cells),
            num_rows=self.n_cells,
            num_cols=self.n_cells,
            dtype=ti.f32,
        )
        b = ti.ndarray(ti.f32, shape=self.n_cells)
        self.fill_linear_system(A, b)

        # Solve the linear system.
        solver = SparseSolver(dtype=ti.f32, solver_type="LLT")
        solver.compute(A.build())
        p = solver.solve(b)

        # print("^" * 100)
        # print()
        # print(">>> A")
        # print(K)
        # print()
        # print(">>> b")
        # # print(b.to_numpy())
        # for bbb in b.to_numpy():
        #     print(bbb)
        # print()
        # print(">>> p")
        # # print(p.to_numpy())
        # for ppp in p.to_numpy():
        #     print(ppp)

        # FIXME: remove this debugging statements or move to test file
        solver_succeeded, pressure = solver.info(), p.to_numpy()
        assert solver_succeeded, "SOLVER DID NOT FIND A SOLUTION!"
        assert not np.any(np.isnan(pressure)), "NAN VALUE IN PRESSURE ARRAY!"

        # FIXME: Apply the pressure to the intermediate velocity field.
        self.fill_pressure_field(p)
        self.apply_pressure()