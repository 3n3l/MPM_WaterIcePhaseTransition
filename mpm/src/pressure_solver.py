from taichi.linalg import SparseMatrixBuilder, SparseSolver, SparseCG
from src.enums import Classification

import taichi as ti
import numpy as np


@ti.data_oriented
class PressureSolver:
    def __init__(self, mpm_solver, should_use_direct_solver: bool = True) -> None:
        self.n_cells = mpm_solver.n_grid * mpm_solver.n_grid
        self.boundary_width = mpm_solver.boundary_width
        self.inv_dx = mpm_solver.inv_dx
        self.n_grid = mpm_solver.n_grid
        self.dx = mpm_solver.dx
        self.dt = mpm_solver.dt

        self.classification_c = mpm_solver.classification_c
        self.inv_lambda_c = mpm_solver.inv_lambda_c
        self.pressure_c = mpm_solver.pressure_c
        self.JE_c = mpm_solver.JE_c
        self.JP_c = mpm_solver.JP_c

        self.classification_x = mpm_solver.classification_x
        self.classification_y = mpm_solver.classification_y
        self.velocity_x = mpm_solver.velocity_x
        self.velocity_y = mpm_solver.velocity_y
        self.volume_x = mpm_solver.volume_x
        self.volume_y = mpm_solver.volume_y
        self.mass_x = mpm_solver.mass_x
        self.mass_y = mpm_solver.mass_y

        self.should_use_direct_solver = should_use_direct_solver

    @ti.kernel
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
        inv_dx_squared = self.inv_dx * self.inv_dx

        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            # Raveled index.
            idx = (i * self.n_grid) + j

            # Build the right-hand side of the linear system.
            b[idx] = (1 - self.JE_c[i, j]) / (self.dt * self.JE_c[i, j])
            b[idx] -= self.inv_dx * (self.velocity_x[i + 1, j] - self.velocity_x[i, j])
            b[idx] -= self.inv_dx * (self.velocity_y[i, j + 1] - self.velocity_y[i, j])

            # FIXME: these variables are just used to print everything and can be removed after debugging
            A_t = 0.0
            A_l = 0.0
            A_c = 0.0
            A_r = 0.0
            A_b = 0.0

            # FIXME: this should be on non empty cells, but then the colliding
            #        simulation boundary results in underdetermined linear system
            # if self.c_classification[i, j] != Classification.Empty:
            if self.classification_c[i, j] == Classification.Interior:
                # TODO: save lambda in field instead of inverse (but compute inverse for stability)
                lambda_c = 1 / self.inv_lambda_c[i, j]
                # A[idx, idx] += delta * self.cell_JP[i, j] / (self.cell_JE[i, j] * cell_lambda * self.dt)
                A_c += self.JP_c[i, j] / (self.JE_c[i, j] * lambda_c * self.dt)

                # We will apply a Neumann boundary condition on the colliding faces,
                # to guarantee zero flux into colliding cells, by just not adding these
                # face values in the Laplacian for the off-diagonal values.
                if (
                    i != 0
                    and self.classification_x[i - 1, j] != Classification.Colliding
                    # FIXME: that the adjacent cell is empty shouldn't matter,
                    #        but without this the solver won't converge
                    and self.classification_c[i - 1, j] != Classification.Empty
                ):
                    # inv_rho = self.x_volume[i - 1, j] / self.x_mass[i - 1, j]
                    inv_rho = self.volume_x[i, j] / self.mass_x[i, j]
                    A[idx, idx - self.n_grid] -= self.dt * inv_dx_squared * inv_rho
                    A_l -= self.dt * inv_dx_squared * inv_rho
                    # A[idx, idx] += self.dt * Gic * inv_rho
                    A_c += self.dt * inv_dx_squared * inv_rho

                if (
                    i != self.n_grid - 1
                    and self.classification_x[i + 1, j] != Classification.Colliding
                    # FIXME: that the adjacent cell is empty shouldn't matter,
                    #        but without this the solver won't converge
                    and self.classification_c[i + 1, j] != Classification.Empty
                ):
                    inv_rho = self.volume_x[i + 1, j] / self.mass_x[i + 1, j]
                    A[idx, idx + self.n_grid] -= self.dt * inv_dx_squared * inv_rho
                    A_r -= self.dt * inv_dx_squared * inv_rho
                    # A[idx, idx] += self.dt * Gic * inv_rho
                    A_c += self.dt * inv_dx_squared * inv_rho

                if (
                    j != 0
                    and self.classification_y[i, j - 1] != Classification.Colliding
                    # FIXME: that the adjacent cell is empty shouldn't matter,
                    #        but without this the solver won't converge
                    and self.classification_c[i, j - 1] != Classification.Empty
                ):
                    # inv_rho = self.y_volume[i, j - 1] / self.y_mass[i, j - 1]
                    inv_rho = self.volume_y[i, j] / self.mass_y[i, j]
                    A[idx, idx - 1] -= self.dt * inv_dx_squared * inv_rho
                    A_b -= self.dt * inv_dx_squared * inv_rho
                    # A[idx, idx] += self.dt * Gic * inv_rho
                    A_c += self.dt * inv_dx_squared * inv_rho

                if (
                    j != self.n_grid - 1
                    and self.classification_y[i, j + 1] != Classification.Colliding
                    # FIXME: that the adjacent cell is empty shouldn't matter,
                    #        but without this the solver won't converge
                    and self.classification_c[i, j + 1] != Classification.Empty
                ):
                    inv_rho = self.volume_y[i, j + 1] / self.mass_y[i, j + 1]
                    A[idx, idx + 1] -= self.dt * inv_dx_squared * inv_rho
                    A_t -= self.dt * inv_dx_squared * inv_rho
                    # A[idx, idx] += self.dt * Gic * inv_rho
                    A_c += self.dt * inv_dx_squared * inv_rho

                A[idx, idx] += A_c

            else:  # Homogeneous Dirichlet boundary condition.
                A[idx, idx] += 1.0
                b[idx] = 0.0
                A_c += 1.0

            continue
            if self.classification_c[i, j] != Classification.Interior:
                continue
            print("~" * 100)
            print()
            if self.classification_c[i, j] == Classification.Interior:
                print(f">>> INTERIOR, idx = {idx}, i = {i}, j = {j}")
            elif self.classification_c[i, j] == Classification.Colliding:
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
            print(f"x_velocity[i, j]      ->", self.velocity_x[i, j])
            # print(f"x_velocity[i + 1, j]  ->", self.x_velocity[i + 1, j])
            # print(f"x_velocity[i - 1, j]  ->", self.x_velocity[i - 1, j])

            print()
            print(f"y_velocity[i, j]      ->", self.velocity_y[i, j])
            # print(f"y_velocity[i, j - 1]  ->", self.y_velocity[i, j - 1])
            # print(f"y_velocity[i, j + 1]  ->", self.y_velocity[i, j + 1])

            print()
            print(f"c_JE[i, j]            ->", self.JE_c[i, j])
            print(f"c_JP[i, j]            ->", self.JP_c[i, j])
            print(f"c_inv_lambda[i, j]    ->", self.inv_lambda_c[i, j])

            print()
            print(f"b[{idx}]                ->", b[idx])
            print()

    @ti.kernel
    def fill_pressure_field(self, p: ti.types.ndarray()):  # pyright: ignore
        # TODO: move this to apply_pressure, delete self.cell_pressure (IF POSSIBLE)
        for i, j in self.pressure_c:
            row = (i * self.n_grid) + j
            self.pressure_c[i, j] = p[row]

    @ti.kernel
    def apply_pressure(self):
        z = self.inv_dx * self.dt
        for i, j in ti.ndrange((1, self.n_grid), (0, self.n_grid - 1)):
            # TODO: this could just be done for the classification step and then saved into a field?
            x_face_is_not_interior = self.classification_c[i - 1, j] != Classification.Interior
            x_face_is_not_interior &= self.classification_c[i, j] != Classification.Interior
            if x_face_is_not_interior:
                continue  # don't bother

            x_face_is_colliding = self.classification_c[i - 1, j] == Classification.Colliding
            x_face_is_colliding |= self.classification_c[i, j] == Classification.Colliding
            if x_face_is_colliding:
                self.velocity_x[i, j] = 0
                continue

            # Backward difference between the two adjacent cells.
            inv_rho = self.volume_x[i, j] / self.mass_x[i, j]
            self.velocity_x[i, j] -= z * inv_rho * (self.pressure_c[i, j] - self.pressure_c[i - 1, j])

        for i, j in ti.ndrange((0, self.n_grid - 1), (1, self.n_grid)):
            y_face_is_not_interior = self.classification_c[i, j - 1] != Classification.Interior
            y_face_is_not_interior &= self.classification_c[i, j] != Classification.Interior
            if y_face_is_not_interior:
                continue  # don't bother

            y_face_is_colliding = self.classification_c[i, j - 1] == Classification.Colliding
            y_face_is_colliding |= self.classification_c[i, j] == Classification.Colliding
            if y_face_is_colliding:
                self.velocity_y[i, j] = 0
                continue

            # Backward difference between the two adjacent cells.
            inv_rho = self.volume_y[i, j] / self.mass_y[i, j]
            self.velocity_y[i, j] -= z * inv_rho * (self.pressure_c[i, j] - self.pressure_c[i, j - 1])

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
        # TODO: create boolean and cli argument to control this

        # Solve the linear system.
        if self.should_use_direct_solver:
            solver = SparseSolver(dtype=ti.f32, solver_type="LLT")
            solver.compute(A.build())
            p = solver.solve(b)
            # FIXME: remove this debugging statements or move to test file
            solver_succeeded, pressure = solver.info(), p.to_numpy()
            assert solver_succeeded, "SOLVER DID NOT FIND A SOLUTION!"
            assert not np.any(np.isnan(pressure)), "NAN VALUE IN PRESSURE ARRAY!"
        else:
            solver = SparseCG(A.build(), b)
            p, _ = solver.solve()

        # FIXME: Apply the pressure to the intermediate velocity field.
        self.fill_pressure_field(p)
        self.apply_pressure()
