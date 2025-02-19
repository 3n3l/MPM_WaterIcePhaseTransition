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
        # self.rho_0 = mpm_solver.rho_0
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

    @ti.kernel
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
        # delta = 0.00032  # relaxation
        delta = 1.0  # relaxation
        # delta = 1.0 / self.dt  # relaxation
        # TODO: as the scaling is on both sides of the equation this could just be ignored??
        # Gic = self.inv_dx * self.inv_dx * self.inv_dx * self.inv_dx  # central difference coefficients
        Gic = self.inv_dx * self.inv_dx  # central difference coefficients
        # Gic = 1.0  # central difference coefficients

        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            # Unraveled index.
            idx = (i * self.n_grid) + j

            b[idx] = -((self.cell_JE[i, j] - 1) / (self.dt * self.cell_JE[i, j]))
            # Velocities components on colliding faces are considered to be zero.
            # if i != 0: # and self.x_classification[i - 1, j] != Classification.Colliding:
            #     b[idx] += self.inv_dx * (self.x_velocity[i - 1, j] - self.x_velocity[i, j])
            if i != self.n_grid - 1:  # and self.x_classification[i + 1, j] != Classification.Colliding:
                b[idx] -= self.inv_dx * (self.x_velocity[i + 1, j] - self.x_velocity[i, j])
            # if j != 0: # and self.y_classification[i, j - 1] != Classification.Colliding:
            #     b[idx] += self.inv_dx * (self.y_velocity[i, j - 1] - self.y_velocity[i, j])
            if j != self.n_grid - 1:  # and self.y_classification[i, j + 1] != Classification.Colliding:
                b[idx] -= self.inv_dx * (self.y_velocity[i, j + 1] - self.y_velocity[i, j])

            A_t = 0.0
            A_l = 0.0
            A_c = 0.0
            A_r = 0.0
            A_b = 0.0
            # if self.c_classification[i, j] != Classification.Empty:
            if self.c_classification[i, j] == Classification.Interior:
                # TODO: What about colliding cells? They are also empty, but the paper does not talk about this?
                cell_lambda = 1 / self.cell_inv_lambda[i, j]  # TODO: save lambda instead of inverse (but compute inverse)
                # A[idx, idx] += delta * self.cell_JP[i, j] / (self.cell_JE[i, j] * cell_lambda * self.dt)
                A_c += delta * self.cell_JP[i, j] / (self.cell_JE[i, j] * cell_lambda * self.dt)

                # FIXME: error somewhere in here
                if i != 0 and self.x_classification[i - 1, j] != Classification.Colliding:
                    inv_rho = self.x_volume[i - 1, j] / self.x_mass[i - 1, j]
                    # if self.x_classification[i - 1, j] == Classification.Interior:
                    A[idx, idx - self.n_grid] -= self.dt * delta * Gic * inv_rho
                    A_l -= self.dt * delta * Gic * inv_rho

                    # A[idx, idx] += self.dt * delta * Gic * inv_rho
                    A_c += self.dt * delta * Gic * inv_rho


                if i != self.n_grid - 1 and self.x_classification[i + 1, j] != Classification.Colliding:
                    inv_rho = self.x_volume[i + 1, j] / self.x_mass[i + 1, j]
                    # if self.x_classification[i + 1, j] == Classification.Interior:
                    A[idx, idx + self.n_grid] -= self.dt * delta * Gic * inv_rho
                    A_r -= self.dt * delta * Gic * inv_rho

                    # A[idx, idx] += self.dt * delta * Gic * inv_rho
                    A_c += self.dt * delta * Gic * inv_rho

                if j != 0 and self.y_classification[i, j - 1] != Classification.Colliding:
                    inv_rho = self.y_volume[i, j - 1] / self.y_mass[i, j - 1]
                    # if self.y_classification[i, j - 1] == Classification.Interior:
                    A[idx, idx - 1] -= self.dt * delta * Gic * inv_rho
                    A_b -= self.dt * delta * Gic * inv_rho

                    # A[idx, idx] += self.dt * delta * Gic * inv_rho
                    A_c += self.dt * delta * Gic * inv_rho

                if j != self.n_grid - 1 and self.y_classification[i, j + 1] != Classification.Colliding:
                    inv_rho = self.y_volume[i, j + 1] / self.y_mass[i, j + 1]
                    # if self.y_classification[i, j + 1] == Classification.Interior:
                    A[idx, idx + 1] -= self.dt * delta * Gic * inv_rho
                    A_t -= self.dt * delta * Gic * inv_rho

                    # A[idx, idx] += self.dt * delta * Gic * inv_rho
                    A_c += self.dt * delta * Gic * inv_rho

                A[idx, idx] += A_c

            else:  # Homogeneous Dirichlet boundary condition.
                # A[idx, idx] += self.dt * delta * Gic
                A[idx, idx] += 1.0
                b[idx] = 0.0

                A_c += 1.0
                # A_c += self.dt * delta * Gic

            nan_found =  False
            nan_found |= A_t != A_t
            nan_found |= A_l != A_l
            nan_found |= A_c != A_c
            nan_found |= A_r != A_r
            nan_found |= A_b != A_b
            nan_found |= b[idx] != b[idx]
            if not nan_found:
            # if self.c_classification[i, j] != Classification.Interior:
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
            print(f"b[{idx}]                ->", b[idx])
            print()

    @ti.kernel
    def fill_pressure_field(self, p: ti.types.ndarray()):  # pyright: ignore
        # TODO: move this to apply_pressure, delete self.cell_pressure (IF POSSIBLE)
        for i, j in self.c_pressure:
            row = (i * self.n_grid) + j
            self.c_pressure[i, j] = p[row]

    @ti.kernel
    def apply_pressure(self):
        left_boundary = self.boundary_width
        right_boundary = self.n_grid - self.boundary_width
        # z = self.dt * self.inv_dx * self.inv_dx  # Central Difference Coefficients
        z = self.dt
        # pressure = ti.static(self.c_pressure)

        for i, j in ti.ndrange((left_boundary, right_boundary + 1), (left_boundary, right_boundary)):
            # TODO: this could just be done for the classification step and then saved into a field?
            x_face_is_not_interior = self.c_classification[i - 1, j] != Classification.Interior
            x_face_is_not_interior |= self.c_classification[i, j] != Classification.Interior
            if x_face_is_not_interior:
                continue  # don't bother

            x_face_is_colliding = self.c_classification[i - 1, j] == Classification.Colliding
            x_face_is_colliding |= self.c_classification[i, j] == Classification.Colliding
            if x_face_is_colliding:
                self.x_velocity[i, j] = 0
                continue

            # inv_rho = self.x_volume[i + 1, j] / self.x_mass[i + 1, j]
            # self.x_velocity[i, j] -= z * inv_rho * (self.c_pressure[i + 1, j] - self.c_pressure[i, j])

            inv_rho = self.x_volume[i - 1, j] / self.x_mass[i - 1, j]
            self.x_velocity[i, j] -= z * inv_rho * (self.c_pressure[i, j] - self.c_pressure[i - 1, j])

            # inv_rho = self.x_volume[i, j + 1] / self.x_mass[i, j + 1]
            # self.x_velocity[i, j] -= z * inv_rho * (self.c_pressure[i, j + 1] - self.c_pressure[i, j])
            #
            # inv_rho = self.x_volume[i, j - 1] / self.x_mass[i, j - 1]
            # self.x_velocity[i, j] -= z * inv_rho * (self.c_pressure[i, j - 1] - self.c_pressure[i, j])

            # print(">" * 100)
            # # print("v_i:")
            # # print(self.x_volume[i, j])
            # # print(self.x_volume[i + 1, j])
            # # print(self.x_volume[i - 1, j])
            # # print(self.x_volume[i, j + 1])
            # # print(self.x_volume[i, j - 1])
            # # print("m_i:")
            # # print(self.x_mass[i, j])
            # # print(self.x_mass[i + 1, j])
            # # print(self.x_mass[i - 1, j])
            # # print(self.x_mass[i, j + 1])
            # # print(self.x_mass[i, j - 1])
            # print("1 / rho_i:")
            # print(self.x_volume[i + 1, j] / self.x_mass[i + 1, j])
            # print(self.x_volume[i - 1, j] / self.x_mass[i - 1, j])
            # print(self.x_volume[i, j + 1] / self.x_mass[i, j + 1])
            # print(self.x_volume[i, j - 1] / self.x_mass[i, j - 1])
            # print("p_c:")
            # print(self.c_pressure[i, j])
            # print(self.c_pressure[i + 1, j])
            # print(self.c_pressure[i - 1, j])
            # print(self.c_pressure[i, j + 1])
            # print(self.c_pressure[i, j - 1])
            # print("x_v:")
            # print(self.x_velocity[i, j])

        for i, j in ti.ndrange((left_boundary, right_boundary), (left_boundary, right_boundary + 1)):
            y_face_is_not_interior = self.c_classification[i, j - 1] != Classification.Interior
            y_face_is_not_interior |= self.c_classification[i, j] != Classification.Interior
            if y_face_is_not_interior:
                continue  # don't bother

            y_face_is_colliding = self.c_classification[i, j - 1] == Classification.Colliding
            y_face_is_colliding |= self.c_classification[i, j] == Classification.Colliding
            if y_face_is_colliding:
                self.y_velocity[i, j] = 0
                continue

            inv_rho = self.y_volume[i, j - 1] / self.y_mass[i, j - 1]
            self.y_velocity[i, j] -= z * inv_rho * (self.c_pressure[i, j] - self.c_pressure[i, j - 1])

            # inv_rho = self.y_volume[i - 1, j] / self.y_mass[i - 1, j]
            # self.y_velocity[i, j] -= z * inv_rho * (self.c_pressure[i - 1, j] - self.c_pressure[i, j])
            #
            # inv_rho = self.y_volume[i, j + 1] / self.y_mass[i, j + 1]
            # self.y_velocity[i, j] -= z * inv_rho * (self.c_pressure[i, j + 1] - self.c_pressure[i, j])
            #
            # inv_rho = self.y_volume[i, j - 1] / self.y_mass[i, j - 1]
            # self.y_velocity[i, j] -= z * inv_rho * (self.c_pressure[i, j - 1] - self.c_pressure[i, j])

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

        # A.print_triplets()

        # Solve the linear system.
        solver = SparseSolver(dtype=ti.f32, solver_type="LLT")
        solver.compute(A.build())
        p = solver.solve(b)

        # FIXME: remove this debugging statements or move to test file
        solver_succeeded, pressure = solver.info(), p.to_numpy()
        assert solver_succeeded, "SOLVER DID NOT FIND A SOLUTION!"
        assert not np.any(np.isnan(pressure)), "NAN VALUE IN PRESSURE ARRAY!"

        # FIXME: Apply the pressure to the intermediate velocity field.
        self.fill_pressure_field(p)
        self.apply_pressure()
