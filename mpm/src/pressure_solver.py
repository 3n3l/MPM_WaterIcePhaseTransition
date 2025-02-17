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
        self.rho_0 = mpm_solver.rho_0
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
    def fill_linear_system(
        self,
        D_s: ti.types.sparse_matrix_builder(),  # pyright: ignore
        L_r: ti.types.sparse_matrix_builder(),  # pyright: ignore
        L_p: ti.types.sparse_matrix_builder(),  # pyright: ignore
        b: ti.types.ndarray(),  # pyright: ignore
    ):
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            # if self.c_classification[i, j] == Classification.Interior:
            #     print("xv    ->", self.x_volume[i, j])
            #     print("yv    ->", self.y_volume[i, j])
            #     print("xm    ->", self.x_mass[i, j])
            #     print("ym    ->", self.y_mass[i, j])
            #     print("xi    ->", self.x_volume[i, j] / self.x_mass[i, j])
            #     print("yi    ->", self.y_volume[i, j] / self.y_mass[i, j])

            # Unraveled index.
            idx = (i * self.n_grid) + j

            # TODO: What about colliding cells? They are also empty, but the paper does not talk about this?
            # TODO: is this done on empty cells? Empty cells should evaluate to 0 pressure so this doesn't matter?
            D_s[idx, idx] += self.cell_inv_lambda[i, j] * self.cell_JP[i, j] * (1 / (self.cell_JE[i, j] * self.dt))

            b[idx] = -((self.cell_JE[i, j] - 1) / (self.dt * self.cell_JE[i, j]))
            if i != 0:
                b[idx] -= self.inv_dx * self.inv_dx * self.x_velocity[i - 1, j]
                b[idx] += self.inv_dx * self.inv_dx * self.x_velocity[i, j]
            if i != self.n_grid - 1:
                b[idx] -= self.inv_dx * self.inv_dx * self.x_velocity[i + 1, j]
                b[idx] += self.inv_dx * self.inv_dx * self.x_velocity[i, j]
            if j != 0:
                b[idx] -= self.inv_dx * self.inv_dx * self.y_velocity[i, j - 1]
                b[idx] += self.inv_dx * self.inv_dx * self.x_velocity[i, j]
            if j != self.n_grid - 1:
                b[idx] -= self.inv_dx * self.inv_dx * self.y_velocity[i, j + 1]
                b[idx] += self.inv_dx * self.inv_dx * self.x_velocity[i, j]

            # Build the Laplacian for the face density, apply homogeneous Neumann boundary condition to colliding faces.
            # TODO: or faces between colliding cells only?
            # +---+-t-+---+
            # |   |   |   |
            # +-l-+-c-+-r-+
            # |   |   |   |
            # +---+-b-+---+
            # |   |   |   |
            # +---+---+---+
            if i != 0:
                if self.x_classification[i - 1, j] != Classification.Colliding:
                    L_r[idx, idx - self.n_grid] -= self.x_volume[i - 1, j] / self.x_mass[i - 1, j]
                    L_r[idx, idx] += self.x_volume[i, j] / self.x_mass[i, j]
                elif i != self.n_grid - 1:  # Homogeneous Neumann boundary condition.
                    L_r[idx, idx - self.n_grid] -= self.x_volume[i + 1, j] / self.x_mass[i + 1, j]
                    L_r[idx, idx] += self.x_volume[i, j] / self.x_mass[i, j]
                    # b[idx - self.n_grid] = 0

            if i != self.n_grid - 1:
                if self.x_classification[i + 1, j] != Classification.Colliding:
                    L_r[idx, idx + self.n_grid] -= self.x_volume[i + 1, j] / self.x_mass[i + 1, j]
                    L_r[idx, idx] += self.x_volume[i, j] / self.x_mass[i, j]
                elif i != 0:  # Homogeneous Neumann boundary condition.
                    L_r[idx, idx + self.n_grid] -= self.x_volume[i - 1, j] / self.x_mass[i - 1, j]
                    L_r[idx, idx] += self.x_volume[i, j] / self.x_mass[i, j]
                    # b[idx + self.n_grid] = 0

            if j != 0:
                if self.y_classification[i, j - 1] != Classification.Colliding:
                    L_r[idx, idx - 1] -= self.x_volume[i, j - 1] / self.x_mass[i, j - 1]
                    L_r[idx, idx] += self.x_volume[i, j] / self.x_mass[i, j]
                elif j != self.n_grid - 1:  # Homogeneous Neumann boundary condition.
                    L_r[idx, idx - 1] -= self.x_volume[i, j + 1] / self.x_mass[i, j + 1]
                    L_r[idx, idx] += self.x_volume[i, j] / self.x_mass[i, j]
                    # b[idx - 1] = 0

            if j != self.n_grid - 1:
                if self.y_classification[i, j + 1] != Classification.Colliding:
                    L_r[idx, idx + 1] -= self.x_volume[i, j + 1] / self.x_mass[i, j + 1]
                    L_r[idx, idx] += self.x_volume[i, j] / self.x_mass[i, j]
                elif j != 0:  # Homogeneous Neumann boundary condition.
                    L_r[idx, idx + 1] -= self.x_volume[i, j - 1] / self.x_mass[i, j - 1]
                    L_r[idx, idx] += self.x_volume[i, j] / self.x_mass[i, j]
                    # b[idx + 1] = 0

            # Build the Laplacian for the cell pressure, apply homogeneous Dirichlet boundary condition to empty cells.
            # TODO: Colliding cells are also empty and need to be accounted for here?
            # if self.cell_classification[i, j] != Classification.Empty:

            # +---+---+---+ We build our 5-point-stencil for each cell by setting center value c to minus the number of
            # |   | t |   | non-colliding neighbor cells, entries for cells t, l, r, b are set to one if the corresponding
            # +---+---+---+ cell is interior. We will apply homogeneous Dirichlet boundary conditions to all cells c that
            # | l | c | r | are not interior itself.
            # +---+---+---+
            # |   | b |   |
            # +---+---+---+
            if self.c_classification[i, j] == Classification.Interior:
                if i != 0 and self.c_classification[i - 1, j] != Classification.Colliding:
                    L_p[idx, idx] -= 1.0
                    if self.c_classification[i - 1, j] == Classification.Interior:
                        L_p[idx, idx - self.n_grid] += 1.0

                if i != self.n_grid - 1 and self.c_classification[i + 1, j] != Classification.Colliding:
                    L_p[idx, idx] -= 1.0
                    if self.c_classification[i + 1, j] == Classification.Interior:
                        L_p[idx, idx + self.n_grid] += 1.0

                if j != 0 and self.c_classification[i, j - 1] != Classification.Colliding:
                    L_p[idx, idx] -= 1.0
                    if self.c_classification[i, j - 1] != Classification.Interior:
                        L_p[idx, idx - 1] += 1.0

                if j != self.n_grid - 1 and self.c_classification[i, j + 1] != Classification.Colliding:
                    L_p[idx, idx] -= 1.0
                    if self.c_classification[i, j + 1] == Classification.Interior:
                        L_p[idx, idx + 1] += 1.0
            else:  # Homogeneous Dirichlet boundary condition.
                L_p[idx, idx] += 1
                b[idx] = 0

            # print("~" * 100)
            # print("D_s[idx - 1, idx] ->", D_s[idx, idx - self.n_grid])
            # print("D_s[idx + 1, idx] ->", D_s[idx, idx + self.n_grid])
            # print("D_s[idx, idx - 1] ->", D_s[idx, idx - 1])
            # print("D_s[idx, idx + 1] ->", D_s[idx, idx + 1])
            # print("D_s[idx, idx]     ->", D_s[idx, idx])
            #
            # print("L_r[idx - 1, idx] ->", L_r[idx, idx - self.n_grid])
            # print("L_r[idx + 1, idx] ->", L_r[idx, idx + self.n_grid])
            # print("L_r[idx, idx - 1] ->", L_r[idx, idx - 1])
            # print("L_r[idx, idx + 1] ->", L_r[idx, idx + 1])
            # print("L_r[idx, idx]     ->", L_r[idx, idx])
            #
            # print("L_p[idx - 1, idx] ->", L_p[idx, idx - self.n_grid])
            # print("L_p[idx + 1, idx] ->", L_p[idx, idx + self.n_grid])
            # print("L_p[idx, idx - 1] ->", L_p[idx, idx - 1])
            # print("L_p[idx, idx + 1] ->", L_p[idx, idx + 1])
            # print("L_p[idx, idx]     ->", L_p[idx, idx])
            #
            # print("b[idx]            ->", b[idx])
            # print()

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
        z = self.dt * self.inv_dx * self.inv_dx  # Central Difference Coefficients

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

            self.x_velocity[i, j] += 4 * z * (self.x_volume[i, j] / self.x_mass[i, j]) * self.c_pressure[i, j]
            self.x_velocity[i, j] -= z * (self.x_volume[i + 1, j] / self.x_mass[i + 1, j]) * self.c_pressure[i + 1, j]
            self.x_velocity[i, j] -= z * (self.x_volume[i - 1, j] / self.x_mass[i - 1, j]) * self.c_pressure[i - 1, j]
            self.x_velocity[i, j] -= z * (self.x_volume[i, j + 1] / self.x_mass[i, j + 1]) * self.c_pressure[i, j + 1]
            self.x_velocity[i, j] -= z * (self.x_volume[i, j - 1] / self.x_mass[i, j - 1]) * self.c_pressure[i, j - 1]

        for i, j in ti.ndrange((left_boundary, right_boundary), (left_boundary, right_boundary + 1)):
            y_face_is_not_interior = self.c_classification[i, j - 1] != Classification.Interior
            y_face_is_not_interior |= self.c_classification[i, j] != Classification.Interior
            if y_face_is_not_interior:
                continue  # don't bother

            y_face_is_colliding = self.c_classification[i - 1, j] == Classification.Colliding
            y_face_is_colliding |= self.c_classification[i, j] == Classification.Colliding
            if y_face_is_colliding:
                self.y_velocity[i, j] = 0
                continue

            self.y_velocity[i, j] += 4 * z * (self.y_volume[i, j] / self.y_mass[i, j]) * self.c_pressure[i, j]
            self.y_velocity[i, j] -= z * (self.y_volume[i + 1, j] / self.y_mass[i + 1, j]) * self.c_pressure[i + 1, j]
            self.y_velocity[i, j] -= z * (self.y_volume[i - 1, j] / self.y_mass[i - 1, j]) * self.c_pressure[i - 1, j]
            self.y_velocity[i, j] -= z * (self.y_volume[i, j + 1] / self.y_mass[i, j + 1]) * self.c_pressure[i, j + 1]
            self.y_velocity[i, j] -= z * (self.y_volume[i, j - 1] / self.y_mass[i, j - 1]) * self.c_pressure[i, j - 1]

    def solve(self):
        # TODO: max_num_triplets could be optimized?
        # D_s = SparseMatrixBuilder(self.n_cells, self.n_cells, max_num_triplets=(self.n_cells))
        # L_r = SparseMatrixBuilder(self.n_cells, self.n_cells, max_num_triplets=(self.n_cells * 5))
        # L_p = SparseMatrixBuilder(self.n_cells, self.n_cells, max_num_triplets=(self.n_cells * self.n_cells))
        D_s = SparseMatrixBuilder(self.n_cells, self.n_cells, max_num_triplets=(self.n_cells * self.n_cells))
        L_r = SparseMatrixBuilder(self.n_cells, self.n_cells, max_num_triplets=(self.n_cells * self.n_cells))
        L_p = SparseMatrixBuilder(self.n_cells, self.n_cells, max_num_triplets=(self.n_cells * self.n_cells))
        b = ti.ndarray(ti.f32, shape=self.n_cells)
        self.fill_linear_system(D_s, L_r, L_p, b)
        D_s, L_r, L_p = D_s.build(), L_r.build(), L_p.build()
        # TODO: it might be more efficient to scale each entry directly?
        delta = 0.0000032  # relaxation
        # delta = 1.0  # relaxation
        coefficients = self.inv_dx * self.inv_dx
        R = (delta * D_s) + self.dt * ((coefficients * L_r) @ (delta * coefficients * L_p))
        print(R)

        # Solve the linear system.
        solver = SparseSolver()
        solver.compute(R)
        p = solver.solve(b)

        # FIXME: remove this debugging statements or move to test file
        solver_succeeded, pressure = solver.info(), p.to_numpy()
        assert solver_succeeded, "SOLVER DID NOT FIND A SOLUTION!"
        assert not np.any(np.isnan(pressure)), "NAN VALUE IN PRESSURE ARRAY!"

        # FIXME: Apply the pressure to the intermediate velocity field.
        # self.fill_pressure_field(p)
        # self.apply_pressure()
