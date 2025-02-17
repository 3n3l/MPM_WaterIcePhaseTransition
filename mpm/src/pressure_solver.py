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

    @ti.func
    def sample(self, some_field, i_index, j_index):
        i = ti.max(0, ti.min(some_field.shape[0] - 1, int(i_index)))
        j = ti.max(0, ti.min(some_field.shape[1] - 1, int(j_index)))
        return some_field[i, j]

    @ti.kernel
    def fill_linear_system(
        self,
        D_s: ti.types.sparse_matrix_builder(),  # pyright: ignore
        L_r: ti.types.sparse_matrix_builder(),  # pyright: ignore
        L_p: ti.types.sparse_matrix_builder(),  # pyright: ignore
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

        # scale = self.dt * self.inv_dx * self.inv_dx
        z = self.dt * self.inv_dx * self.inv_dx

        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            # if self.c_classification[i, j] == Classification.Interior:
            #     print("xv    ->", self.x_volume[i, j])
            #     print("yv    ->", self.y_volume[i, j])
            #     print("xm    ->", self.x_mass[i, j])
            #     print("ym    ->", self.y_mass[i, j])
            #     print("xi    ->", self.x_volume[i, j] / self.x_mass[i, j])
            #     print("yi    ->", self.y_volume[i, j] / self.y_mass[i, j])

            # TODO: b: use central difference for face values (TODO: also for everything else???) or maybe not?
            #       MAC would use forward differences there, as this is supposed to simulate divergence the
            #       forward differences make a lot more sense???

            # Unraveled index.
            idx = (i * self.n_grid) + j

            # if self.cell_classification[i, j] != Classification.Interior:
            #     continue  # don't bother

            # Fill the left hand side of the linear system.
            # Fill the right hand side of the linear system.

            # TODO: is this done on empty cells? Empty cells should evaluate to 0 pressure so this doesn't matter?
            D_s[idx, idx] += self.cell_inv_lambda[i, j] * self.cell_JP[i, j] * (1 / (self.cell_JE[i, j] * self.dt))

            # TODO: What about colliding cells? They are also empty, but the paper does not talk about this?
            # if self.cell_classification[i, j] != Classification.Empty:
            # if self.cell_classification[i, j] == Classification.Interior:
            # TODO: this right here samples from neighboring cells/faces, but down below values outside the simulation are just ignored

            b[idx] = -((self.cell_JE[i, j] - 1) / (self.dt * self.cell_JE[i, j]))
            b[idx] -= 4 * z * self.x_velocity[i, j]
            b[idx] += z * self.sample(self.x_velocity, i - 1, j)
            b[idx] += z * self.sample(self.x_velocity, i + 1, j)
            b[idx] += z * self.sample(self.y_velocity, i, j - 1)
            b[idx] += z * self.sample(self.y_velocity, i, j + 1)

            # else:  # Homogeneous Dirichlet boundary condition on empty cells (TODO: atm this includes colliding cells).
            # b[idx] = 0
            # D_s[idx, idx] += 1.0
            # L_r[idx, idx] += 1.0
            # L_p[idx, idx] += 1.0
            # continue

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
                    L_r[idx, idx - self.n_grid] += self.x_volume[i - 1, j] / self.x_mass[i - 1, j]
                    L_r[idx, idx] -= self.x_volume[i, j] / self.x_mass[i, j]
                elif i != self.n_grid - 1:  # Homogeneous Neumann boundary condition.
                    L_r[idx, idx - self.n_grid] += self.x_volume[i + 1, j] / self.x_mass[i + 1, j]
                    L_r[idx, idx] -= self.x_volume[i, j] / self.x_mass[i, j]
                    b[idx - self.n_grid] = 0

            if i != self.n_grid - 1:
                if self.x_classification[i + 1, j] != Classification.Colliding:
                    L_r[idx, idx + self.n_grid] += self.x_volume[i + 1, j] / self.x_mass[i + 1, j]
                    L_r[idx, idx] -= self.x_volume[i, j] / self.x_mass[i, j]
                elif i != 0:  # Homogeneous Neumann boundary condition.
                    L_r[idx, idx + self.n_grid] += self.x_volume[i - 1, j] / self.x_mass[i - 1, j]
                    L_r[idx, idx] -= self.x_volume[i, j] / self.x_mass[i, j]
                    b[idx + self.n_grid] = 0

            if j != 0:
                if self.y_classification[i, j - 1] != Classification.Colliding:
                    L_r[idx, idx - 1] += self.x_volume[i, j - 1] / self.x_mass[i, j - 1]
                    L_r[idx, idx] -= self.x_volume[i, j] / self.x_mass[i, j]
                elif j != self.n_grid - 1:  # Homogeneous Neumann boundary condition.
                    L_r[idx, idx - 1] += self.x_volume[i, j + 1] / self.x_mass[i, j + 1]
                    L_r[idx, idx] -= self.x_volume[i, j] / self.x_mass[i, j]
                    b[idx - 1] = 0

            if j != self.n_grid - 1:
                if self.y_classification[i, j + 1] != Classification.Colliding:
                    L_r[idx, idx + 1] += self.x_volume[i, j + 1] / self.x_mass[i, j + 1]
                    L_r[idx, idx] -= self.x_volume[i, j] / self.x_mass[i, j]
                elif j != 0:  # Homogeneous Neumann boundary condition.
                    L_r[idx, idx + 1] += self.x_volume[i, j - 1] / self.x_mass[i, j - 1]
                    L_r[idx, idx] -= self.x_volume[i, j] / self.x_mass[i, j]
                    b[idx + 1] = 0


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
                    L_p[idx, idx + self.n_grid] += 1.0
                    if self.c_classification[i + 1, j] == Classification.Interior:
                        L_p[idx, idx] -= 1.0

                if j != 0 and self.c_classification[i, j - 1] != Classification.Colliding:
                    L_p[idx, idx - 1] += 1.0
                    if self.c_classification[i, j - 1] != Classification.Interior:
                        L_p[idx, idx] -= 1.0

                if j != self.n_grid - 1 and self.c_classification[i, j + 1] != Classification.Colliding:
                    L_p[idx, idx + 1] += 1.0
                    if self.c_classification[i, j + 1] == Classification.Interior:
                        L_p[idx, idx] -= 1.0
            else:  # Homogeneous Dirichlet boundary condition.
                L_p[idx, idx] += 1
                b[idx] = 0

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

            self.x_velocity[i, j] -= 4 * z * (self.x_volume[i, j] / self.x_mass[i, j]) * self.c_pressure[i, j]
            self.x_velocity[i, j] += z * (self.x_volume[i + 1, j] / self.x_mass[i + 1, j]) * self.c_pressure[i + 1, j]
            self.x_velocity[i, j] += z * (self.x_volume[i - 1, j] / self.x_mass[i - 1, j]) * self.c_pressure[i - 1, j]
            self.x_velocity[i, j] += z * (self.x_volume[i, j + 1] / self.x_mass[i, j + 1]) * self.c_pressure[i, j + 1]
            self.x_velocity[i, j] += z * (self.x_volume[i, j - 1] / self.x_mass[i, j - 1]) * self.c_pressure[i, j - 1]

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

            self.y_velocity[i, j] -= 4 * z * (self.y_volume[i, j] / self.y_mass[i, j]) * self.c_pressure[i, j]
            self.y_velocity[i, j] += z * (self.y_volume[i + 1, j] / self.y_mass[i + 1, j]) * self.c_pressure[i + 1, j]
            self.y_velocity[i, j] += z * (self.y_volume[i - 1, j] / self.y_mass[i - 1, j]) * self.c_pressure[i - 1, j]
            self.y_velocity[i, j] += z * (self.y_volume[i, j + 1] / self.y_mass[i, j + 1]) * self.c_pressure[i, j + 1]
            self.y_velocity[i, j] += z * (self.y_volume[i, j - 1] / self.y_mass[i, j - 1]) * self.c_pressure[i, j - 1]

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
        R = D_s + ((self.dt * self.inv_dx * self.inv_dx * L_r) @ (self.inv_dx * self.inv_dx * L_p))

        # Solve the linear system.
        solver = SparseSolver()
        solver.compute(R)
        p = solver.solve(b)

        # FIXME: remove this debugging statements
        solver_succeeded = solver.info()
        assert solver_succeeded, "SOLVER DID NOT FIND A SOLUTION!"
        # pressure = p.to_numpy()
        # print("P ->", np.min(pressure), np.max(pressure))

        # Apply the pressure to the intermediate velocity field.
        self.fill_pressure_field(p)
        self.apply_pressure()
