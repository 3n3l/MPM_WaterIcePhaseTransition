from taichi.linalg import SparseMatrixBuilder, SparseSolver, SparseCG
from src.parsing import should_use_direct_solver
from src.constants import Classification

import taichi as ti
import numpy as np

GRAVITY = -9.81


@ti.data_oriented
class PressureSolver:
    def __init__(self, mpm_solver) -> None:
        self.n_cells = mpm_solver.n_grid * mpm_solver.n_grid
        self.boundary_width = mpm_solver.boundary_width
        self.inv_dx = mpm_solver.inv_dx
        self.n_grid = mpm_solver.n_grid
        self.dt = mpm_solver.dt

        self.classification_c = mpm_solver.classification_c
        self.inv_lambda_c = mpm_solver.inv_lambda_c
        self.JE_c = mpm_solver.JE_c
        self.JP_c = mpm_solver.JP_c

        self.velocity_c = mpm_solver.velocity_c
        self.volume_c = mpm_solver.volume_c
        self.mass_c = mpm_solver.mass_c

    @ti.func
    def is_valid(self, i: int, j: int) -> bool:
        return i >= 0 and i <= self.n_grid - 1 and j >= 0 and j <= self.n_grid - 1

    @ti.func
    def is_colliding(self, i: int, j: int) -> bool:
        return self.is_valid(i, j) and self.classification_c[i, j] == Classification.Colliding

    @ti.func
    def is_interior(self, i: int, j: int) -> bool:
        return self.is_valid(i, j) and self.classification_c[i, j] == Classification.Interior

    @ti.func
    def is_empty(self, i: int, j: int) -> bool:
        return self.is_valid(i, j) and self.classification_c[i, j] == Classification.Empty

    @ti.kernel
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
        coefficient = self.dt * self.inv_dx * self.inv_dx
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            center = 0.0  # to keep max_num_triplets as low as possible
            idx = (i * self.n_grid) + j  # raveled index

            if self.is_interior(i, j):
                # Build the right-hand side of the linear system:
                # FIXME: this pushes the solids apart :(
                # b[idx] = (1 - self.JE_c[i, j]) / (self.dt * self.JE_c[i, j])

                # This uses a modified divergence, where the velocities of faces
                # bordering colliding (solid) cells are considered to be zero.
                # NOTE: we subtract the divergence, instead of adding it.
                if not self.is_colliding(i + 1, j):
                    b[idx] -= self.inv_dx * self.velocity_c[i + 1, j][0]
                if not self.is_colliding(i - 1, j):
                    b[idx] += self.inv_dx * self.velocity_c[i, j][0]
                if not self.is_colliding(i, j + 1):
                    b[idx] -= self.inv_dx * self.velocity_c[i, j + 1][1]
                if not self.is_colliding(i, j - 1):
                    b[idx] += self.inv_dx * self.velocity_c[i, j][1]

                # Build the left-hand side of the linear system:
                # FIXME: this here breaks everything :(
                # center += (self.JP_c[i, j] / (self.dt * self.JE_c[i, j])) / self.inv_lambda_c[i, j]

                # We will apply a Neumann boundary condition on the colliding faces,
                # to guarantee zero flux into colliding cells, by just not adding these
                # face values in the Laplacian for the off-diagonal values.
                if not self.is_colliding(i - 1, j):
                    inv_rho = 1 / 1000  # self.volume_x[i, j] / self.mass_x[i, j]
                    center -= coefficient * inv_rho
                    if self.is_interior(i - 1, j):
                        A[idx, idx - self.n_grid] += coefficient * inv_rho

                if not self.is_colliding(i + 1, j):
                    inv_rho = 1 / 1000  # self.volume_x[i + 1, j] / self.mass_x[i + 1, j]
                    center -= coefficient * inv_rho
                    if self.is_interior(i + 1, j):
                        A[idx, idx + self.n_grid] += coefficient * inv_rho

                if not self.is_colliding(i, j - 1):
                    inv_rho = 1 / 1000  # self.volume_y[i, j] / self.mass_y[i, j]
                    center -= coefficient * inv_rho
                    if self.is_interior(i, j - 1):
                        A[idx, idx - 1] += coefficient * inv_rho

                if not self.is_colliding(i, j + 1):
                    inv_rho = 1 / 1000  # self.volume_y[i, j + 1] / self.mass_y[i, j + 1]
                    center -= coefficient * inv_rho
                    if self.is_interior(i, j + 1):
                        A[idx, idx + 1] += coefficient * inv_rho

                A[idx, idx] += center

            else:  # Homogeneous Dirichlet boundary condition.
                A[idx, idx] += 1.0
                b[idx] = 0.0

    @ti.kernel
    def apply_pressure(self, pressure: ti.types.ndarray()):  # pyright: ignore
        coefficient = self.dt * self.inv_dx
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            idx = i * self.n_grid + j
            if self.is_interior(i - 1, j) or self.is_interior(i, j):
                # NOTE: we add the pressure, instead of subtracting it.
                if not (self.is_colliding(i - 1, j) or self.is_colliding(i, j)):
                    pressure_gradient = pressure[idx] - pressure[idx - self.n_grid]
                    inv_rho = 1 / 1000  # self.volume_x[i, j] / self.mass_x[i, j]
                    self.velocity_c[i, j][0] += inv_rho * coefficient * pressure_gradient
                else:
                    self.velocity_c[i, j][0] = 0
            if self.is_interior(i, j - 1) or self.is_interior(i, j):
                if not (self.is_colliding(i, j - 1) or self.is_colliding(i, j)):
                    pressure_gradient = pressure[idx] - pressure[idx - 1]
                    inv_rho = 1 / 1000  # self.volume_y[i, j] / self.mass_y[i, j]
                    self.velocity_c[i, j][1] += inv_rho * coefficient * pressure_gradient
                else:
                    self.velocity_c[i, j][1] = 0

    def solve(self):
        A = SparseMatrixBuilder(
            max_num_triplets=(5 * self.n_cells),
            num_rows=self.n_cells,
            num_cols=self.n_cells,
            dtype=ti.f32,
        )
        b = ti.ndarray(ti.f32, shape=self.n_cells)
        self.fill_linear_system(A, b)

        # Solve the linear system:
        if should_use_direct_solver:
            solver = SparseSolver(dtype=ti.f32, solver_type="LLT")
            solver.compute(A.build())
            p = solver.solve(b)
            # FIXME: remove this debugging statements or move to test file
            solver_succeeded, pressure = solver.info(), p.to_numpy()
            assert solver_succeeded, "SOLVER DID NOT FIND A SOLUTION!"
            assert not np.any(np.isnan(pressure)), "NAN VALUE IN PRESSURE ARRAY!"
        else:
            solver = SparseCG(A.build(), b, atol=1e-6, max_iter=500)
            p, _ = solver.solve()

        # Correct pressure:
        self.apply_pressure(p)
