from taichi.linalg import SparseMatrixBuilder, SparseSolver, SparseCG
from src.parsing import should_use_direct_solver
from src.constants import Classification

import taichi as ti
import numpy as np


@ti.data_oriented
class HeatSolver:
    def __init__(self, mpm_solver) -> None:
        self.n_cells = mpm_solver.n_grid * mpm_solver.n_grid
        self.inv_dx = mpm_solver.inv_dx
        self.n_grid = mpm_solver.n_grid
        self.mpm_solver = mpm_solver
        self.dx = mpm_solver.dx
        self.dt = mpm_solver.dt

        self.classification_c = mpm_solver.classification_c
        self.temperature_c = mpm_solver.temperature_c
        self.capacity_c = mpm_solver.capacity_c
        self.mass_c = mpm_solver.mass_c

        self.classification_x = mpm_solver.classification_x
        self.classification_y = mpm_solver.classification_y
        self.conductivity_x = mpm_solver.conductivity_x
        self.conductivity_y = mpm_solver.conductivity_y

        self.should_use_direct_solver = should_use_direct_solver

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
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), T: ti.types.ndarray()):  # pyright: ignore
        for i, j in self.temperature_c:
            # Raveled index.
            idx = (i * self.n_grid) + j

            # Set right-hand side to the cell temperature
            T[idx] = self.temperature_c[i, j]

            center = 0.0

            # We enforce Dirichlet temperature boundary conditions at CELLS that are in contact
            # with fixed temperature bodies (like a heated pan (-> colliding cells) or air (-> empty cells)).
            if self.is_interior(i, j):
                inv_mass_capacity = 1 / (self.mass_c[i, j] * self.capacity_c[i, j])
                center += 1.0

                # We enforce homogeneous Neumann boundary conditions at FACES adjacent to
                # cells that can be considered empty or corresponding to insulated objects.
                # NOTE: dx^d is cancelled out by 1 / dx^2 because d == 2.
                if not self.is_empty(i - 1, j):
                    center -= self.dt * inv_mass_capacity * self.conductivity_x[i, j]
                    if self.is_interior(i - 1, j):
                        A[idx, idx - self.n_grid] += self.dt * inv_mass_capacity * self.conductivity_x[i, j]
                else:  # Neumann (homogeneous)
                    T[idx] = self.mpm_solver.ambient_temperature[None]

                if not self.is_empty(i + 1, j):
                    center -= self.dt * inv_mass_capacity * self.conductivity_x[i + 1, j]
                    if self.is_interior(i + 1, j):
                        A[idx, idx + self.n_grid] += self.dt * inv_mass_capacity * self.conductivity_x[i + 1, j]
                else:  # Neumann (homogeneous)
                    T[idx] = self.mpm_solver.ambient_temperature[None]

                if not self.is_empty(i, j - 1):
                    center -= self.dt * inv_mass_capacity * self.conductivity_y[i, j]
                    if self.is_interior(i, j - 1):
                        A[idx, idx - 1] += self.dt * inv_mass_capacity * self.conductivity_y[i, j]
                else:  # Neumann (homogeneous)
                    T[idx] = self.mpm_solver.ambient_temperature[None]

                if not self.is_empty(i, j + 1):
                    center -= self.dt * inv_mass_capacity * self.conductivity_y[i, j + 1]
                    if self.is_interior(i, j + 1):
                        A[idx, idx + 1] += self.dt * inv_mass_capacity * self.conductivity_y[i, j + 1]
                else:  # Neumann (homogeneous)
                    T[idx] = self.mpm_solver.ambient_temperature[None]

                A[idx, idx] += center
            else:  # Dirichlet (not homogeneous)
                A[idx, idx] += 1.0

    @ti.kernel
    def fill_temperature_field(self, T: ti.types.ndarray()):  # pyright: ignore
        for i, j in self.temperature_c:
            self.temperature_c[i, j] = T[(i * self.n_grid) + j]

    def solve(self):
        # TODO: max_num_triplets could be optimized to N * 5?
        A = SparseMatrixBuilder(
            max_num_triplets=(5 * self.n_cells),
            num_rows=self.n_cells,
            num_cols=self.n_cells,
            dtype=ti.f32,
        )
        b = ti.ndarray(ti.f32, shape=self.n_cells)
        self.fill_linear_system(A, b)

        # Solve the linear system.
        if self.should_use_direct_solver:
            solver = SparseSolver(dtype=ti.f32, solver_type="LLT")
            solver.compute(A.build())
            T = solver.solve(b)
            # FIXME: remove this debugging statements or move to test file
            # solver_succeeded, temperature = solver.info(), T.to_numpy()
            # assert solver_succeeded, "SOLVER DID NOT FIND A SOLUTION!"
            # assert not np.any(np.isnan(temperature)), "NAN VALUE IN NEW TEMPERATURE ARRAY!"
        else:
            solver = SparseCG(A.build(), b, atol=1e-6, max_iter=500)
            T, _ = solver.solve()

        self.fill_temperature_field(T)
