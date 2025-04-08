from taichi.linalg import SparseMatrixBuilder, SparseSolver, SparseCG
from src.enums import Classification

import taichi as ti
import numpy as np


@ti.data_oriented
class HeatSolver:
    def __init__(self, mpm_solver, should_use_direct_solver: bool = True) -> None:
        self.n_cells = mpm_solver.n_grid * mpm_solver.n_grid
        self.inv_dx = mpm_solver.inv_dx
        self.n_grid = mpm_solver.n_grid
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

    @ti.kernel
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), T: ti.types.ndarray()):  # pyright: ignore
        for i, j in self.temperature_c:
            # Raveled index.
            idx = (i * self.n_grid) + j

            # Set right-hand side to the cell temperature
            T[idx] = self.temperature_c[i, j]

            # FIXME: these variables are just used to print everything and can be removed after debugging
            A_t = 0.0
            A_l = 0.0
            A_c = 0.0
            A_r = 0.0
            A_b = 0.0

            # We enforce Dirichlet temperature boundary conditions at CELLS that are in contact
            # with fixed temperature bodies (like a heated pan (-> colliding cells) or air (-> empty cells)).
            if self.classification_c[i, j] == Classification.Interior:
                inv_mass_capacity = 1 / (self.mass_c[i, j] * self.capacity_c[i, j])
                A_c += 1.0

                # We enforce homogeneous Neumann boundary conditions at FACES adjacent to
                # cells that can be considered empty or corresponding to insulated objects.
                # NOTE: dx^d is cancelled out by 1 / dx^2 because d == 2.
                if i != 0 and self.classification_c[i - 1, j] != Classification.Empty:
                    A[idx, idx - self.n_grid] -= self.dt * inv_mass_capacity * self.conductivity_x[i, j]
                    A_l -= self.dt * inv_mass_capacity * self.conductivity_x[i, j]
                    A_c += self.dt * inv_mass_capacity * self.conductivity_x[i, j]

                if i != self.n_grid - 1 and self.classification_c[i + 1, j] != Classification.Empty:
                    A[idx, idx + self.n_grid] -= self.dt * inv_mass_capacity * self.conductivity_x[i + 1, j]
                    A_r -= self.dt * inv_mass_capacity * self.conductivity_x[i + 1, j]
                    A_c += self.dt * inv_mass_capacity * self.conductivity_x[i + 1, j]

                if j != 0 and self.classification_c[i, j - 1] != Classification.Empty:
                    A[idx, idx - 1] -= self.dt * inv_mass_capacity * self.conductivity_y[i, j]
                    A_b -= self.dt * inv_mass_capacity * self.conductivity_y[i, j]
                    A_c += self.dt * inv_mass_capacity * self.conductivity_y[i, j]

                if j != self.n_grid - 1 and self.classification_c[i, j + 1] != Classification.Empty:
                    A[idx, idx + 1] -= self.dt * inv_mass_capacity * self.conductivity_y[i, j + 1]
                    A_t -= self.dt * inv_mass_capacity * self.conductivity_y[i, j + 1]
                    A_c += self.dt * inv_mass_capacity * self.conductivity_y[i, j + 1]

                A[idx, idx] += A_c
            else:  # Dirichlet boundary condition (not homogeneous)
                A[idx, idx] += 1.0
                A_c = 1.0

            continue
            if self.classification_c[i, j] != Classification.Interior:
                continue
            # if not (cell_is_interior and not_adjacent_to_fixed_temperature):
            # if (T[idx] < 0 or T[idx] > 0 or T[idx] == 0):
            #     continue
            print("~" * 100)
            print()
            if self.classification_c[i, j] == Classification.Interior:
                print(f">>> INTERIOR, idx = {idx}, i = {i}, j = {j}")
            elif self.classification_c[i, j] == Classification.Colliding:
                print(f">>> COLLIDING, idx = {idx}, i = {i}, j = {j}")
            else:
                print(f">>> EMPTY, idx = {idx}, i = {i}, j = {j}")

            if i != 0:
                print("cell i - 1 is empty->", self.classification_c[i - 1, j] != Classification.Empty)
            if i != self.n_grid - 1:
                print("cell i + 1 is empty->", self.classification_c[i + 1, j] != Classification.Empty)
            if j != 0:
                print("cell i - 1 is empty->", self.classification_c[i, j - 1] != Classification.Empty)
            if j != self.n_grid - 1:
                print("cell i + 1 is empty->", self.classification_c[i, j + 1] != Classification.Empty)

            print()
            print(f"A[{idx}, {idx} + 1]   ->", A_t)
            print(f"A[{idx} - 1, {idx}]   ->", A_l)
            print(f"A[{idx}, {idx}]       ->", A_c)
            print(f"A[{idx} + 1, {idx}]   ->", A_r)
            print(f"A[{idx}, {idx} - 1]   ->", A_b)

            print()
            print("m_c                    ->", self.mass_c[i, j])
            print("c_c                    ->", self.capacity_c[i, j])
            print("1 / m_c * c_c          ->", 1 / (self.mass_c[i, j] * self.capacity_c[i, j]))

            print()
            print("x_conductivity[i, j]   ->", self.conductivity_x[i, j])
            if i != self.n_grid - 1:
                print("x_conductivity[i + 1, j] ->", self.conductivity_x[i + 1, j])

            print()
            print("y_conductivity[i, j]   ->", self.conductivity_y[i, j])
            if j != self.n_grid - 1:
                print("y_conductivity[i, j + 1] ->", self.conductivity_y[i, j + 1])

            print()
            print(f"b[{idx}]                ->", T[idx])
            print()

    @ti.kernel
    def fill_temperature_field(self, T: ti.types.ndarray()):  # pyright: ignore
        for i, j in self.temperature_c:
            self.temperature_c[i, j] = T[(i * self.n_grid) + j]

    def solve(self):
        # TODO: max_num_triplets could be optimized to N * 5?
        A = SparseMatrixBuilder(
            max_num_triplets=(self.n_cells * self.n_cells),
            # max_num_triplets=(5 * self.n_cells),
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
            solver_succeeded, temperature = solver.info(), T.to_numpy()
            assert solver_succeeded, "SOLVER DID NOT FIND A SOLUTION!"
            assert not np.any(np.isnan(temperature)), "NAN VALUE IN NEW TEMPERATURE ARRAY!"
        else:
            solver = SparseCG(A.build(), b)
            T, _ = solver.solve()

        self.fill_temperature_field(T)
