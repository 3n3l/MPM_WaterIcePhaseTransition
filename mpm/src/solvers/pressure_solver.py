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
        self.rho = mpm_solver.rho_0
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

    # @ti.kernel
    # def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
    #     inv_dx_squared = self.inv_dx * self.inv_dx
    #
    #     for i, j in ti.ndrange(self.n_grid, self.n_grid):
    #         # Raveled index.
    #         idx = (i * self.n_grid) + j
    #
    #         # FIXME: these variables are just used to print everything and can be removed after debugging
    #         A_t = 0.0
    #         A_l = 0.0
    #         A_c = 0.0
    #         A_r = 0.0
    #         A_b = 0.0
    #
    #         # FIXME: this should be on non empty cells, but then the colliding
    #         #        simulation boundary results in underdetermined linear system
    #
    #         # if self.classification_c[i, j] != Classification.Empty:
    #         if self.classification_c[i, j] == Classification.Interior:
    #             A_c += self.JP_c[i, j] / (self.dt * self.JE_c[i, j]) * self.inv_lambda_c[i, j]
    #
    #             # Build the right-hand side of the linear system.
    #             # b[idx] = (1 - self.JE_c[i, j]) / (self.dt * self.JE_c[i, j])
    #             b[idx] = -((self.JE_c[i, j] - 1) / (self.dt * self.JE_c[i, j]))
    #             b[idx] -= self.inv_dx * (self.velocity_x[i + 1, j] - self.velocity_x[i, j])
    #             b[idx] -= self.inv_dx * (self.velocity_y[i, j + 1] - self.velocity_y[i, j])
    #
    #             # We will apply a Neumann boundary condition on the colliding faces,
    #             # to guarantee zero flux into colliding cells, by just not adding these
    #             # face values in the Laplacian for the off-diagonal values.
    #             if i != 0 and self.classification_c[i - 1, j] != Classification.Colliding:
    #                 inv_rho = self.volume_x[i, j] / self.mass_x[i, j]
    #                 if self.classification_c[i - 1, j] == Classification.Colliding:
    #                     print("inv_rho =", inv_rho)
    #                 A_c -= self.dt * inv_dx_squared * inv_rho
    #                 # if self.classification_c[i - 1, j] != Classification.Empty:
    #                 if self.classification_c[i - 1, j] == Classification.Interior:
    #                     A[idx, idx - self.n_grid] += self.dt * inv_dx_squared * inv_rho
    #
    #             if i != self.n_grid - 1 and self.classification_c[i + 1, j] != Classification.Colliding:
    #                 inv_rho = self.volume_x[i + 1, j] / self.mass_x[i + 1, j]
    #                 A_c -= self.dt * inv_dx_squared * inv_rho
    #                 # if self.classification_c[i + 1, j] != Classification.Empty:
    #                 if self.classification_c[i + 1, j] == Classification.Interior:
    #                     A[idx, idx + self.n_grid] += self.dt * inv_dx_squared * inv_rho
    #
    #             if j != 0 and self.classification_c[i, j - 1] != Classification.Colliding:
    #                 inv_rho = self.volume_y[i, j] / self.mass_y[i, j]
    #                 A_c -= self.dt * inv_dx_squared * inv_rho
    #                 # if self.classification_c[i, j - 1] != Classification.Empty:
    #                 if self.classification_c[i, j - 1] == Classification.Interior:
    #                     A[idx, idx - 1] += self.dt * inv_dx_squared * inv_rho
    #
    #             if j != self.n_grid - 1 and self.classification_c[i, j + 1] != Classification.Colliding:
    #                 inv_rho = self.volume_y[i, j + 1] / self.mass_y[i, j + 1]
    #                 A_c -= self.dt * inv_dx_squared * inv_rho
    #                 # if self.classification_c[i, j + 1] != Classification.Empty:
    #                 if self.classification_c[i, j + 1] == Classification.Interior:
    #                     A[idx, idx + 1] += self.dt * inv_dx_squared * inv_rho
    #
    #             A[idx, idx] += A_c
    #
    #         else:  # Homogeneous Dirichlet boundary condition.
    #             A[idx, idx] += 1.0
    #             b[idx] = 0.0
    #             A_c += 1.0
    #
    #         continue
    #         # if self.classification_c[i, j] != Classification.Colliding:
    #         #     continue
    #         if self.classification_c[i, j] != Classification.Interior:
    #             continue
    #         # if self.classification_c[i, j] != Classification.Empty:
    #         #     continue
    #         print("~" * 100)
    #         print()
    #         if self.classification_c[i, j] == Classification.Interior:
    #             print(f">>> INTERIOR, idx = {idx}, i = {i}, j = {j}")
    #         elif self.classification_c[i, j] == Classification.Colliding:
    #             print(f">>> COLLIDING, idx = {idx}, i = {i}, j = {j}")
    #         else:
    #             print(f">>> EMPTY, idx = {idx}, i = {i}, j = {j}")
    #
    #         print()
    #         print(f"A[{idx}, {idx} + 1]   ->", A_t)
    #         print(f"A[{idx} - 1, {idx}]   ->", A_l)
    #         print(f"A[{idx}, {idx}]       ->", A_c)
    #         print(f"A[{idx} + 1, {idx}]   ->", A_r)
    #         print(f"A[{idx}, {idx} - 1]   ->", A_b)
    #
    #         print()
    #         print(f"velocity_x[i, j]      ->", self.velocity_x[i, j])
    #         print(f"velocity_x[i + 1, j]  ->", self.velocity_x[i + 1, j])
    #         print(f"velocity_x[i - 1, j]  ->", self.velocity_x[i - 1, j])
    #
    #         print()
    #         print(f"velocity_y[i, j]      ->", self.velocity_y[i, j])
    #         print(f"velocity_y[i, j + 1]  ->", self.velocity_y[i, j + 1])
    #         print(f"velocity_y[i, j - 1]  ->", self.velocity_y[i, j - 1])
    #
    #         print()
    #         print(f"JE_c[i, j]            ->", self.JE_c[i, j])
    #         print(f"JP_c[i, j]            ->", self.JP_c[i, j])
    #         print(f"inv_lambda_c[i, j]    ->", self.inv_lambda_c[i, j])
    #         print(f"1 / inv_lambda_c[i, j]->", 1.0 / self.inv_lambda_c[i, j])
    #
    #         print()
    #         print(f"b[{idx}]                ->", b[idx])
    #         print()

    @ti.kernel
    def fill_linear_system(self, A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray()):  # pyright: ignore
        coefficient = self.rho * self.dx / self.dt
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            center_entry = 0.0  # to keep max_num_triplets as low as possible
            idx = (i * self.n_grid) + j  # raveled index
            if self.is_interior(i, j):
                # Build the right-hand side of the linear system.
                # This uses a modified divergence, where the velocities of faces
                # bordering colliding (solid) cells are considered to be zero.
                if not self.is_colliding(i + 1, j):
                    # rho = self.mass_x[i + 1, j] / self.volume_x[i + 1, j]
                    # z = rho * self.dx * self.inv_dt
                    b[idx] += coefficient * self.velocity_x[i + 1, j]
                if not self.is_colliding(i - 1, j):
                    # rho = self.mass_x[i, j] / self.volume_x[i, j]
                    # z = rho * self.dx * self.inv_dt
                    b[idx] -= coefficient * self.velocity_x[i, j]
                if not self.is_colliding(i, j + 1):
                    # rho = self.mass_y[i, j + 1] / self.volume_y[i, j + 1]
                    # z = rho * self.dx * self.inv_dt
                    b[idx] += coefficient * self.velocity_y[i, j + 1]
                if not self.is_colliding(i, j - 1):
                    # rho = self.mass_y[i, j] / self.volume_y[i, j]
                    # z = rho * self.dx * self.inv_dt
                    b[idx] -= coefficient * self.velocity_y[i, j]

                # We will apply a Neumann boundary condition on the colliding faces,
                # to guarantee zero flux into colliding cells, by just not adding these
                # face values in the Laplacian for the off-diagonal values.
                # NOTE: we can use the raveled index to quickly access adjacent cells with:
                # idx(i, j) = (i * n) + j
                #   => idx(i - 1, j) = ((i - 1) * n) + j = (i * n) + j - n = idx(i, j) - n
                #   => idx(i, j - 1) = (i * n) + j - 1 = idx(i, j) - 1, etc.
                if not self.is_colliding(i - 1, j):
                    center_entry -= 1.0
                    if not self.is_empty(i - 1, j):
                        A[idx, idx - self.n_grid] += 1.0

                if not self.is_colliding(i + 1, j):
                    center_entry -= 1.0
                    if not self.is_empty(i + 1, j):
                        A[idx, idx + self.n_grid] += 1.0

                if not self.is_colliding(i, j - 1):
                    center_entry -= 1.0
                    if not self.is_empty(i, j - 1):
                        A[idx, idx - 1] += 1.0

                if not self.is_colliding(i, j + 1):
                    center_entry -= 1.0
                    if not self.is_empty(i, j + 1):
                        A[idx, idx + 1] += 1.0

                A[idx, idx] += center_entry

            else:  # Homogeneous Dirichlet boundary condition.
                A[idx, idx] += 1.0
                b[idx] = 0.0

    @ti.kernel
    def apply_pressure(self, pressure: ti.types.ndarray()):  # pyright: ignore
        coefficient = self.dt / (self.dx * self.rho)
        # coefficient = self.inv_dx * self.dt
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            idx = i * self.n_grid + j
            if self.is_interior(i - 1, j) or self.is_interior(i, j):
                if not (self.is_colliding(i - 1, j) or self.is_colliding(i, j)):
                    inv_rho = 1  # self.volume_x[i, j] / self.mass_x[i, j]
                    pressure_gradient = inv_rho * (pressure[idx] - pressure[idx - self.n_grid])
                    self.velocity_x[i, j] -= coefficient * pressure_gradient
            if self.is_interior(i, j - 1) or self.is_interior(i, j):
                if not (self.is_colliding(i, j - 1) or self.is_colliding(i, j)):
                    inv_rho = 1  # self.volume_y[i, j] / self.mass_y[i, j]
                    pressure_gradient = inv_rho * (pressure[idx] - pressure[idx - 1])
                    self.velocity_y[i, j] -= coefficient * pressure_gradient

    @ti.kernel
    def calculate_pressure(self):
        """Calculate the pressure according to the constitutive model."""
        for i, j in self.pressure_c:
            self.pressure_c[i, j] = -(1 / self.inv_lambda_c[i, j]) * (self.JE_c[i, j] - 1)

    @ti.kernel
    def compare_pressure(self):
        """Compare the solved for pressure to the pressure according to the constitutive model."""
        for i, j in self.pressure_c:
            if self.classification_c[i, j] == Classification.Interior:
                pressure = -(1 / self.inv_lambda_c[i, j]) * (self.JE_c[i, j] - 1)
                print(f"INTERIOR  -> solved = {self.pressure_c[i, j]}, constitutive = {pressure}")
            # elif self.classification_c[i,j] == Classification.Colliding:
            #     pressure = -(1 / self.inv_lambda_c[i, j]) * (self.JE_c[i, j] - 1)
            #     print(f"COLLIDING -> solved = {self.pressure_c[i, j]}, constitutive = {pressure}")

    def solve(self):
        A = SparseMatrixBuilder(
            # TODO: max_num_triplets could be optimized to N * 5?
            max_num_triplets=(10 * self.n_cells),
            num_rows=self.n_cells,
            num_cols=self.n_cells,
            dtype=ti.f32,
        )
        b = ti.ndarray(ti.f32, shape=self.n_cells)
        self.fill_linear_system(A, b)

        # Solve the linear system:
        if self.should_use_direct_solver:
            solver = SparseSolver(dtype=ti.f32, solver_type="LLT")
            solver.compute(A.build())
            p = solver.solve(b)
            # FIXME: remove this debugging statements or move to test file
            solver_succeeded, pressure = solver.info(), p.to_numpy()
            assert solver_succeeded, "SOLVER DID NOT FIND A SOLUTION!"
            assert not np.any(np.isnan(pressure)), "NAN VALUE IN PRESSURE ARRAY!"
        else:
            solver = SparseCG(A.build(), b, max_iter=100)
            p, _ = solver.solve()
            # print("min, max =", np.min(p), np.max(p))

        # Compare constitutive and solved pressure:
        # self.compare_pressure()

        # Use explicit pressure:
        self.calculate_pressure()  # Uncomment to use constitutive pressure

        # Correct pressure:
        # self.fill_pressure_field(p)  # Uncomment to use solved pressure
        self.apply_pressure(p)
