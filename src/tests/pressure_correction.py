import taichi as ti

ti.init(arch=ti.cpu)


class PressureSolver:
    def __init__(self) -> None:
        pass

    @ti.func
    def sample(self, x, i, j):
        indices = ti.Vector([int(i), int(j)])
        indices = ti.max(0, ti.min(self.n_grid - 1, indices))
        return x[indices]

    @ti.kernel
    def compute_divergence(self):
        for i, j in self.cell_divergence:
            self.cell_divergence[i, j] = 0
            self.cell_divergence[i, j] += self.face_velocity_x[i + 1, j]
            self.cell_divergence[i, j] += self.face_velocity_y[i, j + 1]
            self.cell_divergence[i, j] -= self.face_velocity_x[i, j]
            self.cell_divergence[i, j] -= self.face_velocity_y[i, j]

    @ti.kernel
    def fill_left_hand_side(self, p: ti.template(), Ap: ti.template()):  # pyright: ignore
        delta = 0.00032  # relaxation
        inv_rho = 1 / self.rho_0  # TODO: compute rho per cell
        z = self.dt * self.inv_dx * self.inv_dx * inv_rho
        for i, j in ti.ndrange((1, self.n_grid - 1), (1, self.n_grid - 1)):
            # for i, j in self.cell_pressure:
            Ap[i, j] = 0
            if self.cell_classification[i, j] == Classification.Interior:
                l = p[i - 1, j]
                r = p[i + 1, j]
                t = p[i, j + 1]
                b = p[i, j - 1]

                # TODO: collect face volumes here first

                # FIXME: the error is somewhere here:
                # Ap[i, j] = delta * p[i, j] * self.cell_inv_lambda[i, j]
                # Ap[i, j] *= self.cell_JP[i, j] * (1 / (self.cell_JE[i, j] * self.dt))

                Ap[i, j] += z * delta * (4.0 * p[i, j] - l - r - t - b)
                self.Ap[i, j] = Ap[i, j]  # FIXME: for testing only

    @ti.kernel
    def fill_right_hand_side(self):
        z = self.inv_dx
        for i, j in ti.ndrange((1, self.n_grid), (1, self.n_grid)):
            # for i, j in self.cell_pressure:
            self.right_hand_side[i, j] = 0
            if self.cell_classification[i, j] == Classification.Interior:

                # FIXME: the error is somewhere here:
                # self.right_hand_side[i, j] = -1 * (self.cell_JE[i, j] - 1) / (self.dt * self.cell_JE[i, j])

                self.right_hand_side[i, j] += z * (self.face_velocity_x[i + 1, j] - self.face_velocity_x[i, j])
                self.right_hand_side[i, j] += z * (self.face_velocity_y[i, j + 1] - self.face_velocity_y[i, j])

            if self.cell_classification[i, j] == Classification.Interior:
                print("-" * 100)
                print("PRESSU ->", self.cell_pressure[i, j])
                print("CELLJE ->", self.cell_JE[i, j])
                print("CELLJP ->", self.cell_JP[i, j])
                print("X_VELO ->", self.face_velocity_x[i, j])
                print("Y_VELO ->", self.face_velocity_y[i, j])
                print("DTTTTT ->", self.dt)
                print("LAMBDA ->", self.cell_inv_lambda[i, j])
                print("BBBBBB ->", self.right_hand_side[i, j])
                print("AAAAAA ->", self.Ap[i, j])

    def solve_pressure(self):
        self.fill_right_hand_side()
        self.cell_pressure.fill(0)
        MatrixFreeCG(
            A=self.left_hand_side,
            b=self.right_hand_side,
            x=self.cell_pressure,
            maxiter=100,
            tol=1e-5,
            quiet=False,
        )

    @ti.kernel
    def apply_pressure(self):
        inv_rho = 1 / self.rho_0  # TODO: compute inv_rho from face volumes per cell
        z = self.dt * inv_rho * self.inv_dx * self.inv_dx
        for i, j in self.face_mass_x:
            # if self.cell_classification[i - 1, j] == Classification.Interior:
            self.face_velocity_x[i, j] -= z * self.sample(self.cell_pressure, i - 1, j)
            # if self.cell_classification[i, j] == Classification.Interior:
            self.face_velocity_x[i, j] -= z * self.sample(self.cell_pressure, i, j)
            # self.face_velocity_x[i, j] -= z * self.sample(self.cell_pressure, i, j + 1)
            # self.face_velocity_x[i, j] -= z * self.sample(self.cell_pressure, i, j - 1)
            # self.face_velocity_x[i, j] -= 4 * z * self.cell_pressure[i, j]
            # self.face_velocity_x[i, j] -= 4 * z * self.cell_pressure[i, j]
        for i, j in self.face_mass_y:
            # self.face_velocity_y[i, j] -= z * self.sample(self.cell_pressure, i - 1, j)
            # self.face_velocity_y[i, j] -= z * self.sample(self.cell_pressure, i + 1, j)
            # if self.cell_classification[i, j - 1] == Classification.Interior:
            self.face_velocity_y[i, j] -= z * self.sample(self.cell_pressure, i, j - 1)
            # if self.cell_classification[i, j] == Classification.Interior:
            self.face_velocity_y[i, j] -= z * self.sample(self.cell_pressure, i, j)
            # self.face_velocity_y[i, j] -= 4 * z * self.cell_pressure[i, j]
            # self.face_velocity_y[i, j] -= 4 * z * self.cell_pressure[i, j]

# @ti.kernel
# def classify_cells(self):
#     # We can extract the offset coordinates from the faces by adding one to the respective axis,
#     # e.g. we get the two x-faces with [i, j] and [i + 1, j], where each cell looks like:
#     # -  ^  -
#     # >  *  >
#     # -  ^  -
#     for i, j in self.cell_classification:
#         # TODO: A cell is marked as colliding if all of its surrounding faces are colliding.
#         # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.
#         # TODO: Colliding cells are either assigned the temperature of the object it collides with or a user-defined
#         # spatially-varying value depending on the setup. If the free surface is being enforced as a Dirichlet
#         # temperature condition, the ambient air temperature is recorded for empty cells. No other cells
#         # require temperatures to be recorded at this stage.
#         is_colliding = False
#
#         # A cell is interior if the cell and all of its surrounding faces have mass.
#         is_interior = self.cell_mass[i, j] > 0
#         is_interior &= self.face_mass_x[i, j] > 0
#         is_interior &= self.face_mass_y[i, j] > 0
#         is_interior &= self.face_mass_x[i + 1, j] > 0
#         is_interior &= self.face_mass_y[i, j + 1] > 0
#
#         if is_colliding:
#             self.cell_classification[i, j] = Classification.Colliding
#         elif is_interior:
#             self.cell_classification[i, j] = Classification.Interior
#         else:
#             self.cell_classification[i, j] = Classification.Empty
#
# @ti.func
# def sample(self, x, i, j):
#     indices = ti.Vector([int(i), int(j)])
#     indices = ti.max(0, ti.min(self.n_grid - 1, indices))
#     return x[indices]
#
# @ti.kernel
# def compute_divergence(self):
#     for i, j in self.cell_divergence:
#         self.cell_divergence[i, j] = 0
#         self.cell_divergence[i, j] += self.face_velocity_x[i + 1, j]
#         self.cell_divergence[i, j] += self.face_velocity_y[i, j + 1]
#         self.cell_divergence[i, j] -= self.face_velocity_x[i, j]
#         self.cell_divergence[i, j] -= self.face_velocity_y[i, j]
#
# @ti.kernel
# def fill_left_hand_side(self, p: ti.template(), Ap: ti.template()):  # pyright: ignore
#     delta = 0.00032  # relaxation
#     inv_rho = 1 / self.rho_0  # TODO: compute rho per cell
#     z = self.dt * self.inv_dx * self.inv_dx * inv_rho
#     for i, j in ti.ndrange((1, self.n_grid - 1), (1, self.n_grid - 1)):
#         # for i, j in self.cell_pressure:
#         Ap[i, j] = 0
#         if self.cell_classification[i, j] == Classification.Interior:
#             l = p[i - 1, j]
#             r = p[i + 1, j]
#             t = p[i, j + 1]
#             b = p[i, j - 1]
#
#             # TODO: collect face volumes here first
#
#             # FIXME: the error is somewhere here:
#             # Ap[i, j] = delta * p[i, j] * self.cell_inv_lambda[i, j]
#             # Ap[i, j] *= self.cell_JP[i, j] * (1 / (self.cell_JE[i, j] * self.dt))
#
#             Ap[i, j] += z * delta * (4.0 * p[i, j] - l - r - t - b)
#             self.Ap[i, j] = Ap[i, j]  # FIXME: for testing only
#
# @ti.kernel
# def fill_right_hand_side(self):
#     z = self.inv_dx
#     for i, j in ti.ndrange((1, self.n_grid), (1, self.n_grid)):
#         # for i, j in self.cell_pressure:
#         self.right_hand_side[i, j] = 0
#         if self.cell_classification[i, j] == Classification.Interior:
#
#             # FIXME: the error is somewhere here:
#             # self.right_hand_side[i, j] = -1 * (self.cell_JE[i, j] - 1) / (self.dt * self.cell_JE[i, j])
#
#             self.right_hand_side[i, j] += z * (self.face_velocity_x[i + 1, j] - self.face_velocity_x[i, j])
#             self.right_hand_side[i, j] += z * (self.face_velocity_y[i, j + 1] - self.face_velocity_y[i, j])
#
#         if self.cell_classification[i, j] == Classification.Interior:
#             print("-" * 100)
#             print("PRESSU ->", self.cell_pressure[i, j])
#             print("CELLJE ->", self.cell_JE[i, j])
#             print("CELLJP ->", self.cell_JP[i, j])
#             print("X_VELO ->", self.face_velocity_x[i, j])
#             print("Y_VELO ->", self.face_velocity_y[i, j])
#             print("DTTTTT ->", self.dt)
#             print("LAMBDA ->", self.cell_inv_lambda[i, j])
#             print("BBBBBB ->", self.right_hand_side[i, j])
#             print("AAAAAA ->", self.Ap[i, j])
#
# def solve_pressure(self):
#     self.fill_right_hand_side()
#     self.cell_pressure.fill(0)
#     MatrixFreeCG(
#         A=self.left_hand_side,
#         b=self.right_hand_side,
#         x=self.cell_pressure,
#         maxiter=100,
#         tol=1e-5,
#         quiet=False,
#     )
#
# @ti.kernel
# def apply_pressure(self):
#     inv_rho = 1 / self.rho_0  # TODO: compute inv_rho from face volumes per cell
#     z = self.dt * inv_rho * self.inv_dx * self.inv_dx
#     for i, j in self.face_mass_x:
#         # if self.cell_classification[i - 1, j] == Classification.Interior:
#         self.face_velocity_x[i, j] -= z * self.sample(self.cell_pressure, i - 1, j)
#         # if self.cell_classification[i, j] == Classification.Interior:
#         self.face_velocity_x[i, j] -= z * self.sample(self.cell_pressure, i, j)
#         # self.face_velocity_x[i, j] -= z * self.sample(self.cell_pressure, i, j + 1)
#         # self.face_velocity_x[i, j] -= z * self.sample(self.cell_pressure, i, j - 1)
#         # self.face_velocity_x[i, j] -= 4 * z * self.cell_pressure[i, j]
#         # self.face_velocity_x[i, j] -= 4 * z * self.cell_pressure[i, j]
#     for i, j in self.face_mass_y:
#         # self.face_velocity_y[i, j] -= z * self.sample(self.cell_pressure, i - 1, j)
#         # self.face_velocity_y[i, j] -= z * self.sample(self.cell_pressure, i + 1, j)
#         # if self.cell_classification[i, j - 1] == Classification.Interior:
#         self.face_velocity_y[i, j] -= z * self.sample(self.cell_pressure, i, j - 1)
#         # if self.cell_classification[i, j] == Classification.Interior:
#         self.face_velocity_y[i, j] -= z * self.sample(self.cell_pressure, i, j)
#         # self.face_velocity_y[i, j] -= 4 * z * self.cell_pressure[i, j]
#         # self.face_velocity_y[i, j] -= 4 * z * self.cell_pressure[i, j]
