from enums import Classification

import taichi as ti

ti.init(arch=ti.cpu, debug=True)
# ti.init(arch=ti.cuda, debug=True)


@ti.data_oriented
class PCG:
    def __init__(self, n_grid: int) -> None:
        self.n_grid = n_grid

        # self.fill_laplacian_matrix(K, classification)

    @ti.kernel
    def fill_laplacian_matrix(
        self,
        A: ti.types.sparse_matrix_builder(),  # pyright: ignore
        classification: ti.template(),  # pyright: ignore
    ):
        for i, j in ti.ndrange(self.n_grid, self.n_grid):
            row = (i * self.n_grid) + j
            center = 0.0
            if (j != 0) and (classification[i, j - 1] == Classification.Interior):
                A[row, row - 1] += -1.0
                center += 1.0
            if (j != self.n_grid - 1) and (classification[i, j + 1] == Classification.Interior):
                A[row, row + 1] += -1.0
                center += 1.0
            if (i != 0) and (classification[i - 1, j] == Classification.Interior):
                A[row, row - self.n_grid] += -1.0
                center += 1.0
            if (i != self.n_grid - 1) and (classification[i + 1, j] == Classification.Interior):
                A[row, row + self.n_grid] += -1.0
                center += 1.0
            A[row, row] += center

    @ti.kernel
    def fill_right_side(self, b: ti.template()):  # pyright: ignore
        z = self.inv_dx
        for i, j in ti.ndrange((1, self.n_grid), (1, self.n_grid)):
            # for i, j in self.cell_pressure:
            b[i, j] = 0
            if self.cell_classification[i, j] == Classification.Interior:

                # FIXME: the error is somewhere here:
                # b[i, j] = -1 * (self.cell_JE[i, j] - 1) / (self.dt * self.cell_JE[i, j])

                b[i, j] += z * (self.face_velocity_x[i + 1, j] - self.face_velocity_x[i, j])
                b[i, j] += z * (self.face_velocity_y[i, j + 1] - self.face_velocity_y[i, j])

            if self.cell_classification[i, j] == Classification.Interior:
                print("-" * 100)
                print("PRESSU ->", self.cell_pressure[i, j])
                print("CELLJE ->", self.cell_JE[i, j])
                print("CELLJP ->", self.cell_JP[i, j])
                print("X_VELO ->", self.face_velocity_x[i, j])
                print("Y_VELO ->", self.face_velocity_y[i, j])
                print("DTTTTT ->", self.dt)
                print("LAMBDA ->", self.cell_inv_lambda[i, j])
                print("BBBBBB ->", b[i, j])
                print("AAAAAA ->", self.Ap[i, j])
