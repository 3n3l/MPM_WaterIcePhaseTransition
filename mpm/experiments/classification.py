from operator import concat
import utils  # import first to append parent directory to path

from src.enums import Classification

import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)


class TextColor:
    Cyan = "\033[96m"
    Yellow = "\033[93m"
    Red = "\033[91m"
    End = "\033[0m"


def print_colored_text(text: str, color: str) -> str:
    return f"{color}{text}{TextColor.End}"


def print_cyan(text: str) -> str:
    return print_colored_text(text, TextColor.Cyan)


def print_yellow(text: str) -> str:
    return print_colored_text(text, TextColor.Yellow)


def print_red(text: str) -> str:
    return print_colored_text(text, TextColor.Red)


n_grid = 6
dx = 1 / n_grid
inv_dx = float(n_grid)
additional_offset = 0.5
boundary_width = 0

classification_c = ti.field(dtype=ti.int8, shape=(n_grid, n_grid))
classification_x = ti.field(dtype=ti.int8, shape=(n_grid + 1, n_grid))
classification_y = ti.field(dtype=ti.int8, shape=(n_grid, n_grid + 1))

mass_c = ti.field(dtype=ti.float32, shape=(n_grid, n_grid))
mass_x = ti.field(dtype=ti.float32, shape=(n_grid + 1, n_grid))
mass_y = ti.field(dtype=ti.float32, shape=(n_grid, n_grid + 1))

# Additional stagger for the grid and additional 0.5 to force flooring, used for the weight computations.
x_stagger = ti.Vector([(dx * 0.5) + 0.5, 0.5])
y_stagger = ti.Vector([0.5, (dx * 0.5) + 0.5])
c_stagger = ti.Vector([0.5, 0.5])

# Additional offsets for the grid, used for the distance (fx) computations.
x_offset = ti.Vector([(dx * 0.5), 0.0])
y_offset = ti.Vector([0.0, (dx * 0.5)])

rho_0 = 1000  # TODO: this is kg/m^3 for water, what about ice?
particle_vol = (dx * 0.5) ** 2
mass_p = particle_vol * rho_0


@ti.kernel
# def particle_to_grid(position_p: ti.template()):  # pyright: ignore
def particle_to_grid(x: float, y: float):  # pyright: ignore
    # We use an additional offset of 0.5 for element-wise flooring.
    position_p = ti.Vector([x, y])
    base_c = ti.floor((position_p * inv_dx - c_stagger), dtype=ti.i32)
    base_x = ti.floor((position_p * inv_dx - x_stagger), dtype=ti.i32)
    base_y = ti.floor((position_p * inv_dx - y_stagger), dtype=ti.i32)
    dist_c = position_p * inv_dx - ti.cast(base_c, ti.f32)
    dist_x = position_p * inv_dx - ti.cast(base_x, ti.f32) - x_offset
    dist_y = position_p * inv_dx - ti.cast(base_y, ti.f32) - y_offset

    print("BASE:", base_c)

    # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2)
    # w_c = [0.5 * (1.5 - dist_c) ** 2, 0.75 - (dist_c - 1) ** 2, 0.5 * (dist_c - 0.5) ** 2]
    # w_x = [0.5 * (1.5 - dist_x) ** 2, 0.75 - (dist_x - 1) ** 2, 0.5 * (dist_x - 0.5) ** 2]
    # w_y = [0.5 * (1.5 - dist_y) ** 2, 0.75 - (dist_y - 1) ** 2, 0.5 * (dist_y - 0.5) ** 2]

    # Cubic kernels (JST16 Eqn. 122 with x=fx, abs(fx-1), abs(fx-2))
    # Taken from https://github.com/taichi-dev/advanced_examples/blob/main/mpm/mpm99_cubic.py
    # TODO: calculate own weights for x=fx, fx-1, fx-2, or at least simplify these terms here?
    w_c = [
        0.5 * dist_c**3 - dist_c**2 + 2.0 / 3.0,
        0.5 * (-(dist_c - 1.0)) ** 3 - (-(dist_c - 1.0)) ** 2 + 2.0 / 3.0,
        1.0 / 6.0 * (2.0 + (dist_c - 2.0)) ** 3,
    ]
    w_x = [
        0.5 * dist_x**3 - dist_x**2 + 2.0 / 3.0,
        0.5 * (-(dist_x - 1.0)) ** 3 - (-(dist_x - 1.0)) ** 2 + 2.0 / 3.0,
        1.0 / 6.0 * (2.0 + (dist_x - 2.0)) ** 3,
    ]
    w_y = [
        0.5 * dist_y**3 - dist_y**2 + 2.0 / 3.0,
        0.5 * (-(dist_y - 1.0)) ** 3 - (-(dist_y - 1.0)) ** 2 + 2.0 / 3.0,
        1.0 / 6.0 * (2.0 + (dist_y - 2.0)) ** 3,
    ]

    for i, j in ti.static(ti.ndrange(3, 3)):
        offset = ti.Vector([i, j])
        weight_c = w_c[i][0] * w_c[j][1]
        weight_x = w_x[i][0] * w_x[j][1]
        weight_y = w_y[i][0] * w_y[j][1]

        print("BASE + OFFSET:", base_c + offset, "MASS:", weight_c * mass_p)

        # Rasterize mass to grid faces.
        mass_x[base_x + offset] += weight_x * mass_p
        mass_y[base_y + offset] += weight_y * mass_p

        # Rasterize mass to cell centers.
        mass_c[base_c + offset] += weight_c * mass_p


@ti.kernel
def classify_cells():
    for i, j in classification_x:
        # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.

        # The simulation boundary is always colliding.
        # x_face_is_colliding = i >= (n_grid - boundary_width) or i <= boundary_width
        # x_face_is_colliding |= j >= (n_grid - boundary_width) or j <= boundary_width
        # if x_face_is_colliding:
        #     classification_x[i, j] = Classification.Colliding
        #     continue

        # For convenience later on: a face is marked interior if it has mass.
        if mass_x[i, j] > 0:
            classification_x[i, j] = Classification.Interior
            continue

        # All remaining faces are empty.
        classification_x[i, j] = Classification.Empty

    for i, j in classification_y:
        # TODO: A MAC face is colliding if the level set computed by any collision object is negative at the face center.

        # The simulation boundary is always colliding.
        # y_face_is_colliding = i >= (n_grid - boundary_width) or i <= boundary_width
        # y_face_is_colliding |= j >= (n_grid - boundary_width) or j <= boundary_width
        # if y_face_is_colliding:
        #     classification_y[i, j] = Classification.Colliding
        #     continue

        # For convenience later on: a face is marked interior if it has mass.
        if mass_y[i, j] > 0:
            classification_y[i, j] = Classification.Interior
            continue

        # All remaining faces are empty.
        classification_y[i, j] = Classification.Empty

    for i, j in classification_c:
        # TODO: Colliding cells are either assigned the temperature of the object it collides with or a user-defined
        # spatially-varying value depending on the setup. If the free surface is being enforced as a Dirichlet
        # temperature condition, the ambient air temperature is recorded for empty cells. No other cells
        # require temperatures to be recorded at this stage.

        # A cell is marked as colliding if all of its surrounding faces are colliding.
        cell_is_colliding = classification_x[i, j] == Classification.Colliding
        cell_is_colliding &= classification_x[i + 1, j] == Classification.Colliding
        cell_is_colliding &= classification_y[i, j] == Classification.Colliding
        cell_is_colliding &= classification_y[i, j + 1] == Classification.Colliding
        if cell_is_colliding:
            # cell_temperature[i, j] = ambient_temperature[None]
            classification_c[i, j] = Classification.Colliding
            continue

        # A cell is interior if the cell and all of its surrounding faces have mass.
        cell_is_interior = mass_c[i, j] > 0
        cell_is_interior &= mass_x[i, j] > 0
        cell_is_interior &= mass_x[i + 1, j] > 0
        cell_is_interior &= mass_y[i, j] > 0
        cell_is_interior &= mass_y[i, j + 1] > 0
        if cell_is_interior:
            classification_c[i, j] = Classification.Interior
            continue

        # All remaining cells are empty.
        classification_c[i, j] = Classification.Empty

        # The ambient air temperature is recorded for empty cells.
        # temperature_c[i, j] = ambient_temperature[None]


def print_mass(mass: np.ndarray) -> None:
    nx, ny = mass.shape
    for i in range(nx):
        for j in range(ny):
            colorizer = print_yellow if mass[i, j] == 0 else print_red
            print(colorizer("%.1f" % mass[i, j]), end=" ")
        print()


def print_classification(classification: np.ndarray) -> None:
    cls_to_str = {
        Classification.Colliding: print_cyan("c"),
        Classification.Empty: print_yellow("x"),
        Classification.Interior: print_red("i"),
    }
    nx, ny = classification.shape
    for i in range(nx):
        for j in range(ny):
            print(cls_to_str[classification[i, j]], end="  ")
        print()


def main():
    # positions = [(0.0, 0.0), (0.1, 0.1), (0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (0.9, 0.9), (1.0, 1.0)]
    positions = [(0.55, 0.45)]
    for x, y in positions:
        print()
        particle_to_grid(x, y)
        classify_cells()

        # print(mass_x.to_numpy().T.shape)
        # print()
        # print(np.flip(mass_x.to_numpy(), 0).shape)
        # print()
        # print(np.flip(mass_x.to_numpy(), 1).shape)
        # print()
        # print(np.flip(mass_x.to_numpy(), None).shape)
        # continue

        # NOTE: ndarrays need to be flipped to represent what they would look like on the screen.
        # NOTE: ndarrays must also be transposed to represent what they would look like on the screen.
        # FIXME: surely both operations can be replaced with one that does both? 
        print("~" * 100)
        print("CELL:")
        print("\nMASS:")
        print_mass(np.flip(mass_c.to_numpy().T, 0))
        print("\nCLASSIFICATION:")
        print_classification(np.flip(classification_c.to_numpy().T, 0))
        print()

        print("~" * 100)
        print("X-FACE:")
        print("\nMASS:")
        print_mass(np.flip(mass_x.to_numpy().T, 0))
        print("\nCLASSIFICATION:")
        print_classification(np.flip(classification_x.to_numpy().T, 0))
        print()

        print("~" * 100)
        print("Y-FACE:")
        print("\nMASS:")
        print_mass(np.flip(mass_y.to_numpy().T, 0))
        print("\nCLASSIFICATION:")
        print_classification(np.flip(classification_y.to_numpy().T, 0))
        print()


if __name__ == "__main__":
    main()
