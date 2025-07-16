import utils  # import first to append parent directory to path

from src.constants import Classification, State

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


n_grid = 8
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

# To log which faces contributed to the particle velocity in G2P
contributed_c = ti.field(dtype=ti.float32, shape=(n_grid, n_grid))
contributed_x = ti.field(dtype=ti.float32, shape=(n_grid + 1, n_grid))
contributed_y = ti.field(dtype=ti.float32, shape=(n_grid, n_grid + 1))

rho_0 = 1000  # TODO: this is kg/m^3 for water, what about ice?
particle_vol = (dx * 0.5) ** 2
mass_p = particle_vol * rho_0


@ti.func
def compute_cubic_weight(x: float) -> float:
    w = 0.0

    if 0 <= ti.abs(x) < 1:
        w = (0.5 * ti.abs(x) ** 3) - (x**2) + 0.67
    elif 1 <= ti.abs(x) < 2:
        w = -0.167 * (ti.abs(x) ** 3) + (x**2) - (2 * ti.abs(x)) + 1.34
        # w = 0.167 * ((2 - ti.abs(x)) ** 3)

    return w


@ti.kernel
def particle_to_grid(position_p: ti.template(), should_use_cubic: bool):  # pyright: ignore
    if should_use_cubic:
        # Lower left corner of the interpolation grid:
        # base_x = ti.floor((position_p * inv_dx - ti.Vector([1.0, 1.5])), dtype=ti.i32)
        # base_y = ti.floor((position_p * inv_dx - ti.Vector([1.5, 1.0])), dtype=ti.i32)
        # base_c = ti.floor((position_p * inv_dx - ti.Vector([1.5, 1.5])), dtype=ti.i32)

        base_x = ti.floor((position_p * inv_dx - ti.Vector([0.0, 0.5])), dtype=ti.i32)
        base_y = ti.floor((position_p * inv_dx - ti.Vector([0.5, 0.0])), dtype=ti.i32)
        base_c = ti.floor((position_p * inv_dx - ti.Vector([0.0, 0.0])), dtype=ti.i32)

        # Distance between lower left corner and particle position:
        dist_x = position_p * inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.0, 0.5])
        dist_y = position_p * inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.5, 0.0])
        dist_c = position_p * inv_dx - ti.cast(base_c, ti.f32)

        # # Cubic kernels (JST16 Eqn. 122 with x=fx, x=|fx-1|, x=|fx-2|, x=|fx-3|, where fx is the distance
        # # between base node and particle position). Based on https://www.bilibili.com/opus/662560355423092789
        # # TODO: this could be shortened to x=fx, fx-1, fx-2, fx+1?!
        # w_c = [
        #     ((-0.166 * dist_c**3) + (dist_c**2) - (2 * dist_c) + 1.33),
        #     ((0.5 * ti.abs(dist_c - 1.0) ** 3) - ((dist_c - 1.0) ** 2) + 0.66),
        #     ((0.5 * ti.abs(dist_c - 2.0) ** 3) - ((dist_c - 2.0) ** 2) + 0.66),
        #     ((-0.166 * ti.abs(dist_c - 3.0) ** 3) + ((dist_c - 3.0) ** 2) - (2 * ti.abs(dist_c - 3.0)) + 1.33),
        # ]
        # w_x = [
        #     ((-0.166 * dist_x**3) + (dist_x**2) - (2 * dist_x) + 1.33),
        #     ((0.5 * ti.abs(dist_x - 1.0) ** 3) - ((dist_x - 1.0) ** 2) + 0.66),
        #     ((0.5 * ti.abs(dist_x - 2.0) ** 3) - ((dist_x - 2.0) ** 2) + 0.66),
        #     ((-0.166 * ti.abs(dist_x - 3.0) ** 3) + ((dist_x - 3.0) ** 2) - (2 * ti.abs(dist_x - 3.0)) + 1.33),
        # ]
        # w_y = [
        #     ((-0.166 * dist_y**3) + (dist_y**2) - (2 * dist_y) + 1.33),
        #     ((0.5 * ti.abs(dist_y - 1.0) ** 3) - ((dist_y - 1.0) ** 2) + 0.66),
        #     ((0.5 * ti.abs(dist_y - 2.0) ** 3) - ((dist_y - 2.0) ** 2) + 0.66),
        #     ((-0.166 * ti.abs(dist_y - 3.0) ** 3) + ((dist_y - 3.0) ** 2) - (2 * ti.abs(dist_y - 3.0)) + 1.33),
        # ]

        # w_x = [
        #     1.0 / 6.0 * (2.0 - (dist_x + 1)) ** 3,
        #     0.5 * dist_x**3 - dist_x**2 + 2.0 / 3.0,
        #     0.5 * (-(dist_x - 1.0)) ** 3 - (-(dist_x - 1.0)) ** 2 + 2.0 / 3.0,
        #     1.0 / 6.0 * (2.0 + (dist_x - 2.0)) ** 3,
        # ]
        # w_y = [
        #     1.0 / 6.0 * (2.0 - (dist_y + 1)) ** 3,
        #     0.5 * dist_y**3 - dist_y**2 + 2.0 / 3.0,
        #     0.5 * (-(dist_y - 1.0)) ** 3 - (-(dist_y - 1.0)) ** 2 + 2.0 / 3.0,
        #     1.0 / 6.0 * (2.0 + (dist_y - 2.0)) ** 3,
        # ]
        # w_c = [
        #     1.0 / 6.0 * (2.0 - (dist_c + 1)) ** 3,
        #     0.5 * dist_c**3 - dist_c**2 + 2.0 / 3.0,
        #     0.5 * (-(dist_c - 1.0)) ** 3 - (-(dist_c - 1.0)) ** 2 + 2.0 / 3.0,
        #     1.0 / 6.0 * (2.0 + (dist_c - 2.0)) ** 3,
        # ]

        w_x = [
            ((-0.166 * (dist_x + 1)**3) + ((dist_x + 1)**2) - (2 * (dist_x + 1)) + 1.33),
            ((0.5 * dist_x ** 3) - (dist_x**2) + 0.66),
            ((0.5 * ti.abs(dist_x - 1.0) ** 3) - ((dist_x - 1.0) ** 2) + 0.66),
            ((-0.166 * ti.abs(dist_x - 2.0) ** 3) + ((dist_x - 2.0) ** 2) - (2 * ti.abs(dist_x - 2.0)) + 1.33),
        ]
        w_y = [
            ((-0.166 * (dist_y + 1)**3) + ((dist_y + 1)**2) - (2 * (dist_y + 1)) + 1.33),
            ((0.5 * dist_y ** 3) - (dist_y**2) + 0.66),
            ((0.5 * ti.abs(dist_y - 1.0) ** 3) - ((dist_y - 1.0) ** 2) + 0.66),
            ((-0.166 * ti.abs(dist_y - 2.0) ** 3) + ((dist_y - 2.0) ** 2) - (2 * ti.abs(dist_y - 2.0)) + 1.33),
        ]
        w_c = [
            ((-0.166 * (dist_c + 1)**3) + ((dist_c + 1)**2) - (2 * (dist_c + 1)) + 1.33),
            ((0.5 * dist_c ** 3) - (dist_c**2) + 0.66),
            ((0.5 * ti.abs(dist_c - 1.0) ** 3) - ((dist_c - 1.0) ** 2) + 0.66),
            ((-0.166 * ti.abs(dist_c - 2.0) ** 3) + ((dist_c - 2.0) ** 2) - (2 * ti.abs(dist_c - 2.0)) + 1.33),
        ]

        # print(f"base_c = {base_c}")
        # print(f"base_x = {base_x}")
        # print(f"base_y = {base_y}")

        # print()
        # print(f"dist_c     = {dist_c}")
        # print(f"dist_c - 1 = {ti.abs(dist_c - 1)}")
        # print(f"dist_c - 2 = {ti.abs(dist_c - 2)}")
        # print(f"dist_c - 3 = {ti.abs(dist_c - 3)}")

        # print()
        # print(f"dist_x     = {dist_x}")
        # print(f"dist_x - 1 = {ti.abs(dist_x - 1)}")
        # print(f"dist_x - 2 = {ti.abs(dist_x - 2)}")
        # print(f"dist_x - 3 = {ti.abs(dist_x - 3)}")

        # print()
        # print(f"dist_y     = {dist_y}")
        # print(f"dist_y - 1 = {ti.abs(dist_y - 1)}")
        # print(f"dist_y - 2 = {ti.abs(dist_y - 2)}")
        # print(f"dist_y - 3 = {ti.abs(dist_y - 3)}")

        # for i, j in ti.static(ti.ndrange((-1, 3), (-1, 3))):
        for i, j in ti.static(ti.ndrange(4, 4)):
            offset = ti.Vector([i, j]) - 1
            weight_x = w_x[i][0] * w_x[j][1]
            weight_y = w_y[i][0] * w_y[j][1]
            weight_c = w_c[i][0] * w_c[j][1]

            # This looks good and symmetrical for position_p = [0.5, 0.5]
            # weight_c = compute_cubic_weight(dist_c[0] - i) * compute_cubic_weight(dist_c[1] - j)  # pyright: ignore

            # Use precomputed weights:
            mass_x[base_x + offset] += weight_x * mass_p
            mass_y[base_y + offset] += weight_y * mass_p
            mass_c[base_c + offset] += weight_c * mass_p

            # NOTE: check against actual computation if this is correct:
            # _weight_x = compute_cubic_weight(dist_x[0] - i) * compute_cubic_weight(dist_x[1] - j)  # pyright: ignore
            # _weight_y = compute_cubic_weight(dist_y[0] - i) * compute_cubic_weight(dist_y[1] - j)  # pyright: ignore
            # _weight_c = compute_cubic_weight(dist_c[0] - i) * compute_cubic_weight(dist_c[1] - j)  # pyright: ignore
            # print(f"weight_x: {weight_x} <- {_weight_x}, eq: {weight_x == _weight_x}")
            # print(f"weight_y: {weight_y} <- {_weight_y}, eq: {weight_y == _weight_y}")
            # print(f"weight_c: {weight_c} <- {_weight_c}, eq: {weight_c == _weight_c}")
            # mass_x[base_x + offset] += _weight_x * mass_p
            # mass_y[base_y + offset] += _weight_y * mass_p
            # mass_c[base_c + offset] += _weight_c * mass_p
    else:
        base_x = ti.floor((position_p * inv_dx - ti.Vector([1.0, 0.5])), dtype=ti.i32)
        base_y = ti.floor((position_p * inv_dx - ti.Vector([0.5, 0.5])), dtype=ti.i32)
        base_c = ti.floor((position_p * inv_dx - ti.Vector([0.5, 0.5])), dtype=ti.i32)

        dist_x = position_p * inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.0, 0.5])
        dist_y = position_p * inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.5, 0.0])
        dist_c = position_p * inv_dx - ti.cast(base_c, ti.f32)

        w_x = [0.5 * (1.5 - dist_x) ** 2, 0.75 - (dist_x - 1) ** 2, 0.5 * (dist_x - 0.5) ** 2]
        w_y = [0.5 * (1.5 - dist_y) ** 2, 0.75 - (dist_y - 1) ** 2, 0.5 * (dist_y - 0.5) ** 2]
        w_c = [0.5 * (1.5 - dist_c) ** 2, 0.75 - (dist_c - 1) ** 2, 0.5 * (dist_c - 0.5) ** 2]

        # TODO: should this be unified with an if-statement?
        #       this might be more efficient, but ugly
        for i, j in ti.static(ti.ndrange(3, 3)):
            # offset = ti.Vector([i, j]) - 1
            offset = ti.Vector([i, j])
            weight_x = w_x[i][0] * w_x[j][1]
            weight_y = w_y[i][0] * w_y[j][1]
            weight_c = w_c[i][0] * w_c[j][1]

            # Use precomputed weights:
            mass_x[base_x + offset] += weight_x * mass_p
            mass_y[base_y + offset] += weight_y * mass_p
            mass_c[base_c + offset] += weight_c * mass_p

            # NOTE: check against actual computation if this is correct:
            # _weight_x = compute_cubic_weight(dist_x[0] - i) * compute_cubic_weight(dist_x[1] - j)  # pyright: ignore
            # _weight_y = compute_cubic_weight(dist_y[0] - i) * compute_cubic_weight(dist_y[1] - j)  # pyright: ignore
            # _weight_c = compute_cubic_weight(dist_c[0] - i) * compute_cubic_weight(dist_c[1] - j)  # pyright: ignore
            # print(f"weight_x: {weight_x} <- {_weight_x}, eq: {weight_x == _weight_x}")
            # print(f"weight_y: {weight_y} <- {_weight_y}, eq: {weight_y == _weight_y}")
            # print(f"weight_c: {weight_c} <- {_weight_c}, eq: {weight_c == _weight_c}")
            # mass_x[base_x + offset] += _weight_x * mass_p
            # mass_y[base_y + offset] += _weight_y * mass_p
            # mass_c[base_c + offset] += _weight_c * mass_p


@ti.func
def is_valid(i: int, j: int) -> bool:
    return i >= 0 and i <= n_grid - 1 and j >= 0 and j <= n_grid - 1


@ti.func
def is_colliding(i: int, j: int) -> bool:
    return is_valid(i, j) and classification_c[i, j] == Classification.Colliding


@ti.kernel
def _classify_cells(x: float, y: float):
    for i, j in classification_c:
        # Reset all the cells that don't belong to the colliding boundary:
        if not is_colliding(i, j):
            classification_c[i, j] = Classification.Empty

    # Find the nearest cell and set it to interior:
    # FIXME: melting here only works with rounding, but this introduces assymetry
    i, j = ti.floor(ti.Vector([x, y]) * inv_dx, dtype=ti.i32)  # pyright: ignore
    if not is_colliding(i, j):  # pyright: ignore
        classification_c[i, j] = Classification.Interior


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


@ti.kernel
def grid_to_particle(position_p: ti.template()):  # pyright: ignore
    base_x = ti.floor((position_p * inv_dx - ti.Vector([0.5, 1.0])), dtype=ti.i32)
    base_y = ti.floor((position_p * inv_dx - ti.Vector([1.0, 0.5])), dtype=ti.i32)
    base_c = ti.floor((position_p * inv_dx - ti.Vector([0.5, 0.5])), dtype=ti.i32)

    dist_x = position_p * inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.0, 0.5])
    dist_y = position_p * inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.5, 0.0])
    dist_c = position_p * inv_dx - ti.cast(base_c, ti.f32)

    # Quadratic kernels (JST16, Eqn. 123, with x=fx, fx-1, fx-2)
    w_c = [0.5 * (1.5 - dist_c) ** 2, 0.75 - (dist_c - 1) ** 2, 0.5 * (dist_c - 0.5) ** 2]
    w_x = [0.5 * (1.5 - dist_x) ** 2, 0.75 - (dist_x - 1) ** 2, 0.5 * (dist_x - 0.5) ** 2]
    w_y = [0.5 * (1.5 - dist_y) ** 2, 0.75 - (dist_y - 1) ** 2, 0.5 * (dist_y - 0.5) ** 2]

    for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
        offset = ti.Vector([i, j])
        c_weight = w_c[i][0] * w_c[j][1]
        x_weight = w_x[i][0] * w_x[j][1]
        y_weight = w_y[i][0] * w_y[j][1]
        contributed_c[base_x + offset] += c_weight
        contributed_x[base_x + offset] += x_weight
        contributed_y[base_y + offset] += y_weight


def main():
    # positions = [(0.0, 0.0), (0.1, 0.1), (0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (0.9, 0.9), (1.0, 1.0)]
    # positions = [(0.32, 0.52), (0.52, 0.32), (0.52, 0.52)]
    # positions = [(0.52, 0.52)]
    # positions = [(0.48, 0.48)]
    # positions = [(0.5, 0.5)]
    positions = [(0.5, 0.5), (0.52, 0.52), (0.48, 0.48)]

    for x, y in positions:
        position_p = ti.Vector([x, y])
        classification_c.fill(Classification.Empty)
        classification_x.fill(Classification.Empty)
        classification_y.fill(Classification.Empty)

        mass_c.fill(0.0)
        mass_x.fill(0.0)
        mass_y.fill(0.0)

        contributed_c.fill(0.0)
        contributed_x.fill(0.0)
        contributed_y.fill(0.0)

        print()
        particle_to_grid(position_p, True)
        classify_cells()
        grid_to_particle(position_p)

        # NOTE: ndarrays need to be flipped to represent what they would look like on the screen.
        # NOTE: ndarrays must also be transposed to represent what they would look like on the screen.
        # TODO: surely both operations can be replaced with one that does both?
        print("~" * 100)
        print(f"-> CUBIC @ {position_p}:")
        # print("CELL:")
        # print("\nMASS:")
        print_mass(np.flip(mass_c.to_numpy().T, 0))
        # print("\nCLASSIFICATION:")
        # print_classification(np.flip(classification_c.to_numpy().T, 0))
        # print("\nCONTRIBUTION G2P:")
        # print_mass(np.flip(contributed_c.to_numpy().T, 0))
        # print()

        # print("~" * 100)
        # print("X-FACE:")
        # print("\nMASS:")
        # print_mass(np.flip(mass_x.to_numpy().T, 0))
        # print("\nCLASSIFICATION:")
        # print_classification(np.flip(classification_x.to_numpy().T, 0))
        # print("\nCONTRIBUTION G2P:")
        # print_mass(np.flip(contributed_x.to_numpy().T, 0))
        # print()

        # print("~" * 100)
        # print("Y-FACE:")
        # print("\nMASS:")
        # print_mass(np.flip(mass_y.to_numpy().T, 0))
        # print("\nCLASSIFICATION:")
        # print_classification(np.flip(classification_y.to_numpy().T, 0))
        # print("\nCONTRIBUTION G2P:")
        # print_mass(np.flip(contributed_y.to_numpy().T, 0))
        # print()

        classification_c.fill(Classification.Empty)
        classification_x.fill(Classification.Empty)
        classification_y.fill(Classification.Empty)

        mass_c.fill(0.0)
        mass_x.fill(0.0)
        mass_y.fill(0.0)

        contributed_c.fill(0.0)
        contributed_x.fill(0.0)
        contributed_y.fill(0.0)

        print()
        particle_to_grid(position_p, False)
        classify_cells()
        grid_to_particle(position_p)

        # print("~" * 100)
        print(f"-> QUADRATIC @ {position_p}:")
        # print("CELL:")
        # print("\nMASS:")
        print_mass(np.flip(mass_c.to_numpy().T, 0))
        # print("\nCLASSIFICATION:")
        # print_classification(np.flip(classification_c.to_numpy().T, 0))
        # print("\nCONTRIBUTION G2P:")
        # print_mass(np.flip(contributed_c.to_numpy().T, 0))
        print()


if __name__ == "__main__":
    main()
