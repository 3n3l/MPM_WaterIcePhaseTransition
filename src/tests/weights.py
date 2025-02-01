import taichi as ti

ti.init(arch=ti.cpu)

# NOTE: this seems to be wrong on paper, but is the best working version in the simulation,
#       right now we always apply properties to the same node and faces, meaning that
#       the coordinates will always be the same for all nodes/faces, this shouldn't be right?
# TODO: debug the simulation, there might be another reason for this?
# TODO: then try a version of this that adds to the closest nodes/faces


@ti.kernel
def compute_weights(x: float, y: float):
    p_position = ti.Vector([x, y])
    n_grid = 4
    dx = 1 / n_grid
    inv_dx = float(n_grid)

    c_stagger = ti.Vector([0.0, 0.0])
    c_base = (p_position * inv_dx - (c_stagger + 0.5)).cast(ti.i32)  # pyright: ignore
    c_fx = p_position * inv_dx - c_base.cast(ti.f32)

    x_stagger = ti.Vector([dx / 2, 0])
    x_base = (p_position * inv_dx - (x_stagger + 0.5)).cast(ti.i32)  # pyright: ignore
    x_fx = p_position * inv_dx - x_base.cast(ti.f32)

    y_stagger = ti.Vector([0, dx / 2])
    y_base = (p_position * inv_dx - (y_stagger + 0.5)).cast(ti.i32)  # pyright: ignore
    y_fx = p_position * inv_dx - y_base.cast(ti.f32)

    print("position:    ", p_position)
    print("c_base:      ", c_base)
    print("c_fx:        ", c_fx)
    print("x_base:      ", x_base)
    print("x_fx:        ", x_fx)
    print("y_base:      ", y_base)
    print("y_fx:        ", y_fx)


def main():
    positions = [(0.1, 0.1), (0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (0.9, 0.9)]
    for x, y in positions:
        print()
        print("-" * 50)
        compute_weights(x, y)


if __name__ == "__main__":
    main()
