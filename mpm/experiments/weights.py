import taichi as ti

ti.init(arch=ti.cpu)

# NOTE: this seems to be wrong on paper, but is the best working version in the simulation?!
# TODO: debug the simulation, there might be another reason for this?!
# TODO: then try a version of this that adds to the closest nodes/faces


@ti.kernel
def compute_weights(x: float, y: float):
    p_position = ti.Vector([x, y])
    n_grid = 4
    dx = 1 / n_grid
    inv_dx = float(n_grid)

    additional_offset = 0.5

    c_stagger = ti.Vector([additional_offset, additional_offset])
    x_stagger = ti.Vector([(dx / 2) + additional_offset, additional_offset])
    y_stagger = ti.Vector([additional_offset, (dx / 2) + additional_offset])

    # c_base = (p_position * inv_dx - c_stagger).cast(ti.i32)
    # c_base = ti.floor(p_position * inv_dx - c_stagger, ti.i32)
    c_base = ti.cast(p_position * inv_dx - c_stagger, ti.i32)
    c_fx = p_position * inv_dx - ti.cast(c_base, ti.f32)

    # x_base = (p_position * inv_dx - x_stagger).cast(ti.i32)
    # x_base = ti.floor(p_position * inv_dx - x_stagger, ti.i32)
    x_base = ti.cast(p_position * inv_dx - x_stagger, ti.i32)
    x_fx = p_position * inv_dx - ti.cast(x_base - ti.Vector([dx / 2, 0.0]), ti.f32)

    # y_base = (p_position * inv_dx - y_stagger).cast(ti.i32)
    # y_base = ti.floor(p_position * inv_dx - y_stagger, ti.i32)
    y_base = ti.cast(p_position * inv_dx - y_stagger, ti.i32)
    y_fx = p_position * inv_dx - ti.cast(y_base - ti.Vector([0.0, dx / 2]), ti.f32)

    print("position:    ", p_position)
    print("c_base:      ", c_base)
    print("c_fx:        ", c_fx)
    print("x_base:      ", x_base)
    print("x_fx:        ", x_fx)
    print("y_base:      ", y_base)
    print("y_fx:        ", y_fx)


def main():
    positions = [(0.0, 0.0), (0.1, 0.1), (0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (0.9, 0.9), (1.0, 1.0)]
    for x, y in positions:
        print()
        print("=" * 50)
        compute_weights(x, y)


if __name__ == "__main__":
    main()
