import utils

from src.geometries import Rectangle, Circle
from src.enums import Color, Phase

import taichi as ti

# ti.init(arch=ti.gpu)
ti.init(arch=ti.cpu, debug=True)

n_particles = ti.field(dtype=int, shape=())
n_particles[None] = 0
max_particles = 1_000_000
position_p = ti.Vector.field(2, dtype=float, shape=max_particles)

# TODO: choosing samples and radius is a bit weird, better would be to control sample density?
#       But then how to control the max number of iterations? Filling an arbitrary space seems atleast np-hard?


@ti.data_oriented
class PoissonDiskSampler:
    def __init__(self, r: float = 0.0025, k: int = 500, desired_samples: int = 500_000) -> None:
        self.r = r  # Minimum distance between samples
        self.k = k  # Samples to choose before rejection
        self.dx = r / ti.sqrt(2)  # Cell size is bounded by this
        self.n_grid = int(1 / self.dx)  # Number of cells in the grid

        # Initialize an n-dimension background grid fro storing samples.
        self.background_grid = ti.field(dtype=ti.i32, shape=(self.n_grid, self.n_grid))

        # Fill the grid, -1 indicates no sample, a non-negative integer gives
        # the indes of the sample located in a cell.
        # TODO: this can be reset in sample()
        self.background_grid.fill(-1)

        # The sampled points will go here, we sample until this field is filled.
        # TODO: this can be removed if used on mpm_solver.position_p?
        #       But some methods need self.sample[index], where index comes from the grid
        self.samples = ti.Vector.field(2, dtype=ti.f32, shape=desired_samples)

        self.desired_samples = ti.field(int, shape=())
        self.sample_count = ti.field(int, shape=())
        self.head = ti.field(int, shape=())
        self.tail = ti.field(int, shape=())

    @ti.func
    def _position_to_index(self, point: ti.template()) -> ti.Vector:  # pyright: ignore
        x = ti.cast(point[0] * self.n_grid, ti.i32)
        y = ti.cast(point[1] * self.n_grid, ti.i32)
        return ti.Vector([x, y])

    @ti.func
    def _has_collision(self, base_point: ti.template()) -> bool:  # pyright: ignore
        x, y = self._position_to_index(base_point)
        _min = (ti.max(0, x - 2), ti.min(self.n_grid, x + 3))  # pyright: ignore
        _max = (ti.max(0, y - 2), ti.min(self.n_grid, y + 3))  # pyright: ignore
        distance_min = ti.sqrt(2)  # Maximum possible distance
        # Search in a 3x3 grid neighborhood around the position
        for i, j in ti.ndrange(_min, _max):
            if (index := self.background_grid[i, j]) != -1:
                # We found a point and can compute the distance:
                found_point = self.samples[index]
                distance = (found_point - base_point).norm()
                if distance < distance_min:
                    distance_min = distance
        # TODO: why subtract 1e-6???
        return distance_min < self.r - 1e-6

    @ti.func
    def _some_condition_is_not_reached(self) -> bool:
        # TODO: give this a proper name
        stuff = self.head[None] < self.tail[None]
        stuff &= self.head[None] < ti.min(self.sample_count[None], self.desired_samples[None])
        return stuff

    @ti.func
    def _sample_single_point(self, geometry: ti.template()) -> None:  # pyright: ignore
        while self._some_condition_is_not_reached():
            prev_position = self.samples[self.head[None]]
            self.head[None] += 1

            for _ in range(self.k):
                theta = ti.random() * 2 * ti.math.pi
                offset = ti.Vector([ti.cos(theta), ti.sin(theta)])
                offset *= (1 + ti.random()) * self.r
                next_position = prev_position + offset
                next_index = self._position_to_index(next_position)
                next_x, next_y = next_position[0], next_position[1]  # pyright: ignore

                point_has_been_found = not self._has_collision(next_position)  # no collision
                point_has_been_found &= 0 <= next_x < 1 and 0 <= next_y < 1  # in simulation bounds
                point_has_been_found &= geometry.in_bounds(next_x, next_y)  # in geometry bounds
                if point_has_been_found:
                    # FIXME: this is not needed
                    self.samples[self.tail[None]] = next_position
                    # FIXME: this adds to the global positions
                    position_p[n_particles[None] + self.tail[None]] = next_position

                    self.background_grid[next_index] = self.tail[None]
                    self.tail[None] += 1

    @ti.kernel
    def sample(self, desired_samples: ti.i32, geometry: ti.template()):  # pyright: ignore
        # Reset values for the new sample:
        self.desired_samples[None] = desired_samples
        self.sample_count[None] = 1
        self.head[None] = 0
        self.tail[None] = 1

        # Start in the center of the geometry
        initial_point = geometry.center
        self.samples[0] = initial_point
        index = self._position_to_index(initial_point)
        self.background_grid[index] = 0

        # Reset the background grid:
        for i, j in self.background_grid:
            self.background_grid[i, j] = -1

        # TODO: desired_samples = max_samples
        for _ in ti.ndrange(self.desired_samples[None]):
            self._sample_single_point(geometry)
            self.sample_count[None] += 1
            n_particles[None] += 1  # FIXME: just for testing, must be integrated to MPM


@ti.kernel
def naive_sample_rectangle(n_new: int, rectangle: ti.template()):  # pyright: ignore
    for p in ti.ndrange((n_particles[None], n_particles[None] + n_new)):
        x = ti.random() * rectangle.width + rectangle.x
        y = ti.random() * rectangle.height + rectangle.y
        position_p[p] = [x, y]
    n_particles[None] = n_particles[None] + n_new


@ti.kernel
def naive_sample_circle(n_new: int, circle: ti.template()):  # pyright: ignore
    for p in ti.ndrange((n_particles[None], n_particles[None] + n_new)):
        t = 2 * ti.math.pi * ti.random()
        r = circle.radius * ti.math.sqrt(ti.random())
        x = (r * ti.sin(t)) + circle.x
        y = (r * ti.cos(t)) + circle.y
        position_p[p] = [x, y]
    n_particles[None] = n_particles[None] + n_new


def main() -> None:
    window = ti.ui.Window(
        "Poisson Disk Sampling [LEFT] vs. Naive Implementation [RIGHT]",
        res=(720, 720),
        fps_limit=60,
    )
    canvas = window.get_canvas()

    n_samples = 5_000

    ### Poisson Disk Sampling
    pds = PoissonDiskSampler()
    pds.sample(
        3 * n_samples,
        Circle(
            phase=Phase.Ice,
            radius=0.1,
            velocity=(0, 0),
            center=(0.25, 0.75),
        ),
    )
    pds.sample(
        3 * n_samples,
        Rectangle(
            phase=Phase.Ice,
            size=(0.2, 0.2),
            velocity=(0, 0),
            lower_left=(0.15, 0.15),
        ),
    )

    ### Naive Implementation:
    naive_sample_circle(
        n_samples,
        Circle(
            phase=Phase.Ice,
            radius=0.1,
            velocity=(0, 0),
            center=(0.75, 0.75),
        ),
    )
    naive_sample_rectangle(
        n_samples,
        Rectangle(
            phase=Phase.Ice,
            size=(0.2, 0.2),
            velocity=(0, 0),
            lower_left=(0.65, 0.15),
        ),
    )

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                window.running = False

        canvas.set_background_color(Color.Background)
        canvas.circles(color=Color.Ice, centers=position_p, radius=0.001)
        window.show()


if __name__ == "__main__":
    main()
