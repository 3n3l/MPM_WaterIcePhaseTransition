from src.solvers import MPM_Solver
from src.constants import State

import taichi as ti


@ti.data_oriented
class PoissonDiskSampler:
    def __init__(self, mpm_solver: MPM_Solver, r: float = 0.003, k: int = 300) -> None:
        # Some of the solver's constants wills be used:
        self.solver = mpm_solver

        self.r = r  # Minimum distance between samples
        self.k = k  # Samples to choose before rejection
        self.dx = r / ti.sqrt(2)  # Cell size is bounded by this
        self.n_grid = int(1 / self.dx)  # Number of cells in the grid

        # Initialize an n-dimension background grid to store samples:
        self.background_grid = ti.field(dtype=ti.i32, shape=(self.n_grid, self.n_grid))

        # We can't use a resizable list, so we point to the head and tail:
        self._head = ti.field(int, shape=())
        self._tail = ti.field(int, shape=())

    @ti.func
    def _has_collision(self, base_point: ti.template()) -> bool:  # pyright: ignore
        x, y = self.point_to_index(base_point)
        _min = (ti.max(0, x - 2), ti.min(self.n_grid, x + 3))  # pyright: ignore
        _max = (ti.max(0, y - 2), ti.min(self.n_grid, y + 3))  # pyright: ignore
        distance_min = ti.sqrt(2)  # Maximum possible distance

        # Search in a 3x3 grid neighborhood around the position
        # TODO: compute lower left as base like in mpm
        # TODO: check all distances against < self.r, return immediately
        for i, j in ti.ndrange(_min, _max):
            if (index := self.background_grid[i, j]) != -1:
                # We found a point and can compute the distance:
                found_point = self.solver.position_p[index]
                distance = (found_point - base_point).norm()
                if distance < distance_min:
                    distance_min = distance

        return distance_min < self.r

    @ti.func
    def _in_bounds(self, point: ti.template(), geometry: ti.template()) -> bool:  # pyright: ignore
        in_bounds = self.solver.in_bounds(point[0], point[1])  # in simulation bounds
        in_bounds &= geometry.in_bounds(point[0], point[1])  # in geometry bounds
        return in_bounds

    @ti.func
    def point_to_index(self, point: ti.template()) -> ti.Vector:  # pyright: ignore
        return ti.cast(point * self.n_grid, ti.i32)  # pyright: ignore

    @ti.func
    def point_fits(self, point: ti.template(), geometry: ti.template()) -> bool:  # pyright: ignore
        point_has_been_found = not self._has_collision(point)  # no collision
        point_has_been_found &= self._in_bounds(point, geometry)  # in bounds
        return point_has_been_found

    @ti.func
    def can_sample_more_points(self) -> bool:
        return (self._head[None] < self._tail[None]) and (self._head[None] < self.solver.max_particles)

    @ti.func
    def initialize_grid(self, n_particles: ti.i32, positions: ti.template()):  # pyright: ignore
        for i, j in self.background_grid:
            self.background_grid[i, j] = -1

        for p in ti.ndrange(n_particles):
            # We ignore uninitialized particles:
            if self.solver.state_p[p] == State.Hidden:
                continue

            index = self.point_to_index(positions[p])
            self.background_grid[index] = p

    @ti.func
    def initialize_pointers(self, n_particles: ti.i32):  # pyright: ignore
        self._tail[None] = n_particles + 1
        self._head[None] = n_particles

    @ti.func
    def generate_point_around(self, prev_position: ti.template()) -> ti.Vector:  # pyright: ignore
        theta = ti.random() * 2 * ti.math.pi
        offset = ti.Vector([ti.cos(theta), ti.sin(theta)])
        offset *= (1 + ti.random()) * self.r
        return prev_position + offset

    @ti.func
    def generate_initial_point(self, geometry: ti.template()) -> ti.Vector:  # pyright: ignore
        initial_point = geometry.random_seed()
        n_samples = 0  # otherwise this might not halt
        while not self.point_fits(initial_point, geometry) and n_samples < self.k:
            initial_point = geometry.random_seed()
            n_samples += 1

        index = self.point_to_index(initial_point)
        self.background_grid[index] = self.solver.n_particles[None]

        return initial_point

    @ti.func
    def increment_head(self):
        self._head[None] += 1

    @ti.func
    def increment_tail(self):
        self._tail[None] += 1

    @ti.func
    def tail(self):
        return self._tail[None]
