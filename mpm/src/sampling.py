from src.mpm_solver import MPM_Solver
from src.enums import State

import taichi as ti


@ti.data_oriented
class PoissonDiskSampler:
    def __init__(self, mpm_solver: MPM_Solver, r: float = 0.004, k: int = 100) -> None:
        # Some of the solver's constants wills be used:
        self.solver = mpm_solver

        self.r = r  # Minimum distance between samples
        self.k = k  # Samples to choose before rejection
        self.dx = r / ti.sqrt(2)  # Cell size is bounded by this
        self.n_grid = int(1 / self.dx)  # Number of cells in the grid

        # Initialize an n-dimension background grid to store samples:
        self.background_grid = ti.field(dtype=ti.i32, shape=(self.n_grid, self.n_grid))

        # We can't use a resizable list, so we point to the head and tail:
        self.head = ti.field(int, shape=())
        self.tail = ti.field(int, shape=())

    @ti.func
    def _seed_particle(self, position: ti.template(), geometry: ti.template()) -> None:  # pyright: ignore
        index = self.tail[None]

        # Seed from the geometry:
        self.solver.conductivity_p[index] = geometry.conductivity
        self.solver.temperature_p[index] = geometry.temperature
        self.solver.capacity_p[index] = geometry.capacity
        self.solver.velocity_p[index] = geometry.velocity
        self.solver.color_p[index] = geometry.color
        self.solver.phase_p[index] = geometry.phase
        self.solver.heat_p[index] = geometry.heat

        # Set properties to default values:
        self.solver.mass_p[index] = self.solver.particle_vol * self.solver.rho_0
        self.solver.inv_lambda_p[index] = 1 / self.solver.lambda_0[None]
        self.solver.FE_p[index] = ti.Matrix([[1, 0], [0, 1]])
        self.solver.C_p[index] = ti.Matrix.zero(float, 2, 2)
        self.solver.state_p[index] = State.Active
        self.solver.position_p[index] = position
        self.solver.JE_p[index] = 1.0
        self.solver.JP_p[index] = 1.0

    @ti.func
    def _position_to_index(self, position: ti.template()) -> ti.Vector:  # pyright: ignore
        return ti.cast(position * self.n_grid, ti.i32)  # pyright: ignore

    @ti.func
    def _has_collision(self, base_point: ti.template()) -> bool:  # pyright: ignore
        x, y = self._position_to_index(base_point)
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
    def _could_sample_more_points(self) -> bool:
        return (self.head[None] < self.tail[None]) and (self.head[None] <= self.solver.max_particles)

    @ti.func
    def _sample_single_point(self, geometry: ti.template()):  # pyright: ignore
        while self._could_sample_more_points():
            prev_position = self.solver.position_p[self.head[None]]
            self.head[None] += 1

            for _ in range(self.k):
                theta = ti.random() * 2 * ti.math.pi
                offset = ti.Vector([ti.cos(theta), ti.sin(theta)])
                offset *= (1 + ti.random()) * self.r
                next_position = prev_position + offset
                next_index = self._position_to_index(next_position)
                next_x, next_y = next_position[0], next_position[1]  # pyright: ignore

                point_has_been_found = not self._has_collision(next_position)  # no collision
                # TODO: check for simulation bounds instead of 0, 1
                point_has_been_found &= 0 <= next_x < 1 and 0 <= next_y < 1  # in simulation bounds
                # point_has_been_found &= self.solver.in_bounds(next_x, next_y)  # in simulation bounds
                point_has_been_found &= geometry.in_bounds(next_x, next_y)  # in geometry bounds
                if point_has_been_found:
                    self.background_grid[next_index] = self.tail[None]
                    self._seed_particle(next_position, geometry)
                    self.tail[None] += 1

    @ti.kernel
    def sample_geometry(self, geometry: ti.template()):  # pyright: ignore
        # Reset the background grid:
        for i, j in self.background_grid:
            self.background_grid[i, j] = -1

        # Seed the background grid from current positions:
        for p in ti.ndrange(self.solver.n_particles[None]):
            # We ignore uninitialized particles:
            if self.solver.state_p[p] == State.Hidden:
                continue

            position = self.solver.position_p[p]
            index = self._position_to_index(position)
            self.background_grid[index] = p

        # Update state, for a fresh sample this will be (0, 1), in the running simulation
        # this will reset this to where we left of, allowing to add more particles
        self.head[None] = self.solver.n_particles[None]
        self.tail[None] = self.solver.n_particles[None] + 1

        # Set the initial point for this sample to the center of the geometry.
        # TODO: this can collide and should be looking for an empty spot?!
        # TODO: a falling spout source is better off starting seeding at the top
        # TODO: move this point to the geometry, as 'initial_seed' or something
        initial_point = geometry.center + ti.math.vec2(0, 0.5 * geometry.height)

        # self.solver.position_p[self.solver.n_particles[None]] = initial_point
        index = self._position_to_index(initial_point)
        self.background_grid[index] = self.solver.n_particles[None]
        self._seed_particle(initial_point, geometry)
        self.head[None] += 1
        self.tail[None] += 1

        self._sample_single_point(geometry)
        self.solver.n_particles[None] = self.tail[None]
