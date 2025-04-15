import taichi as ti

from src.enums import State
from src.geometries import Geometry
from src.mpm_solver import MPM_Solver


@ti.data_oriented
class PoissonDiskSampler:
    def __init__(self, mpm_solver: MPM_Solver, r: float = 0.0025, k: int = 100) -> None:
        # Get these from the mpm solver
        self.max_samples = mpm_solver.max_particles
        self.sample_count = mpm_solver.n_particles
        # self.n_particles = mpm_solver.n_particles
        self.position_p = mpm_solver.position_p
        self.solver = mpm_solver

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

        # The sampled points will go here, the index is stored on the grid for easy access.
        self.samples_by_index = ti.Vector.field(2, dtype=ti.f32, shape=self.max_samples)

        # self.max_samples = ti.field(int, shape=())
        # self.sample_count = ti.field(int, shape=())
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
                found_point = self.samples_by_index[index]
                distance = (found_point - base_point).norm()
                if distance < distance_min:
                    distance_min = distance
        # TODO: why subtract 1e-6???
        return distance_min < self.r - 1e-6

    @ti.func
    def _some_condition_is_not_reached(self) -> bool:
        # TODO: give this a proper name
        stuff = self.head[None] < self.tail[None]
        stuff &= self.head[None] <= ti.min(self.sample_count[None], self.max_samples)
        return stuff

    @ti.func
    def _add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()) -> None:  # pyright: ignore
        p = self.solver.n_particles[None]
        # print("particle:", p, position)

        self.solver.conductivity_p[p] = geometry.conductivity
        self.solver.temperature_p[p] = geometry.temperature
        self.solver.capacity_p[p] = geometry.capacity
        self.solver.velocity_p[p] = geometry.velocity
        self.solver.color_p[p] = geometry.color
        self.solver.phase_p[p] = geometry.phase
        self.solver.heat_p[p] = geometry.heat

        # Reset properties.
        self.solver.mass_p[p] = self.solver.particle_vol * self.solver.rho_0
        self.solver.inv_lambda_p[p] = 1 / self.solver.lambda_0[None]
        self.solver.FE_p[p] = ti.Matrix([[1, 0], [0, 1]])
        self.solver.C_p[p] = ti.Matrix.zero(float, 2, 2)
        self.solver.JE_p[p] = 1
        self.solver.JP_p[p] = 1

        # TODO:
        self.solver.position_p[p] = position
        self.solver.activation_state_p[p] = State.Active
        # self.solver.active_position_p[index] = position
        # self.solver.activation_threshold_p[index] = geometry.frame_threshold
        self.solver.activation_threshold_p[p] = 0
        # self.solver.n_particles[None] += 1

    @ti.func
    def _sample_single_point(self, geometry: ti.template()) -> None:  # pyright: ignore
        while self._some_condition_is_not_reached():
            prev_position = self.samples_by_index[self.head[None]]
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
                # point_has_been_found &= self.solver.in_bounds(next_x, next_y)  # in simulation bounds
                point_has_been_found &= geometry.in_bounds(next_x, next_y)  # in geometry bounds
                if point_has_been_found:
                    particle_index = self.sample_count[None] + self.tail[None]
                    # self.position_p[self.sample_count[None] + self.tail[None]] = next_position
                    self._add_particle(particle_index, next_position, geometry)
                    self.samples_by_index[self.tail[None]] = next_position
                    self.background_grid[next_index] = self.tail[None]
                    self.tail[None] += 1
                    self.sample_count[None] += 1

    @ti.kernel
    # def reset(self) -> None:
    def reset(self):
        # Reset values for the new sample:
        # self.max_samples[None] = desired_samples
        self.sample_count[None] = 1
        self.head[None] = 0
        self.tail[None] = 1

        # Reset the background grid:
        for i, j in self.background_grid:
            self.background_grid[i, j] = -1

    def seed_from_running_simulation(self):
        pass

    @ti.kernel
    def sample(self, desired_samples: ti.i32, geometry: ti.template()):  # pyright: ignore
        # TODO: this needs some check if n_samples + desired_samples > max_samples

        # Reset values for the new sample:
        self.sample_count[None] = 1
        self.head[None] = 0
        self.tail[None] = 1

        # Reset the background grid:
        for i, j in self.background_grid:
            self.background_grid[i, j] = -1

        # Start in the center of the geometry
        initial_point = geometry.center
        self.samples_by_index[0] = initial_point
        index = self._position_to_index(initial_point)
        self.background_grid[index] = 0

        for _ in ti.ndrange(desired_samples):
            self._sample_single_point(geometry)
            # self.sample_count[None] += 1
            # self.solver.n_particles[None] = self.sample_count[None]
            # self.sample_count[None] += 1  # FIXME: just for testing, must be integrated to MPM
