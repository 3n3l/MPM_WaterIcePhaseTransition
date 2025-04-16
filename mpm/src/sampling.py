from src.mpm_solver import MPM_Solver

import taichi as ti


@ti.data_oriented
class PoissonDiskSampler:
    def __init__(self, mpm_solver: MPM_Solver, r: float = 0.005, k: int = 100) -> None:
        # Get these from the mpm solver
        self.max_samples = mpm_solver.max_particles
        # self.n_particles = mpm_solver.n_particles
        # self.position_p = mpm_solver.position_p
        self.solver = mpm_solver

        self.r = r  # Minimum distance between samples
        self.k = k  # Samples to choose before rejection
        self.dx = r / ti.sqrt(2)  # Cell size is bounded by this
        self.n_grid = int(1 / self.dx)  # Number of cells in the grid

        # Initialize an n-dimension background grid to store samples.
        self.background_grid = ti.field(dtype=ti.i32, shape=(self.n_grid, self.n_grid))

        # Fill the grid, -1 indicates no sample, a non-negative integer gives
        # the indes of the sample located in a cell.
        # TODO: this can be reset in sample()
        self.background_grid.fill(-1)

        # The sampled points will go here, the index is stored on the grid for easy access.
        # self.samples_by_index = ti.Vector.field(2, dtype=ti.f32, shape=self.max_samples)
        # self.samples_by_index = self.solver.position_p

        # self.max_samples = ti.field(int, shape=())
        # self.sample_count = ti.field(int, shape=())
        self.head = ti.field(int, shape=())
        self.tail = ti.field(int, shape=())

    @ti.func
    def _position_to_index(self, position: ti.template()) -> ti.Vector:  # pyright: ignore
        x = ti.cast(position[0] * self.n_grid, ti.i32)
        y = ti.cast(position[1] * self.n_grid, ti.i32)
        return ti.Vector([x, y])

    @ti.func
    def _has_collision(self, base_point: ti.template()) -> bool:  # pyright: ignore
        x, y = self._position_to_index(base_point)
        _min = (ti.max(0, x - 2), ti.min(self.n_grid, x + 3))  # pyright: ignore
        _max = (ti.max(0, y - 2), ti.min(self.n_grid, y + 3))  # pyright: ignore
        distance_min = ti.sqrt(2)  # Maximum possible distance

        # Search in a 3x3 grid neighborhood around the position
        for i, j in ti.ndrange(_min, _max):
            # TODO: check all distances against < self.r, return immediately
            if (index := self.background_grid[i, j]) != -1:
                # We found a point and can compute the distance:
                found_point = self.solver.position_p[index]
                distance = (found_point - base_point).norm()
                if distance < distance_min:
                    distance_min = distance

        return distance_min < self.r

    @ti.func
    def _some_condition_is_not_reached(self) -> bool:
        stuff = self.head[None] <= ti.min(self.solver.n_particles[None], self.max_samples)
        stuff &= self.head[None] < self.tail[None]
        return stuff

    @ti.func
    def _seed_particle(self, position: ti.template(), geometry: ti.template()) -> None:  # pyright: ignore
        # print(self.solver.n_particles[None], self.tail[None])
        # p = self.tail[None]
        p = self.solver.n_particles[None]
        # p = self.tail[None]

        print(f"n_particles = {self.solver.n_particles[None]}")
        # print(f"tail = {self.tail[None]}")

        # Seed from the geometry:
        self.solver.conductivity_p[p] = geometry.conductivity
        self.solver.temperature_p[p] = geometry.temperature
        self.solver.capacity_p[p] = geometry.capacity
        self.solver.velocity_p[p] = geometry.velocity
        self.solver.color_p[p] = geometry.color
        self.solver.phase_p[p] = geometry.phase
        self.solver.heat_p[p] = geometry.heat

        # Set properties to default values:
        self.solver.mass_p[p] = self.solver.particle_vol * self.solver.rho_0
        self.solver.inv_lambda_p[p] = 1 / self.solver.lambda_0[None]
        self.solver.FE_p[p] = ti.Matrix([[1, 0], [0, 1]])
        self.solver.C_p[p] = ti.Matrix.zero(float, 2, 2)
        self.solver.position_p[p] = position
        self.solver.JE_p[p] = 1.0
        self.solver.JP_p[p] = 1.0

        # Keep track of the added particle:
        # self.solver.n_particles[None] += 1
        self.tail[None] += 1
        ti.atomic_add(self.solver.n_particles[None], 1)

    @ti.func
    def _sample_single_point(self, geometry: ti.template()) -> None:  # pyright: ignore
        while self._some_condition_is_not_reached():
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
                point_has_been_found &= 0 <= next_x < 1 and 0 <= next_y < 1  # in simulation bounds
                # point_has_been_found &= self.solver.in_bounds(next_x, next_y)  # in simulation bounds
                point_has_been_found &= geometry.in_bounds(next_x, next_y)  # in geometry bounds
                if point_has_been_found:
                    # self.samples_by_index[self.tail[None]] = next_position
                    self.background_grid[next_index] = self.tail[None]
                    self._seed_particle(next_position, geometry)

    # @ti.kernel
    # def reset(self):
    #     self.head[None] = 0
    #     self.tail[None] = 0

    @ti.kernel
    def sample_geometry(self, desired_samples: ti.i32, geometry: ti.template()):  # pyright: ignore
        # Reset the background grid:
        for i, j in self.background_grid:
            self.background_grid[i, j] = -1

        # Seed the background groud from current positions:
        for p in ti.ndrange(self.solver.n_particles[None]):
            position = self.solver.position_p[p]
            index = self._position_to_index(position)
            self.background_grid[index] = p

        # print(self.head[None], self.tail[None])

        # self.head[None] = self.solver.n_particles[None]
        # self.tail[None] = self.solver.n_particles[None] + 1

        # print(self.head[None], self.tail[None])
        # self.solver.n_particles[None] += 1

        # Set the initial point for this sample to the center of the geometry.
        # TODO: this can collide and should be looking for an empty spot?!
        # TODO: a falling spout source is better off starting seeding at the top
        # TODO: move this point to the geometry, as 'initial_seed' or something
        initial_point = geometry.center + ti.math.vec2(0, 0.5 * geometry.height)

        # self.solver.position_p[self.solver.n_particles[None]] = initial_point
        index = self._position_to_index(initial_point)
        self.background_grid[index] = self.solver.n_particles[None]
        self._seed_particle(initial_point, geometry)

        # TODO: use tail or n_particles here? Isn't this the same anyway?
        # self.solver.position_p[self.tail[None]] = initial_point
        # self.background_grid[index] = self.tail[None]

        # Update state, for a fresh sample this will be (0, 1, 1), in the running simulation
        # this will reset this to where we left of, allowing to add more particles
        # self.head[None] = self.solver.n_particles[None]
        # self.tail[None] = self.solver.n_particles[None] + 1
        # self.solver.n_particles[None] += 1

        # TODO: set desired_samples in geometry
        for _ in ti.ndrange(desired_samples):
            self._sample_single_point(geometry)
