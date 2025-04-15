from src.enums import Conductivity, Phase, Color, State, Capacity
from src.configurations import Configuration
from src.sampling import PoissonDiskSampler
from src.mpm_solver import MPM_Solver
from src.geometries import Geometry

# from src.geometries import Geometry

from datetime import datetime

import taichi as ti
import os


@ti.data_oriented
class HeadlessRenderer:
    def __init__(
        self,
        name: str,
        res: tuple[int, int] | int,
        solver: MPM_Solver,
        configurations: list[Configuration],
        max_frames: int = 0,
    ) -> None:
        """Constructs a Renderer object, this advances the MLS-MPM solver and renders the updated particle positions.
        ---
        Parameters:
            name: string displayed at the top of the window
            res: tuple holding window width and height
            solver: the MLS-MPM solver
            configurations: list of configurations for the solver
        """
        # State.
        self.is_paused = True
        self.max_frames = max_frames
        self.should_write_to_disk = False
        self.is_showing_settings = not self.is_paused

        # Create a parent directory, more directories will be created inside this
        # directory that contain newly created frames, videos and GIFs.
        self.parent_dir = ".output"
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)

        # The MLS-MPM solver.
        self.solver = solver

        # Poisson disk sampling.
        self.sampler = PoissonDiskSampler(mpm_solver=self.solver)
        # self.sampler.reset()

        # Properties for the Poisson Disk Sampling
        self.n_grid = solver.n_grid
        self.classification_c = ti.field(dtype=ti.int8, shape=(self.n_grid, self.n_grid))

        # Build the Taichi fields.
        # TODO: remove
        # for configuration in configurations:
        #     configuration.build()

        # Load the initial configuration and reset the solver to this configuration.
        self.configuration_id = 0
        self.configurations = configurations
        self.configuration = configurations[self.configuration_id]
        self.load_configuration(self.configuration)
        # self.reset_solver(self.configuration)

    # TODO: just dump frames here?
    def render(self) -> None:
        pass
        # if self.should_write_to_disk:
        #     self.video_manager.write_frame(self.window.get_image_buffer_as_numpy())

    # TODO: this should be substep?
    def run(self) -> None:
        while self.solver.current_frame[None] < self.max_frames:
            self.substep()

    # @ti.kernel
    # def add_geometry(self, geometry: ti.template()):  # pyright: ignore
    def add_geometry(self, geometry: Geometry) -> None:
        # Idea: Compute background grid from all particle positions,
        #       then to the PDS. This way even geometries that are added
        #       later on won't clash with each other

        # TODO: think about which parameters can be moved here from the solver
        # n_particles = self.solver.n_particles[None]
        self.sampler.sample(10_000, geometry)

        # TODO: geometry.n_particles is not known here
        # for p in ti.ndrange(n_particles, n_particles + geometry.n_particles):
        #     # Apply properties from the geometry.
        #     self.solver.conductivity_p[p] = geometry.conductivity
        #     self.solver.temperature_p[p] = geometry.temperature
        #     self.solver.capacity_p[p] = geometry.capacity
        #     self.solver.velocity_p[p] = geometry.velocity
        #     self.solver.color_p[p] = geometry.color
        #     self.solver.phase_p[p] = geometry.phase
        #     self.solver.heat_p[p] = geometry.heat
        #
        #     # Reset properties.
        #     self.solver.mass_p[p] = self.solver.particle_vol * self.solver.rho_0
        #     self.solver.inv_lambda_p[p] = 1 / self.solver.lambda_0[None]
        #     self.solver.FE_p[p] = ti.Matrix([[1, 0], [0, 1]])
        #     self.solver.C_p[p] = ti.Matrix.zero(float, 2, 2)
        #     # self.solver.position_p[p] = [0.5, 0.5]  # TODO: calculate position with PDS
        #     self.solver.JE_p[p] = 1
        #     self.solver.JP_p[p] = 1
        #     # FIXME: these are all unused now and can be deleted
        #     # self.solver.activation_threshold_p[p] = configuration.activation_threshold_p[p]
        #     # self.solver.activation_state_p[p] = configuration.state_p[p]
        #     # offset_position = configuration.position_p[p] + self.solver.boundary_offset
        #     # p_is_active = configuration.state_p[p] == State.Active
        #     # self.solver.active_position_p[p] = offset_position if p_is_active else [0, 0]
        #
        # self.solver.n_particles[None] += geometry.n_particles

    def substep(self) -> None:
        # TODO: move current_frame to this class
        self.solver.current_frame[None] += 1

        # TODO: check against frame thresholds, add geometry if necessary
        # Load all the subsequent geometries into solver, once the frame threshold is reached:
        # if len(self.configuration.subsequent_geometries) > 0:
        #     current_frame = self.solver.current_frame[None]
        #     frame_threshold = self.configuration.subsequent_geometries[0].frame_threshold
        #     # TODO: this only loads one geometry at the moment,
        #     #       but there could be more with the same frame
        #     if current_frame == frame_threshold:
        #         # TODO: the popping ruins the list in the configuration?
        #         #       So copy it first? Or don't pop?
        #         geometry = self.configuration.subsequent_geometries.pop()
        #         self.add_geometry(geometry)

        for _ in range(int(2e-3 // self.solver.dt)):
            self.solver.reset_grids()
            self.solver.particle_to_grid()
            self.solver.momentum_to_velocity()
            self.solver.classify_cells()
            self.solver.compute_volumes()
            self.solver.pressure_solver.solve()
            self.solver.heat_solver.solve()
            self.solver.grid_to_particle()

    # @ti.kernel
    # @ti.func
    def load_configuration(self, configuration: ti.template()):  # pyright: ignore
        """
        Loads the chosen configuration into the MLS-MPM solver.
        ---
        Parameters:
            configuration: Configuration
        """
        self.solver.ambient_temperature[None] = configuration.ambient_temperature
        # self.solver.n_particles[None] = configuration.n_particles
        self.solver.stickiness[None] = configuration.stickiness
        self.solver.friction[None] = configuration.friction
        self.solver.lambda_0[None] = configuration.lambda_0
        self.solver.theta_c[None] = configuration.theta_c
        self.solver.theta_s[None] = configuration.theta_s
        self.solver.zeta[None] = configuration.zeta
        self.solver.mu_0[None] = configuration.mu_0
        self.solver.nu[None] = configuration.nu
        self.solver.E[None] = configuration.E

        # Load all the initial geometries into the solver:
        # TODO: check against frame thresholds, add geometry if necessary
        # TODO: this must be done after resetting? maybe in some other reset method?
        # while len(self.configuration.initial_geometries) > 0:
            # TODO: don't pop on the configuration list
        # geometry = self.configuration.initial_geometries.pop()
        self.reset_solver(self.configuration)
        self.solver.n_particles[None] = 0
        self.sampler.sample_count[None] = 0
        geometry = self.configuration.initial_geometries[0]
        self.add_geometry(geometry)

    @ti.kernel
    def reset_solver(self, configuration: ti.template()):  # pyright: ignore
        """
        Resets the MLS-MPM solver to the field values of the configuration.
        ---
        Parameters:
            configuration: Configuration
        """
        self.solver.current_frame[None] = 0
        for p in self.solver.position_p:
            # if p < configuration.n_particles:
            #     particle_is_water = configuration.phase_p[p] == Phase.Water
            #     self.solver.conductivity_p[p] = Conductivity.Water if particle_is_water else Conductivity.Ice
            #     self.solver.capacity_p[p] = Capacity.Water if particle_is_water else Capacity.Ice
            #     self.solver.color_p[p] = Color.Water if particle_is_water else Color.Ice
            #     self.solver.heat_p[p] = LATENT_HEAT if particle_is_water else 0.0
            #
            #     self.solver.activation_threshold_p[p] = configuration.activation_threshold_p[p]
            #     self.solver.temperature_p[p] = configuration.temperature_p[p]
            #     self.solver.activation_state_p[p] = configuration.state_p[p]
            #     self.solver.velocity_p[p] = configuration.velocity_p[p]
            #     self.solver.phase_p[p] = configuration.phase_p[p]
            #
            #     offset_position = configuration.position_p[p] + self.solver.boundary_offset
            #     p_is_active = configuration.state_p[p] == State.Active
            #     self.solver.active_position_p[p] = offset_position if p_is_active else [0, 0]
            #     self.solver.position_p[p] = offset_position
            # else:
            #     # TODO: this might be completely irrelevant, as only the first n_particles are used anyway?
            #     #       So work can be saved by just ignoring all the other particles and iterating only
            #     #       over the configuration.n_particles?

            # TODO: As this is done in add geometry, this can probably go???
            #       Because these properties are overwritten by a new geometry anyways?
            #       So as long as the n_particles value is adjusted this should work?

            # Reset all properties.
            self.solver.activation_state_p[p] = State.Inactive
            self.solver.conductivity_p[p] = Conductivity.Zero
            self.solver.color_p[p] = Color.Background
            self.solver.capacity_p[p] = Capacity.Zero
            self.solver.activation_threshold_p[p] = 0
            self.solver.phase_p[p] = Phase.Water
            self.solver.active_position_p[p] = [0, 0]
            self.solver.position_p[p] = [0, 0]
            self.solver.velocity_p[p] = [0, 0]
            self.solver.temperature_p[p] = 0
            self.solver.heat_p[p] = 0

            self.solver.mass_p[p] = self.solver.particle_vol * self.solver.rho_0

            # FIXME: this is just for testing
            # TODO: set lambda depending on phase
            self.solver.inv_lambda_p[p] = 1 / self.solver.lambda_0[None]
            # self.solver.inv_lambda_p[p] = 1 / 9999999999.0

            self.solver.FE_p[p] = ti.Matrix([[1, 0], [0, 1]])
            self.solver.C_p[p] = ti.Matrix.zero(float, 2, 2)
            self.solver.JE_p[p] = 1
            self.solver.JP_p[p] = 1

        # geometry = self.configuration.initial_geometries[0]
        # self.add_geometry(geometry)
        # self.sampler.sample(10_000, self.configuration.initial_geometries[0])

    def dump_frames(self) -> None:
        """Creates an output directory, a VideoManager in this directory and then dumps frames to this directory."""
        date = datetime.now().strftime("%d%m%Y_%H%M%S")
        output_dir = f"{self.parent_dir}/{date}"
        os.makedirs(output_dir)
        self.video_manager = ti.tools.VideoManager(
            output_dir=output_dir,
            framerate=60,
            automatic_build=False,
        )

    def create_video(self) -> None:
        """Converts stored frames in the before created output directory to a video."""
        self.video_manager.make_video(gif=True, mp4=True)
