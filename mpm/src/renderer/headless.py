from src.configurations import Configuration
from src.sampling import PoissonDiskSampler
from src.mpm_solver import MPM_Solver
from datetime import datetime
from src.enums import State

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
        # TODO: pass as parameter
        self.sampler = PoissonDiskSampler(mpm_solver=self.solver)

        # Load the initial configuration and reset the solver to this configuration.
        self.current_frame = 0
        self.configuration_id = 0
        self.configurations = configurations
        self.load_configuration(configurations[self.configuration_id])

    # TODO: just dump frames here?
    def render(self) -> None:
        pass

    # TODO: this should be substep?
    def run(self) -> None:
        while self.solver.current_frame[None] < self.max_frames:
            self.substep()

    def substep(self) -> None:
        # TODO: move current frame to this class
        self.solver.current_frame[None] += 1

        # Load all remaining geometries with a satisfied frame threshold:
        if len(self.subsequent_geometries) > 0:
            if self.solver.current_frame[None] == self.subsequent_geometries[0].frame_threshold:
                geometry = self.subsequent_geometries.pop(0)
                self.sampler.sample_geometry(1000, geometry)

        for _ in range(int(2e-3 // self.solver.dt)):
            self.solver.reset_grids()
            self.solver.particle_to_grid()
            self.solver.momentum_to_velocity()
            self.solver.classify_cells()
            self.solver.compute_volumes()
            self.solver.pressure_solver.solve()
            self.solver.heat_solver.solve()
            self.solver.grid_to_particle()

    def load_configuration(self, configuration: Configuration) -> None:
        """
        Loads the chosen configuration into the MLS-MPM solver.
        ---
        Parameters:
            configuration: Configuration
        """
        self.configuration = configuration

        self.solver.ambient_temperature[None] = configuration.ambient_temperature
        self.solver.lambda_0[None] = configuration.lambda_0
        self.solver.theta_c[None] = configuration.theta_c
        self.solver.theta_s[None] = configuration.theta_s
        self.solver.zeta[None] = configuration.zeta
        self.solver.mu_0[None] = configuration.mu_0
        self.solver.nu[None] = configuration.nu
        self.solver.E[None] = configuration.E

        self.reset()

    def reset(self) -> None:
        """Reset the simulation."""
        # TODO: this should be here?
        self.solver.current_frame[None] = 0
        self.solver.n_particles[None] = 0
        # self.current_frame = 0

        # self.sampler.reset()
        # self.solver.reset()

        # TODO: clean this up, maybe move to solver?
        # This resets the mpm solver:
        # Hidden particles are ignored in the solver, seeding
        # a particle will set the state 
        self.solver.state_p.fill(State.Hidden)
        # NOTE: setting this to something outside the simulation
        #       improves FPS and hides these particles
        self.solver.position_p.fill([42, 42])
        # self.solver.position_p.fill([0.48, 0.48])

        self.subsequent_geometries = self.configuration.subsequent_geometries.copy()

        # Load all the initial geometries into the solver:
        for geometry in self.configuration.initial_geometries:
            self.sampler.sample_geometry(10_000, geometry)

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
