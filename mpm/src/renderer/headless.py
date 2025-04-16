from abc import abstractmethod
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
        poisson_disk_sampler: PoissonDiskSampler,
        configurations: list[Configuration],
        mpm_solver: MPM_Solver,
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
        self.mpm_solver = mpm_solver
        self.poisson_disk_sampler = poisson_disk_sampler

        # Load the initial configuration and reset the solver to this configuration.
        self.current_frame = 0
        self.configuration_id = 0
        self.configurations = configurations
        self.load_configuration(configurations[self.configuration_id])

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    def substep(self) -> None:
        self.current_frame += 1

        # Load all remaining geometries with a satisfied frame threshold:
        if len(self.subsequent_geometries) > 0:
            if self.current_frame == self.subsequent_geometries[0].frame_threshold:
                geometry = self.subsequent_geometries.pop(0)
                self.poisson_disk_sampler.sample_geometry(geometry)

        for _ in range(int(2e-3 // self.mpm_solver.dt)):
            self.mpm_solver.reset_grids()
            self.mpm_solver.particle_to_grid()
            self.mpm_solver.momentum_to_velocity()
            self.mpm_solver.classify_cells()
            self.mpm_solver.compute_volumes()
            self.mpm_solver.pressure_solver.solve()
            self.mpm_solver.heat_solver.solve()
            self.mpm_solver.grid_to_particle()

    def load_configuration(self, configuration: Configuration) -> None:
        """
        Loads the chosen configuration into the MLS-MPM solver.
        ---
        Parameters:
            configuration: Configuration
        """
        self.mpm_solver.ambient_temperature[None] = configuration.ambient_temperature
        self.mpm_solver.lambda_0[None] = configuration.lambda_0
        self.mpm_solver.theta_c[None] = configuration.theta_c
        self.mpm_solver.theta_s[None] = configuration.theta_s
        self.mpm_solver.zeta[None] = configuration.zeta
        self.mpm_solver.mu_0[None] = configuration.mu_0
        self.mpm_solver.nu[None] = configuration.nu
        self.mpm_solver.E[None] = configuration.E
        self.configuration = configuration
        self.reset()

    def reset(self) -> None:
        """Reset the simulation."""

        # Reset the MPM solver:
        self.mpm_solver.state_p.fill(State.Hidden)
        self.mpm_solver.position_p.fill([42, 42])
        self.mpm_solver.n_particles[None] = 0
        self.current_frame = 0

        # We copy this, so we can pop from this list and check the length:
        self.subsequent_geometries = self.configuration.subsequent_geometries.copy()

        # Load all the initial geometries into the solver:
        for geometry in self.configuration.initial_geometries:
            self.poisson_disk_sampler.sample_geometry(geometry)

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
