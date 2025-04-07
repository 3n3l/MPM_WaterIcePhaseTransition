from src.enums import Conductivity, Phase, Color, State, Capacity
from src.mpm_solver import MPM_Solver, LATENT_HEAT
from src.configurations import Configuration

from abc import ABC, abstractmethod
from datetime import datetime

import taichi as ti
import os


@ti.data_oriented
class Renderer(ABC):
    def __init__(
        self,
        name: str,
        res: tuple[int, int] | int,
        solver: MPM_Solver,
        configurations: list[Configuration],
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
        self.should_write_to_disk = False
        self.is_showing_settings = not self.is_paused

        # Create a parent directory, more directories will be created inside this
        # directory that contain newly created frames, videos and GIFs.
        self.parent_dir = ".output"
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)

        # The MLS-MPM solver.
        self.solver = solver

        # Load the initial configuration and reset the solver to this configuration.
        self.configuration_id = 0
        self.configurations = configurations
        self.configuration = configurations[self.configuration_id]
        self.load_configuration(self.configuration)
        self.reset_solver(self.configuration)

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    @ti.kernel
    def load_configuration(self, configuration: ti.template()):  # pyright: ignore
        """
        Loads the chosen configuration into the MLS-MPM solver.
        ---
        Parameters:
            configuration: Configuration
        """
        self.solver.ambient_temperature[None] = configuration.ambient_temperature
        self.solver.n_particles[None] = configuration.n_particles
        self.solver.stickiness[None] = configuration.stickiness
        self.solver.friction[None] = configuration.friction
        self.solver.lambda_0[None] = configuration.lambda_0
        self.solver.theta_c[None] = configuration.theta_c
        self.solver.theta_s[None] = configuration.theta_s
        self.solver.zeta[None] = configuration.zeta
        self.solver.mu_0[None] = configuration.mu_0
        self.solver.nu[None] = configuration.nu
        self.solver.E[None] = configuration.E

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
            if p < configuration.n_particles:
                particle_is_water = configuration.phase_p[p] == Phase.Water
                self.solver.conductivity_p[p] = Conductivity.Water if particle_is_water else Conductivity.Ice
                self.solver.capacity_p[p] = Capacity.Water if particle_is_water else Capacity.Ice
                self.solver.color_p[p] = Color.Water if particle_is_water else Color.Ice
                self.solver.heat_p[p] = LATENT_HEAT if particle_is_water else 0.0

                self.solver.activation_threshold_p[p] = configuration.activity_threshold_p[p]
                self.solver.temperature_p[p] = configuration.temperature_p[p]
                self.solver.activation_state_p[p] = configuration.state_p[p]
                self.solver.velocity_p[p] = configuration.velocity_p[p]
                self.solver.phase_p[p] = configuration.phase_p[p]

                offset_position = configuration.position_p[p] + self.solver.boundary_offset
                p_is_active = configuration.state_p[p] == State.Active
                self.solver.active_position_p[p] = offset_position if p_is_active else [0, 0]
                self.solver.position_p[p] = offset_position
            else:
                # TODO: this might be completely irrelevant, as only the first n_particles are used anyway?
                #       So work can be saved by just ignoring all the other particles and iterating only
                #       over the configuration.n_particles?
                self.solver.activation_state_p[p] = State.Inactive
                self.solver.conductivity_p[p] = Conductivity.Zero
                self.solver.active_position_p[p] = [0, 0]
                self.solver.color_p[p] = Color.Background
                self.solver.capacity_p[p] = Capacity.Zero
                self.solver.activation_threshold_p[p] = 0
                self.solver.phase_p[p] = Phase.Water
                self.solver.position_p[p] = [0, 0]
                self.solver.velocity_p[p] = [0, 0]
                self.solver.temperature_p[p] = 0
                self.solver.heat_p[p] = 0
                self.solver.heat_p[p] = 0

            self.solver.mass_p[p] = self.solver.particle_vol * self.solver.rho_0
            self.solver.inv_lambda_p[p] = 1 / self.solver.lambda_0[None]
            self.solver.FE_p[p] = ti.Matrix([[1, 0], [0, 1]])
            self.solver.C_p[p] = ti.Matrix.zero(float, 2, 2)
            self.solver.JE_p[p] = 1
            self.solver.JP_p[p] = 1

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
