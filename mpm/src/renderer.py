from src.configurations import Configuration
from src.enums import Phase, Color, State
from src.mpm_solver import MPM_Solver
from datetime import datetime

import taichi as ti
import os


@ti.data_oriented
class Renderer:
    def __init__(
        self, name: str, res: tuple[int, int], solver: MPM_Solver, configurations: list[Configuration]
    ) -> None:
        """Constructs a Renderer object, this advances the MLS-MPM solver and renders the updated particle positions.
        ---
        Parameters:
            name: string displayed at the top of the window
            res: tuple holding window width and height
            solver: the MLS-MPM solver
            configurations: list of configurations for the solver
        """
        # GGUI.
        self.window = ti.ui.Window(name, res)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()

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

    @ti.kernel
    def load_configuration(self, configuration: ti.template()):  # pyright: ignore
        """
        Loads the chosen configuration into the MLS-MPM solver.
        ---
        Parameters:
            configuration: Configuration
        """
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
        for p in self.solver.particle_position:
            if p < configuration.n_particles:
                self.solver.particle_color[p] = Color.Water if configuration.p_phase[p] == Phase.Water else Color.Ice
                self.solver.p_activation_threshold[p] = configuration.p_activity_bound[p]
                self.solver.particle_position[p] = configuration.p_position[p] + self.solver.boundary_offset
                self.solver.particle_velocity[p] = configuration.p_velocity[p]
                self.solver.p_activation_state[p] = configuration.p_state[p]
                self.solver.particle_phase[p] = configuration.p_phase[p]
            else:
                # TODO: this might be completely irrelevant, as only the first n_particles are used anyway?
                #       So work can be saved by just ignoring all the other particles and iterating only
                #       over the configuration.n_particles?
                self.solver.particle_color[p] = Color.Background
                self.solver.p_activation_threshold[p] = 0
                self.solver.particle_position[p] = [0, 0]
                self.solver.particle_velocity[p] = [0, 0]
                self.solver.p_activation_state[p] = State.Inactive
                self.solver.particle_phase[p] = Phase.Water

            self.solver.particle_mass[p] = self.solver.particle_vol * self.solver.rho_0
            self.solver.particle_inv_lambda[p] = 1 / self.solver.lambda_0[None]
            self.solver.particle_FE[p] = ti.Matrix([[1, 0], [0, 1]])
            self.solver.particle_C[p] = ti.Matrix.zero(float, 2, 2)
            self.solver.p_active_position[p] = [0, 0]
            self.solver.particle_JE[p] = 1
            self.solver.particle_JP[p] = 1

    def handle_events(self) -> None:
        """Handle key presses arising from window events."""
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.reset_solver(self.configuration)
            elif self.window.event.key in [ti.GUI.BACKSPACE, "s"]:
                self.should_write_to_disk = not self.should_write_to_disk
            elif self.window.event.key in [ti.GUI.SPACE, "p"]:
                self.is_paused = not self.is_paused
            elif self.window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                self.window.running = False  # Stop the simulation

    def show_configurations(self, subwindow) -> None:
        """
        Show all possible configurations in the subwindow, choosing one will
        load that configuration and reset the solver.
        ---
        Parameters:
            subwindow: GGUI subwindow
        """
        prev_configuration_id = self.configuration_id
        for i in range(len(self.configurations)):
            name = self.configurations[i].name
            if subwindow.checkbox(name, self.configuration_id == i):
                self.configuration_id = i
        if self.configuration_id != prev_configuration_id:
            _id = self.configuration_id
            self.configuration = self.configurations[_id]
            self.is_paused = True
            self.load_configuration(self.configuration)
            self.reset_solver(self.configuration)

    def show_parameters(self, subwindow) -> None:
        """
        Show all parameters in the subwindow, the user can then adjust these values
        with sliders which will update the correspoding value in the solver.
        ---
        Parameters:
            subwindow: GGUI subwindow
        """
        # TODO: Implement back stickiness + friction or remove them entirely
        # self.solver.stickiness[None] = subwindow.slider_float("stickiness", self.solver.stickiness[None], 1.0, 5.0)
        # self.solver.friction[None] = subwindow.slider_float("friction", self.solver.friction[None], 1.0, 5.0)
        self.solver.theta_c[None] = subwindow.slider_float("theta_c", self.solver.theta_c[None], 1e-2, 10e-2)
        self.solver.theta_s[None] = subwindow.slider_float("theta_s", self.solver.theta_s[None], 1e-3, 10e-3)
        self.solver.zeta[None] = subwindow.slider_int("zeta", self.solver.zeta[None], 3, 20)
        self.solver.nu[None] = subwindow.slider_float("nu", self.solver.nu[None], 0.1, 0.4)
        self.solver.E[None] = subwindow.slider_float("E", self.solver.E[None], 4.8e4, 2.8e5)
        E = self.solver.E[None]
        nu = self.solver.nu[None]
        self.solver.lambda_0[None] = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.solver.mu_0[None] = E / (2 * (1 + nu))

    def show_buttons(self, subwindow) -> None:
        """
        Show a set of buttons in the subwindow, this mainly holds functions to control the simulation.
        ---
        Parameters:
            subwindow: GGUI subwindow
        """
        if subwindow.button(" Stop recording  " if self.should_write_to_disk else " Start recording "):
            # This button toggles between saving frames and not saving frames.
            self.should_write_to_disk = not self.should_write_to_disk
            if self.should_write_to_disk:
                self.dump_frames()
            else:
                self.create_video()
        if subwindow.button(" Reset Particles "):
            self.reset_solver(self.configuration)
        if subwindow.button(" Start Simulation"):
            self.is_paused = False

    def show_settings(self) -> None:
        """
        Show settings in a GGUI subwindow, this should be called once per generated frames
        and will only show these settings if the simulation is paused at the moment.
        """
        if not self.is_paused:
            self.is_showing_settings = False
            return  # don't bother
        self.is_showing_settings = True
        with self.gui.sub_window("Settings", 0.01, 0.01, 0.98, 0.98) as subwindow:
            self.show_parameters(subwindow)
            self.show_configurations(subwindow)
            self.show_buttons(subwindow)

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

    def render(self) -> None:
        """Renders the simulation with the data from the MLS-MPM solver."""
        self.canvas.set_background_color(Color.Background)
        self.canvas.circles(
            per_vertex_color=self.solver.particle_color,
            centers=self.solver.p_active_position,
            radius=0.0015,
        )
        if self.should_write_to_disk and not self.is_paused and not self.is_showing_settings:
            self.video_manager.write_frame(self.window.get_image_buffer_as_numpy())
        self.window.show()

    def run(self) -> None:
        """Runs this simulation."""
        while self.window.running:
            self.handle_events()
            self.show_settings()
            if not self.is_paused:
                self.solver.substep()
            self.render()
