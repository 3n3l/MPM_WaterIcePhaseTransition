from Configuration import Configuration
from Solver import Solver
from datetime import datetime
import taichi as ti
import os


class Renderer:
    def __init__(
        self,
        solver: Solver,
        configurations: list[Configuration],
    ) -> None:
        # TODO: make a list of solvers?!
        self.solver = solver

        # Parameters to control the simulation
        self.window = ti.ui.Window(name="~TBD~", res=(720, 720), fps_limit=60)
        self.gui = self.window.get_gui()
        self.canvas = self.window.get_canvas()
        self.is_paused = True
        self.should_write_to_disk = False
        self.is_showing_settings = not self.is_paused
        # Create a parent directory, in there folders will be created containing
        # newly created frames, videos and GIFs.
        self.parent_dir = ".output"
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)

        # Load the initial configuration
        self.configuration_id = 0
        self.configurations = configurations
        self.model = configurations[self.configuration_id]
        self.load_configuration()

    def reset(self):
        self.frame = 0
        self.directory = datetime.now().strftime("%d%m%Y_%H%M")
        os.makedirs(f".output/{self.directory}")
        self.solver.reset(self.model.n_particles)

    def load_configuration(self):
        self.solver.initial_position = self.model.position
        self.solver.initial_velocity = self.model.velocity
        self.solver.initial_phase = self.model.phase
        self.solver.stickiness[None] = self.model.stickiness
        self.solver.friction[None] = self.model.friction
        self.solver.lambda_0[None] = self.model.lambda_0
        self.solver.theta_c[None] = self.model.theta_c
        self.solver.theta_s[None] = self.model.theta_s
        self.solver.zeta[None] = self.model.zeta
        self.solver.mu_0[None] = self.model.mu_0
        self.solver.nu[None] = self.model.nu
        self.solver.E[None] = self.model.E

    def handle_events(self):
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.solver.reset(self.model.n_particles)
            elif self.window.event.key in [ti.GUI.BACKSPACE, "s"]:
                self.should_write_to_disk = not self.should_write_to_disk
            elif self.window.event.key in [ti.GUI.SPACE, "p"]:
                self.is_paused = not self.is_paused
            elif self.window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                self.window.running = False  # Stop the simulation

    def show_configurations(self, subwindow):
        prev_configuration_id = self.configuration_id
        for i in range(len(self.configurations)):
            name = self.configurations[i].name
            if subwindow.checkbox(name, self.configuration_id == i):
                self.configuration_id = i
        if self.configuration_id != prev_configuration_id:
            _id = self.configuration_id
            self.model = self.configurations[_id]
            self.load_configuration()
            self.is_paused = True
            self.solver.reset(self.model.n_particles)

    def show_parameters(self, subwindow):
        self.solver.stickiness[None] = subwindow.slider_float("stickiness", self.solver.stickiness[None], 1.0, 5.0)
        self.solver.friction[None] = subwindow.slider_float("friction", self.solver.friction[None], 1.0, 5.0)
        self.solver.theta_c[None] = subwindow.slider_float("theta_c", self.solver.theta_c[None], 1e-2, 3.5e-2)
        self.solver.theta_s[None] = subwindow.slider_float("theta_s", self.solver.theta_s[None], 5.0e-3, 10e-3)
        self.solver.zeta[None] = subwindow.slider_int("zeta", self.solver.zeta[None], 3, 10)
        self.solver.nu[None] = subwindow.slider_float("nu", self.solver.nu[None], 0.1, 0.4)
        self.solver.E[None] = subwindow.slider_float("E", self.solver.E[None], 4.8e4, 2.8e5)
        E = self.solver.E[None]
        nu = self.solver.nu[None]
        self.solver.lambda_0[None] = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.solver.mu_0[None] = E / (2 * (1 + nu))

    def show_buttons(self, subwindow):
        if subwindow.button(" Stop recording  " if self.should_write_to_disk else " Start recording "):
            # This button toggles between saving frames and not saving frames.
            self.should_write_to_disk = not self.should_write_to_disk
            if self.should_write_to_disk:
                # Create directory to dump frames, videos and GIFs.
                date = datetime.now().strftime("%d%m%Y_%H%M")
                output_dir = f"{self.parent_dir}/{date}"
                os.makedirs(output_dir)
                # Create a VideoManager to save frames, videos and GIFs.
                self.video_manager = ti.tools.VideoManager(
                    output_dir=output_dir,
                    framerate=24,
                    automatic_build=False,
                )
            else:
                # Convert stored frames to video and GIF.
                self.video_manager.make_video(gif=True, mp4=True)
        if subwindow.button(" Reset Particles "):
            self.reset()
        if subwindow.button(" Start Simulation"):
            self.is_paused = False

    def show_settings(self):
        if not self.is_paused:
            self.is_showing_settings = False
            return  # don't bother
        self.is_showing_settings = True
        with self.gui.sub_window("Settings", 0.01, 0.01, 0.98, 0.98) as subwindow:
            self.show_parameters(subwindow)
            self.show_configurations(subwindow)
            self.show_buttons(subwindow)

    def render(self):
        self.canvas.set_background_color((0.054, 0.06, 0.09))
        self.canvas.circles(
            centers=self.solver.particle_position, radius=0.0015, per_vertex_color=self.solver.particle_color
        )
        if self.should_write_to_disk and not self.is_paused and not self.is_showing_settings:
            self.video_manager.write_frame(self.window.get_image_buffer_as_numpy())
        self.window.show()

    def run(self):
        self.solver.reset(self.model.n_particles)
        while self.window.running:
            self.handle_events()
            self.show_settings()
            if not self.is_paused:
                self.solver.substep()
            self.render()
