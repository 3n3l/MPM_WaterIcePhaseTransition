from typing import Callable
from src.configurations import Configuration
from src.samplers import PoissonDiskSampler
from src.renderer import BaseRenderer
from src.constants import ColorRGB
from src.solvers import MPM_Solver

import taichi as ti

class DrawOption:
    def __init__(self, name: str, is_active: bool, call_draw: Callable) -> None:
        self.name = name
        self.is_active = is_active
        self.call_draw = call_draw

@ti.data_oriented
class GGUI(BaseRenderer):
    def __init__(
        self,
        name: str,
        res: tuple[int, int],
        mpm_solver: MPM_Solver,
        configurations: list[Configuration],
        poisson_disk_sampler: PoissonDiskSampler,
        initial_configuration: int = 0,
    ) -> None:
        """Constructs a  GGUI renderer, this advances the MLS-MPM solver and renders the updated particle positions.
        ---
        Parameters:
            name: string displayed at the top of the window
            res: tuple holding window width and height
            solver: the MLS-MPM solver
            configurations: list of configurations for the solver
        """
        super().__init__(
            initial_configuration=initial_configuration,
            poisson_disk_sampler=poisson_disk_sampler,
            configurations=configurations,
            mpm_solver=mpm_solver,
        )

        # Foreground.
        self.should_show_temperature_p = False
        self.should_show_phase = True

        # self.show_temperature_p = DrawOption("Temperature", False, self._show_contour())
        # self.show_phase = DrawOption("Phase", True, self._render_particles)
        # self.foreground_options = [self.show_phase]

        # Background.
        self.should_show_classification = False
        self.should_show_temperature_c = False
        self.should_show_background = True

        # GGUI.
        self.window = ti.ui.Window(name, res, fps_limit=60)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.radius = 0.0015

    def show_configurations(self) -> None:
        """
        Show all possible configurations in the subwindow, choosing one will
        load that configuration and reset the solver.
        ---
        Parameters:
            subwindow: GGUI subwindow
        """
        prev_configuration_id = self.configuration_id
        with self.gui.sub_window("Configurations", 0.01, 0.01, 0.48, 0.49) as subwindow:
            for i in range(len(self.configurations)):
                name = self.configurations[i].name
                if subwindow.checkbox(name, self.configuration_id == i):
                    self.configuration_id = i
            if self.configuration_id != prev_configuration_id:
                _id = self.configuration_id
                configuration = self.configurations[_id]
                self.load_configuration(configuration)
                self.is_paused = True

    def show_foreground_options(self) -> None:
        """
        TODO: write this!
        """
        def reset_foreground_options():
            self.should_show_temperature_p = False
            self.should_show_phase = False

        with self.gui.sub_window("Foreground", 0.5, 0.01, 0.49, 0.24) as subwindow:
            if subwindow.checkbox("Phase", self.should_show_phase):
                reset_foreground_options()
                self.should_show_phase = True

            if subwindow.checkbox("Temperature", self.should_show_temperature_p):
                reset_foreground_options()
                self.should_show_temperature_p = True

        # TODO: show more information on particles

    def show_background_options(self) -> None:
        """
        TODO: write this!
        """
        def reset_background_options():
            self.should_show_classification = False
            self.should_show_temperature_c = False
            self.should_show_background = False

        with self.gui.sub_window("Background", 0.5, 0.26, 0.49, 0.24) as subwindow:
            if subwindow.checkbox("Background", self.should_show_background):
                reset_background_options()
                self.should_show_background = True

            if subwindow.checkbox("Classification", self.should_show_classification):
                reset_background_options()
                self.should_show_classification = True

            if subwindow.checkbox("Temperature", self.should_show_temperature_c):
                reset_background_options()
                self.should_show_temperature_c = True

    def show_parameters(self) -> None:
        """
        Show all parameters in the subwindow, the user can then adjust these values
        with sliders which will update the correspoding value in the solver.
        """
        # with self.gui.sub_window("Parameters", 0.01, 0.51, 0.98, 0.24) as subwindow:
        with self.gui.sub_window("Parameters", 0.01, 0.51, 0.98, 0.48) as subwindow:
            self.mpm_solver.theta_c[None] = subwindow.slider_float("theta_c", self.mpm_solver.theta_c[None], 1e-2, 10e-2)
            self.mpm_solver.theta_s[None] = subwindow.slider_float("theta_s", self.mpm_solver.theta_s[None], 1e-3, 10e-3)
            self.mpm_solver.zeta[None] = subwindow.slider_int("zeta", self.mpm_solver.zeta[None], 3, 20)
            self.mpm_solver.nu[None] = subwindow.slider_float("nu", self.mpm_solver.nu[None], 0.1, 0.4)
            self.mpm_solver.E[None] = subwindow.slider_float("E", self.mpm_solver.E[None], 4.8e4, 5.5e5)

            if subwindow.button(" Stop recording  " if self.should_write_to_disk else " Start recording "):
                # This button toggles between saving frames and not saving frames.
                self.should_write_to_disk = not self.should_write_to_disk
                if self.should_write_to_disk:
                    self.dump_frames()
                else:
                    self.create_video()
            if subwindow.button(" Reset Particles "):
                self.reset()
            if subwindow.button(" Start Simulation"):
                self.is_paused = False

        E = self.mpm_solver.E[None]
        nu = self.mpm_solver.nu[None]
        self.mpm_solver.lambda_0[None] = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mpm_solver.mu_0[None] = E / (2 * (1 + nu))

    def show_buttons(self) -> None:
        """
        Show a set of buttons in the subwindow, this mainly holds functions to control the simulation.
        ---
        Parameters:
            subwindow: GGUI subwindow
        """
        with self.gui.sub_window("AAAAAAAAAAA", 0.01, 0.76, 0.48, 0.23) as subwindow:
            if subwindow.button(" Stop recording  " if self.should_write_to_disk else " Start recording "):
                # This button toggles between saving frames and not saving frames.
                self.should_write_to_disk = not self.should_write_to_disk
                if self.should_write_to_disk:
                    self.dump_frames()
                else:
                    self.create_video()
            if subwindow.button(" Reset Particles "):
                self.reset()
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
        # with self.gui.sub_window("Settings", 0.01, 0.8, 0.98, 0.48) as subwindow:
        self.show_parameters()
        self.show_configurations()
        self.show_foreground_options()
        self.show_background_options()
        # self.show_buttons()

    def handle_events(self) -> None:
        """Handle key presses arising from window events."""
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.reset()
            elif self.window.event.key in [ti.GUI.BACKSPACE, "s"]:
                self.should_write_to_disk = not self.should_write_to_disk
            elif self.window.event.key in [ti.GUI.SPACE, "p"]:
                self.is_paused = not self.is_paused
            elif self.window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                self.window.running = False  # Stop the simulation

    def _render_particles(self, per_vertex_color) -> None:
        self.canvas.circles(
            per_vertex_color=per_vertex_color,
            centers=self.mpm_solver.position_p,
            radius=self.radius,
        )

    def _show_contour(self, scalar_field) -> None:
        self.canvas.contour(scalar_field, cmap_name="magma", normalize=True)

    def _show_vector_field(self, vector_field) -> None:
        self.canvas.vector_field(vector_field)

    def render(self) -> None:
        """Renders the simulation with the data from the MLS-MPM solver."""

        # Background.
        if self.should_show_background:
            self.canvas.set_background_color(ColorRGB.Background)
        elif self.should_show_classification:
            self._show_contour(self.mpm_solver.classification_c)
        elif self.should_show_temperature_c:
            self._show_contour(self.mpm_solver.temperature_c)

        # Foreground.
        if self.should_show_phase:
            self._render_particles(per_vertex_color=self.mpm_solver.color_p)
        elif self.should_show_temperature_p:
            self._render_particles(per_vertex_color=self.mpm_solver.color_p)
            # TODO: need to convert temperature to colormap first
            # self._render_particles(per_vertex_color=self.mpm_solver.temperature_p_p)

        if self.should_write_to_disk and not self.is_paused and not self.is_showing_settings:
            self.video_manager.write_frame(self.window.get_image_buffer_as_numpy())
        self.window.show()

    def run(self) -> None:
        """Runs this simulation."""
        # iteration = 0
        while self.window.running:
            # if iteration == 1200:
            #     self.create_video()
            #     self.window.running = False
            self.handle_events()
            self.show_settings()
            if not self.is_paused:
                self.substep()
                # iteration += 1
            self.render()
