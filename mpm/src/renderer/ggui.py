from src.configurations import Configuration
from src.samplers import PoissonDiskSampler
from src.constants import ColorRGB, State
from src.renderer import BaseRenderer
from src.solvers import MPM_Solver

from typing import Callable

import taichi as ti


class DrawingOption:
    """
    This holds name, state and a callable for drawing a chosen foreground/background.
    """

    def __init__(self, name: str, is_active: bool, call_draw: Callable) -> None:
        self.is_active = is_active
        self.draw = call_draw
        self.name = name


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
        super().__init__(
            initial_configuration=initial_configuration,
            poisson_disk_sampler=poisson_disk_sampler,
            configurations=configurations,
            mpm_solver=mpm_solver,
        )

        # GGUI.
        self.window = ti.ui.Window(name, res, fps_limit=60)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.radius = 0.0015

        # Fields that hold certain colors, must be update in each draw call.
        self.temperature_colors_p = ti.Vector.field(3, dtype=ti.f32, shape=self.mpm_solver.max_particles)
        # TODO: also move the phase colors here, then only update the phase colors when drawing the phase?!

        # Construct a vector field as a heat map:
        self.heat_map_length = len(ColorRGB.HeatMap)
        self.heat_map = ti.Vector.field(3, dtype=ti.f32, shape=self.heat_map_length)
        for i, color in enumerate(ColorRGB.HeatMap):
            self.heat_map[i] = color

        # Values to control the drawing of the temperature:
        # TODO: these should be moved somewhere else
        self.should_normalize_temperature = False
        self.min_temperature = -100  # TODO: this should be temperature boundary of simulation
        self.max_temperature = 100  # TODO: this should be temperature boundary of simulation

        # Foreground Options:
        self.foreground_options = [
            DrawingOption("Temperature", False, self.draw_temperature_p),
            DrawingOption("Nothing", False, lambda: None),
            DrawingOption("Phase", True, self.draw_phase_p),
        ]

        # Background Options:
        self.background_options = [
            DrawingOption("Classification", False, lambda: self.show_contour(self.mpm_solver.classification_c)),
            DrawingOption("Temperature", False, lambda: self.show_contour(self.mpm_solver.temperature_c)),
            DrawingOption("Background", True, lambda: self.canvas.set_background_color(ColorRGB.Background)),
            DrawingOption("Mass", False, lambda: self.show_contour(self.mpm_solver.mass_c)),
        ]

    def show_configurations(self) -> None:
        """
        Show all possible configurations inside own subwindow, choosing one will
        load that configuration and reset the solver.
        """
        prev_configuration_id = self.configuration_id
        with self.gui.sub_window("Configurations", 0.01, 0.01, 0.48, 0.74) as subwindow:
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
        Show the foreground drawing options as checkboxes inside own subwindow.
        """
        with self.gui.sub_window("Foreground", 0.5, 0.01, 0.49, 0.24) as subwindow:
            for option in self.foreground_options:
                if subwindow.checkbox(option.name, option.is_active):
                    for _option in self.foreground_options:
                        _option.is_active = False
                    option.is_active = True

    def show_background_options(self) -> None:
        """
        Show the background drawing options as checkboxes inside own subwindow.
        """
        with self.gui.sub_window("Background", 0.5, 0.26, 0.49, 0.24) as subwindow:
            for option in self.background_options:
                if subwindow.checkbox(option.name, option.is_active):
                    for _option in self.background_options:
                        _option.is_active = False
                    option.is_active = True

    def show_parameters(self) -> None:
        """
        Show all parameters in the subwindow, the user can then adjust these values
        with sliders which will update the correspoding value in the solver.
        """
        with self.gui.sub_window("Parameters", 0.01, 0.76, 0.98, 0.23) as subwindow:
            # with self.gui.sub_window("Parameters", 0.01, 0.51, 0.98, 0.48) as subwindow:
            self.mpm_solver.theta_c[None] = subwindow.slider_float("theta_c", self.mpm_solver.theta_c[None], 1e-2, 10e-2)
            self.mpm_solver.theta_s[None] = subwindow.slider_float("theta_s", self.mpm_solver.theta_s[None], 1e-3, 10e-3)
            self.mpm_solver.zeta[None] = subwindow.slider_int("zeta", self.mpm_solver.zeta[None], 3, 20)
            self.mpm_solver.nu[None] = subwindow.slider_float("nu", self.mpm_solver.nu[None], 0.1, 0.4)
            self.mpm_solver.E[None] = subwindow.slider_float("E", self.mpm_solver.E[None], 4.8e4, 5.5e5)

        # TODO: E, nu, lamba and mu are not used right now.
        E = self.mpm_solver.E[None]
        nu = self.mpm_solver.nu[None]
        self.mpm_solver.lambda_0[None] = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mpm_solver.mu_0[None] = E / (2 * (1 + nu))

    def show_buttons(self) -> None:
        """
        Show a set of buttons in the subwindow, this mainly holds functions to control the simulation.
        """
        with self.gui.sub_window("Settings", 0.5, 0.51, 0.49, 0.24) as subwindow:
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
        if not self.is_paused or not self.should_show_settings:
            self.is_showing_settings = False
            return  # don't bother

        self.is_showing_settings = True
        self.show_foreground_options()
        self.show_background_options()
        self.show_configurations()
        self.show_parameters()
        self.show_buttons()

    def handle_events(self) -> None:
        """
        Handle key presses arising from window events.
        """
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.reset()
            elif self.window.event.key in ["h"]:
                self.should_show_settings = not self.should_show_settings
            elif self.window.event.key in [ti.GUI.BACKSPACE, "s"]:
                self.should_write_to_disk = not self.should_write_to_disk
            elif self.window.event.key in [ti.GUI.SPACE, "p"]:
                self.is_paused = not self.is_paused
            elif self.window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                self.window.running = False  # Stop the simulation

    @ti.kernel
    def update_temperature_p(self):
        max_temperature = self.max_temperature
        min_temperature = self.min_temperature

        # Get min, max values for normalization:
        if self.should_normalize_temperature:
            max_temperature, min_temperature = -1e32, 1e32
            for p in self.temperature_colors_p:
                if self.mpm_solver.state_p[p] == State.Hidden:
                    continue  # ignore uninitialized particles
                temperature = self.mpm_solver.temperature_p[p]
                if temperature > max_temperature:
                    max_temperature = temperature
                elif temperature < min_temperature:
                    min_temperature = temperature

        # Affine combination of the colors based on min, max values and the temperature:
        max_index, min_index = self.heat_map_length - 1, 0
        factor = ti.cast(max_index, ti.f32)
        for p in self.temperature_colors_p:
            if self.mpm_solver.state_p[p] == State.Hidden:
                continue  # ignore uninitialized particles
            t = self.mpm_solver.temperature_p[p]
            a = (t - min_temperature) / (max_temperature - min_temperature)
            color1 = self.heat_map[ti.max(min_index, ti.floor(factor * a, ti.i8))]
            color2 = self.heat_map[ti.min(max_index, ti.ceil(factor * a, ti.i8))]
            self.temperature_colors_p[p] = ((1 - a) * color1) + (a * color2)

    def draw_temperature_p(self) -> None:
        """
        Draw the temperature for each particle.
        """
        self.update_temperature_p()
        self.canvas.circles(
            per_vertex_color=self.temperature_colors_p,
            centers=self.mpm_solver.position_p,
            radius=self.radius,
        )

    def draw_phase_p(self,) -> None:
        """
        Draw the phase for each particle.
        """
        self.canvas.circles(
            per_vertex_color=self.mpm_solver.color_p,
            centers=self.mpm_solver.position_p,
            radius=self.radius,
        )

    def show_contour(self, scalar_field) -> None:
        """
        Show the contour of a given scalar field.
        """
        self.canvas.contour(scalar_field, cmap_name="magma", normalize=True)

    def render(self) -> None:
        """
        Renders the simulation with the data from the MLS-MPM solver.
        """
        # Draw chosen foreground/brackground, NOTE: foreground must be drawn last.
        for option in self.background_options + self.foreground_options:
            if option.is_active:
                option.draw()

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

            self.render()
