from src.configurations import Configuration
from src.renderer.renderer import Renderer
from src.mpm_solver import MPM_Solver
from src.enums import Color

import taichi as ti


@ti.data_oriented
class GGUI_Renderer(Renderer):
    def __init__(
        self,
        name: str,
        res: tuple[int, int],
        solver: MPM_Solver,
        configurations: list[Configuration],
    ) -> None:
        """Constructs a  GGUI renderer, this advances the MLS-MPM solver and renders the updated particle positions.
        ---
        Parameters:
            name: string displayed at the top of the window
            res: tuple holding window width and height
            solver: the MLS-MPM solver
            configurations: list of configurations for the solver
        """
        super().__init__(name, res, solver, configurations)

        # GGUI.
        self.window = ti.ui.Window(name, res)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()

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

    def render(self) -> None:
        """Renders the simulation with the data from the MLS-MPM solver."""
        self.canvas.set_background_color(Color.Background)
        self.canvas.circles(
            per_vertex_color=self.solver.color_p,
            centers=self.solver.active_position_p,
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
