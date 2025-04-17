from src.configurations import Configuration
from src.samplers import PoissonDiskSampler
from src.constants import Phase, Color
from src.renderer import BaseRenderer
from src.solvers import MPM_Solver

import taichi as ti


@ti.data_oriented
class GUI(BaseRenderer):
    def __init__(
        self,
        name: str,
        res: int,
        mpm_solver: MPM_Solver,
        configuration: Configuration,
        poisson_disk_sampler: PoissonDiskSampler,
    ) -> None:
        """Constructs a  GUI renderer, this advances the MLS-MPM solver and renders the updated particle positions.
        ---
        Parameters:
            name: string displayed at the top of the window
            res: tuple holding window width and height
            solver: the MLS-MPM solver
            configuration: the one configuration for the solver
        """
        super().__init__(
            poisson_disk_sampler=poisson_disk_sampler,
            configurations=[configuration],
            mpm_solver=mpm_solver,
        )

        # GUI.
        self.gui = ti.GUI(name, res=res, background_color=Color._Background)

    def render(self) -> None:
        """Renders the simulation with the data from the MLS-MPM solver."""
        # TODO: write frames to disk?
        # TODO: colors?
        indices = [0 if p == Phase.Ice else 1 for p in self.mpm_solver.phase_p.to_numpy()]
        position = self.mpm_solver.position_p.to_numpy()
        palette = [Color._Ice, Color._Water]
        radius = 1.5
        self.gui.circles(position, radius, palette=palette, palette_indices=indices)  # pyright: ignore
        self.gui.show()  # change to gui.show(f'{frame:06d}.png') to write images to disk

    def run(self) -> None:
        """Runs this simulation."""
        while self.gui.running:
            if self.gui.get_event(ti.GUI.PRESS):
                if self.gui.event.key == "r":  # pyright: ignore
                    self.reset()
                elif self.gui.event.key == ti.GUI.SPACE:  # pyright: ignore
                    self.is_paused = not self.is_paused
                elif self.gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:  # pyright: ignore
                    break
            if not self.is_paused:
                self.substep()
            self.render()
