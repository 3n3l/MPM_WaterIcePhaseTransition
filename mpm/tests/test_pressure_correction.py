import utils  # import first to append parent directory to path

from src.configurations import Configuration
from src.samplers import PoissonDiskSampler
from src.presets import configuration_list
from src.constants import Classification
from src.renderer import BaseRenderer
from src.solvers import MPM_Solver
from src.parsing import arguments

import taichi as ti
import numpy as np


class TestRenderer(BaseRenderer):
    def __init__(
        self,
        solver: MPM_Solver,
        configurations: list[Configuration],
        poisson_disk_sampler: PoissonDiskSampler,
        initial_configuration: int = 0,
    ) -> None:
        super().__init__(mpm_solver=solver, configurations=configurations, poisson_disk_sampler=poisson_disk_sampler)
        self.divergence = ti.ndarray(ti.f32, shape=(solver.n_grid, solver.n_grid))
        self.we_succeeded = True

    @ti.kernel
    def compute_divergence(self, div: ti.types.ndarray()):  # pyright: ignore
        for i, j in self.mpm_solver.pressure_c:
            div[i, j] = 0
            if self.mpm_solver.classification_c[i, j] == Classification.Interior:
                div[i, j] += self.mpm_solver.velocity_x[i + 1, j] - self.mpm_solver.velocity_x[i, j]
                div[i, j] += self.mpm_solver.velocity_y[i, j + 1] - self.mpm_solver.velocity_y[i, j]

    def run(self) -> None:
        for i in range(1, 501):
            self.substep()
            prev_min = np.abs(np.min(self.divergence.to_numpy()))
            prev_max = np.abs(np.max(self.divergence.to_numpy()))
            self.compute_divergence(self.divergence)
            curr_min = np.abs(np.min(self.divergence.to_numpy()))
            curr_max = np.abs(np.max(self.divergence.to_numpy()))

            print(".", end=("\n" if i % 10 == 0 else " "), flush=True)

            if np.round(curr_min) > np.round(prev_min) or np.round(curr_max) > np.round(prev_max):
                # The solver actually increased the divergence :(
                print("\n\nDivergence increased :(")
                print(f"prev_min = {prev_min}, prev_max = {prev_max}")
                print(f"curr_min = {curr_min}, curr_max = {curr_max}")
                self.we_succeeded = False
                break

            if np.any(np.round(self.divergence.to_numpy(), 2) != 0):  # pyright: ignore
                print("\n\nDivergence too big :(")
                self.we_succeeded = False
                break

            if not self.we_succeeded:
                break


def main() -> None:
    # Initialize Taichi on the chosen architecture:
    if arguments.arch.lower() == "cpu":
        ti.init(arch=ti.cpu, debug=arguments.debug)
    elif arguments.arch.lower() == "gpu":
        ti.init(arch=ti.gpu, debug=arguments.debug)
    else:
        ti.init(arch=ti.cuda, debug=arguments.debug)

    max_particles = 100_000
    solver = MPM_Solver(quality=1, max_particles=max_particles)
    poisson_disk_sampler = PoissonDiskSampler(mpm_solver=solver)
    test_renderer = TestRenderer(
        initial_configuration=arguments.configuration,
        poisson_disk_sampler=poisson_disk_sampler,
        configurations=configuration_list,
        solver=solver,
    )

    for configuration in configuration_list:
        print(f"NOW RUNNING: {configuration.name}")
        test_renderer.load_configuration(configuration)
        test_renderer.run()
        if not test_renderer.we_succeeded:
            break

    print("\n")
    print("Divergence, min ->", np.min(test_renderer.divergence.to_numpy()))
    print("Divergence, max ->", np.max(test_renderer.divergence.to_numpy()))
    print()

    print("\033[92m:)))))))))\033[0m" if test_renderer.we_succeeded else "\033[91m:(\033[0m")


if __name__ == "__main__":
    main()
