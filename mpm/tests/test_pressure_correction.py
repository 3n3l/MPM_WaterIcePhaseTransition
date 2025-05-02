import utils  # import first to append parent directory to path

from src.configurations import Configuration, Rectangle
from src.samplers import PoissonDiskSampler

# from src.presets import configuration_list
from src.constants import Classification, Phase
from src.renderer import BaseRenderer
from src.solvers import MPM_Solver
from src.parsing import arguments

import taichi as ti
import numpy as np

offset = 0.0234375

configuration_list = [
    Configuration(
        name="Dam Break [Water]",
        geometries=[
            Rectangle(
                lower_left=(offset, offset),
                phase=Phase.Water,
                temperature=20.0,
                size=(0.5 - offset, 0.5 - offset),
                velocity=(0, 0),
            ),
        ],
        E=5e5,  # Young's modulus (1.4e5)
        nu=0.45,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=20.0,
    ),
]


class TestRenderer(BaseRenderer):
    def __init__(
        self,
        solver: MPM_Solver,
        configurations: list[Configuration],
        poisson_disk_sampler: PoissonDiskSampler,
    ) -> None:
        super().__init__(mpm_solver=solver, configurations=configurations, poisson_disk_sampler=poisson_disk_sampler)
        self.divergence = ti.ndarray(ti.f32, shape=(solver.n_grid, solver.n_grid))
        self.we_succeeded = True

    @ti.kernel
    def compute_divergence(self, div: ti.types.ndarray()):  # pyright: ignore
        for i, j in self.mpm_solver.mass_c:
            div[i, j] = 0
            if self.mpm_solver.is_interior(i, j):
                div[i, j] += self.mpm_solver.velocity_x[i + 1, j]
                div[i, j] -= self.mpm_solver.velocity_x[i, j]
                div[i, j] += self.mpm_solver.velocity_y[i, j + 1]
                div[i, j] -= self.mpm_solver.velocity_y[i, j]

    def run(self) -> None:
        for i in range(1, 301):
            self.substep()
            prev_min = np.round(np.abs(np.min(self.divergence.to_numpy())))
            prev_max = np.round(np.abs(np.max(self.divergence.to_numpy())))
            self.compute_divergence(self.divergence)
            curr_min = np.round(np.abs(np.min(self.divergence.to_numpy())))
            curr_max = np.round(np.abs(np.max(self.divergence.to_numpy())))

            print(".", end=("\n" if i % 10 == 0 else " "), flush=True)

            if curr_min > prev_min or curr_max > prev_max:
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
