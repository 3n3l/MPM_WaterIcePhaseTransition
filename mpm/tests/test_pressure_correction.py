import utils  # import first to append parent directory to path

from src.configurations import Configuration, Rectangle, Circle
from src.samplers import PoissonDiskSampler

# from src.presets import configuration_list
from src.renderer import BaseRenderer
from src.solvers import MPM_Solver
from src.parsing import arguments
from src.constants import Phase

import taichi as ti
import numpy as np

OFFSET = 0.0234375
MAX_ITERATIONS = 300
LOWER_BOUND = -1e-6
UPPER_BOUND = 1e-6


def print_wrt_bound(value: float) -> str:
    if value < LOWER_BOUND or value > UPPER_BOUND:
        return utils.print_red(str(value))
    else:
        return utils.print_green(str(value))


configuration_list = [
    Configuration(
        name="Dam Break [Water]",
        geometries=[
            Rectangle(
                lower_left=(OFFSET, OFFSET),
                phase=Phase.Water,
                temperature=20.0,
                size=(0.5 - OFFSET, 0.5 - OFFSET),
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
    Configuration(
        name="Simple Spout Source [Water]",
        geometries=[
            Rectangle(
                lower_left=(0.48, 0.48),
                velocity=(0, -3),
                size=(0.04, 0.04),
                frame_threshold=i,
                temperature=20.0,
                phase=Phase.Water,
            )
            for i in range(1, 300)
        ],
        E=5.5e5,  # Young's modulus (1.4e5)
        nu=0.45,  # Poisson's ratio (0.2)
        zeta=1,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=20.0,
    ),
    Configuration(
        name="Spherefall [Water]",
        geometries=[
            Circle(
                center=(0.5, 0.35),
                phase=Phase.Water,
                velocity=(0, -2),
                radius=0.08,
            )
        ],
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=-20.0,
    ),
]

configuration_list.sort(key=lambda c: len(c.name), reverse=True)


class TestRenderer(BaseRenderer):
    def __init__(
        self,
        solver: MPM_Solver,
        configurations: list[Configuration],
        poisson_disk_sampler: PoissonDiskSampler,
    ) -> None:
        super().__init__(mpm_solver=solver, configurations=configurations, poisson_disk_sampler=poisson_disk_sampler)
        self.divergence_sum = ti.ndarray(ti.f32, shape=(solver.n_grid, solver.n_grid))
        self.divergence = ti.ndarray(ti.f32, shape=(solver.n_grid, solver.n_grid))
        self.max_divergence = 0
        self.min_divergence = 0

    @ti.kernel
    def compute_divergence(self, div: ti.types.ndarray(), avg: ti.types.ndarray()):  # pyright: ignore
        for i, j in self.mpm_solver.mass_c:
            div[i, j] = 0
            if self.mpm_solver.is_interior(i, j):
                div[i, j] += self.mpm_solver.velocity_x[i + 1, j]
                div[i, j] -= self.mpm_solver.velocity_x[i, j]
                div[i, j] += self.mpm_solver.velocity_y[i, j + 1]
                div[i, j] -= self.mpm_solver.velocity_y[i, j]
                avg[i, j] += div[i, j]

    def run(self) -> None:
        self.divergence_sum.fill(0)
        self.max_divergence = 0
        self.min_divergence = 0

        for i in range(MAX_ITERATIONS):
            self.substep()
            self.compute_divergence(self.divergence, self.divergence_sum)

            print(".", end=("\n" if i % 10 == 0 else " "), flush=True)

            divergence = self.divergence.to_numpy()
            abs_curr_min = np.min(divergence)
            if abs_curr_min < self.min_divergence:
                self.min_divergence = np.min(divergence)
            abs_curr_max = np.abs(np.max(divergence))
            if abs_curr_max > np.abs(self.max_divergence):
                self.max_divergence = np.max(divergence)


def main() -> None:
    # Initialize Taichi on the chosen architecture:
    if arguments.arch.lower() == "cpu":
        ti.init(arch=ti.cpu, debug=False, verbose=False, log_level=ti.INFO)
    elif arguments.arch.lower() == "gpu":
        ti.init(arch=ti.gpu, debug=arguments.debug)
    else:
        ti.init(arch=ti.cuda, debug=arguments.debug)

    max_particles = 500_000
    solver = MPM_Solver(quality=1, max_particles=max_particles)
    poisson_disk_sampler = PoissonDiskSampler(mpm_solver=solver)
    test_renderer = TestRenderer(
        poisson_disk_sampler=poisson_disk_sampler,
        configurations=configuration_list,
        solver=solver,
    )

    results = []
    all_tests_succeeded = True
    for configuration in configuration_list:
        print(f"NOW RUNNING: {configuration.name}")
        test_renderer.load_configuration(configuration)
        test_renderer.run()

        average_divergence = test_renderer.divergence_sum.to_numpy() / MAX_ITERATIONS
        min_average, max_average = np.min(average_divergence), np.max(average_divergence)
        min_spiking, max_spiking = test_renderer.min_divergence, test_renderer.max_divergence
        test_succeeded = min_average > LOWER_BOUND and max_average < UPPER_BOUND
        all_tests_succeeded &= test_succeeded
        result = (
            f"{configuration.name}\n"
            f"-> average min, max = {print_wrt_bound(min_average)}, {print_wrt_bound(max_average)}\n"
            f"-> spiking min, max = {print_wrt_bound(min_spiking)}, {print_wrt_bound(max_spiking)}\n"
            f"-> {utils.print_green("PASSED!") if test_succeeded else utils.print_red("DID NOT PASS!")}\n"
        )
        results.append(result)


    print(f"\n\niterations = {MAX_ITERATIONS}, lower bound = {LOWER_BOUND}, upper bound = {UPPER_BOUND}\n")
    print(*results, sep="\n", end="\n\n")
    print("\033[92m:)))))))))\033[0m" if all_tests_succeeded else "\033[91m:(\033[0m")


if __name__ == "__main__":
    main()
