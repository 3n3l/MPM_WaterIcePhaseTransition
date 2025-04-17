from src.samplers import PoissonDiskSampler
from src.presets import configuration_list
from src.renderer import GGUI, GUI
from src.solvers import MPM_Solver
from src.parsing import arguments

import taichi as ti


def main():
    # Initialize Taichi on the chosen architecture:
    if arguments.arch.lower() == "cpu":
        ti.init(arch=ti.cpu, debug=arguments.debug)
    elif arguments.arch.lower() == "gpu":
        ti.init(arch=ti.gpu, debug=arguments.debug)
    else:
        ti.init(arch=ti.cuda, debug=arguments.debug)

    solver = MPM_Solver(
        quality=arguments.quality,
        max_particles=100_000,  # TODO: there might be a way to set this again?
        should_use_direct_solver=(arguments.solverType.lower() == "direct"),
    )
    poisson_disk_sampler = PoissonDiskSampler(mpm_solver=solver)

    simulation_name = "MPM - Water and Ice with Phase Transition"
    if arguments.gui.lower() == "ggui":
        renderer = GGUI(
            configurations=configuration_list,
            poisson_disk_sampler=poisson_disk_sampler,
            mpm_solver=solver,
            res=(720, 720),
            name=simulation_name,
        )
        renderer.run()
    elif arguments.gui.lower() == "gui":
        renderer = GUI(
            configuration=configuration_list[arguments.configuration],
            poisson_disk_sampler=poisson_disk_sampler,
            mpm_solver=solver,
            res=720,
            name=simulation_name,
        )
        renderer.run()

    print("\n", "#" * 100, sep="")
    print("###", simulation_name)
    print("#" * 100)
    print(">>> R        -> [R]eset the simulation.")
    print(">>> P|SPACE  -> [P]ause/Un[P]ause the simulation.")
    print()


if __name__ == "__main__":
    main()
