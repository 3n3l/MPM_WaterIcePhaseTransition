from argparse import ArgumentParser, RawTextHelpFormatter
from configurations import configuration_list
from src.sampling import PoissonDiskSampler
from src.renderer.ggui import GGUI_Renderer
from src.renderer.gui import GUI_Renderer
from src.mpm_solver import MPM_Solver

import taichi as ti


def main():
    simulation_name = "MPM - Water and Ice with Phase Transition"
    epilog = "Press R to reset, SPACE to pause/unpause the simulation!"
    parser = ArgumentParser(prog="main.py", epilog=epilog, formatter_class=RawTextHelpFormatter)

    ggui_help = "Use GGUI (depends on Vulkan) or GUI system for the simulation."
    parser.add_argument(
        "-g",
        "--gui",
        default="GGUI",
        nargs="?",
        choices=["GGUI", "GUI"],
        help=ggui_help,
    )

    configuration_help = "\n".join([f"{i}: {c.name}" for i, c in enumerate(configuration_list)])
    parser.add_argument(
        "-c",
        "--configuration",
        default=0,
        nargs="?",
        help=configuration_help,
        type=int,
    )

    solver_type_help = "Choose whether to use a direct or iterative solver for the pressure and heat systems."
    parser.add_argument(
        "-s",
        "--solverType",
        default="Direct",
        nargs="?",
        choices=["Direct", "Iterative"],
        help=solver_type_help,
    )

    quality_help = "Choose a quality multiplicator for the simulation (higher is better)."
    parser.add_argument(
        "-q",
        "--quality",
        default=1,
        nargs="?",
        help=quality_help,
        type=int,
    )

    solver_type_help = "Choose the Taichi architecture to run on."
    parser.add_argument(
        "-a",
        "--arch",
        default="CPU",
        nargs="?",
        choices=["CPU", "GPU", "CUDA"],
        help=solver_type_help,
    )

    solver_type_help = "Turn on debugging."
    parser.add_argument(
        "-d",
        "--debug",
        default=True,
        action="store_false",
        help=solver_type_help,
    )

    args = parser.parse_args()

    # Initialize Taichi on the chosen architecture:
    if args.arch.lower() == "cpu":
        ti.init(arch=ti.cpu, debug=args.debug)
    elif args.arch.lower() == "gpu":
        ti.init(arch=ti.gpu, debug=args.debug)
    else:
        ti.init(arch=ti.cuda, debug=args.debug)

    solver = MPM_Solver(
        quality=args.quality,
        # max_particles=max([c.n_particles for c in configuration_list]),
        max_particles=10_000,  # TODO: there might be a way to set this again?
        should_use_direct_solver=(args.solverType.lower() == "direct"),
    )

    poisson_disk_sampler = PoissonDiskSampler(mpm_solver=solver)

    if args.gui.lower() == "ggui":
        renderer = GGUI_Renderer(
            configurations=configuration_list,
            poisson_disk_sampler=poisson_disk_sampler,
            mpm_solver=solver,
            res=(720, 720),
            name=simulation_name,
        )
        renderer.run()
    elif args.gui.lower() == "gui":
        renderer = GUI_Renderer(
            configuration=configuration_list[args.configuration],
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
