from argparse import ArgumentParser, RawTextHelpFormatter
from src.configurations import Configuration
from src.geometries import Circle, Rectangle
from src.renderer.GGUI import GGUI_Renderer
from src.renderer.GUI import GUI_Renderer
from src.mpm_solver import MPM_Solver
from src.enums import Phase

import taichi as ti

ti.init(arch=ti.cpu, debug=True)
# ti.init(arch=ti.cuda, debug=True)
# ti.init(arch=ti.gpu)


def main():
    configurations = [
        Configuration(
            name="Melting Ice Cube",
            geometries=[
                Rectangle(Phase.Ice, 0.15, 0.15, 5000, (0, 0), (0.425, 0.0), 0, -10.0),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=8.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=500.0,
        ),
        Configuration(
            name="Freezing Water Cube",
            geometries=[
                Rectangle(Phase.Water, 0.2, 0.2, 5000, (0, 0), (0.4, 0.0), 0, 10.0),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=8.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=-5000.0,
        ),
        Configuration(
            name="Waterspout Hits Body of Water",
            geometries=[
                Rectangle(Phase.Water, 0.96, 0.1, 5_000, (0, 0), (0, 0), 0, 20.0),
                *[Rectangle(Phase.Water, 0.1, 0.05, 10, (0, -1), (0.45, 0.45), i, 20.0) for i in range(10, 500)],
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=20.0,
        ),
        Configuration(
            name="Waterspout Hits Ice Cube",
            geometries=[
                *[Rectangle(Phase.Water, 0.02, 0.02, 20, (0, -2), (0.48, 0.55), i, 100000.0) for i in range(5, 300)],
                Rectangle(Phase.Ice, 0.1, 0.1, 3000, (0, 0), (0.45, 0.0), 0, -0.5),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=8.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=10.0,
        ),
        Configuration(
            name="Waterspout Hits Ice Cubes",
            geometries=[
                *[Rectangle(Phase.Water, 0.05, 0.05, 25, (2, -2), (0.1, 0.8), i, 500.0) for i in range(10, 250)],
                Rectangle(Phase.Ice, 0.1, 0.1, 1_000, (0, 0), (0.59, 0.0), 0, -10.0),
                Rectangle(Phase.Ice, 0.1, 0.1, 1_000, (0, 0), (0.70, 0.0), 0, -10.0),
                Rectangle(Phase.Ice, 0.1, 0.1, 1_000, (0, 0), (0.65, 0.1), 0, -10.0),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.1,  # Poisson's ratio (0.2)
            zeta=20,  # Hardening coefficient (10)
            theta_c=3.5e-2,  # Critical compression (2.5e-2)
            theta_s=7.5e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=20.0,
        ),
        Configuration(
            name="Dropping Ice Cubes Into Body of Water",
            geometries=[
                Rectangle(Phase.Water, 0.96, 0.1, 8_000, (0, 0), (0, 0), 0, 50.0),
                Rectangle(Phase.Ice, 0.05, 0.05, 1_000, (0, -1), (0.25, 0.35), 10, -30.0),
                Rectangle(Phase.Ice, 0.05, 0.05, 1_000, (0, -1), (0.45, 0.15), 20, -30.0),
                Rectangle(Phase.Ice, 0.05, 0.05, 1_000, (0, -1), (0.65, 0.25), 30, -30.0),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=0.0,
        ),
        Configuration(
            name="Freezing Lake",
            geometries=[
                Rectangle(Phase.Water, 0.96, 0.1, 20_000, (0, 0), (0, 0), 0, 1.0),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=-500.0,
        ),
        Configuration(
            name="Freezing Waterspout",
            geometries=[
                *[Rectangle(Phase.Water, 0.05, 0.05, 10, (0, -2), (0.45, 0.85), i, 30.0) for i in range(10, 500)],
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=-500.0,
        ),
        Configuration(
            name="Simple Spout Source",
            geometries=[
                *[Rectangle(Phase.Water, 0.05, 0.05, 10, (0, -2), (0.45, 0.85), i, 20.0) for i in range(10, 500)],
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=20.0,
        ),
        Configuration(
            name="Simple Blob Source",
            geometries=[
                *[Circle(Phase.Ice, 0.05, 1000, (5, 0), (0.1, 0.5), i, -20.0) for i in range(0, 250, 25)],
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.25,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=8.5e-2,  # Critical compression (2.5e-2)
            theta_s=7.5e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=-20.0,
        ),
        Configuration(
            name="Spherefall",
            geometries=[
                Circle(Phase.Water, 0.06, 4000, (0, 0), (0.5, 0.5), 0, 10.0),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=20.0,
        ),
        Configuration(
            name="Spherefall",
            geometries=[
                Circle(Phase.Ice, 0.06, 4000, (0, 0), (0.5, 0.5), 0, -10.0),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=8.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=-20.0,
        ),
    ]

    print("\n", "-" * 150)
    epilog = "[Hint] Press R to reset, SPACE to pause/unpause the simulation!"
    parser = ArgumentParser(prog="main.py", epilog=epilog, formatter_class=RawTextHelpFormatter)

    ggui_help = "Use GGUI (depends on Vulkan) or GUI system for the simulation."
    parser.add_argument("-g", "--gui", default="GGUI", nargs="?", choices=["GGUI", "GUI"], help=ggui_help)

    configuration_help = "\n".join([f"{i}: {c.name}" for i, c in enumerate(configurations)])
    parser.add_argument("-c", "--configuration", default=0, nargs="?", help=configuration_help, type=int)
    args = parser.parse_args()

    quality = 1
    max_particles = max([c.n_particles for c in configurations])
    solver = MPM_Solver(quality=quality, max_particles=max_particles)

    if args.gui == "GGUI":
        renderer = GGUI_Renderer(
            name="MPM - Water and Ice with Phase Transition",
            configurations=configurations,
            res=(720, 720),
            solver=solver,
        )
        renderer.run()
    elif args.gui == "GUI":
        renderer = GUI_Renderer(
            name="MPM - Water and Ice with Phase Transition",
            configuration=configurations[args.configuration],
            solver=solver,
            res=720,
        )
        renderer.run()


if __name__ == "__main__":
    main()
