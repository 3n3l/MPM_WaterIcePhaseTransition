from configurations import Configuration
from geometries import Circle, Square
from renderer import Renderer
from solver import Solver
from enums import Phase

import taichi as ti

# ti.init(arch=ti.cpu, debug=True)
ti.init(arch=ti.gpu)


def main():
    print("-" * 150)
    print("[Hint] Press R to [R]eset, P|SPACE to [P]ause/un[P]ause and S|BACKSPACE to [S]tart recording!")
    print("-" * 150)

    configurations = [
        Configuration(
            name="Simple Spout Source (Water)",
            geometries=[
                *[Square(Phase.Water, 0.05, 10, (0, -2), (0.45, 0.85), i) for i in range(10, 500)],
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ),
        Configuration(
            name="Simple Blob Source (Ice)",
            geometries=[
                *[Circle(Phase.Ice, 0.05, 1000, (5, 0), (0.1, 0.5), i) for i in range(0, 250, 25)],
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.25,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=8.5e-2,  # Critical compression (2.5e-2)
            theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        ),
        Configuration(
            name="Waterspout Hits Ice Cubes",
            geometries=[
                *[Square(Phase.Water, 0.05, 50, (2, -2), (0.1, 0.9), i) for i in range(10, 500)],
                Square(Phase.Ice, 0.1, 2000, (0, 0), (0.59, 0.0)),
                Square(Phase.Ice, 0.1, 2000, (0, 0), (0.70, 0.0)),
                Square(Phase.Ice, 0.1, 2000, (0, 0), (0.65, 0.1)),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ),
        Configuration(
            name="Spherefall (Water)",
            geometries=[
                Circle(Phase.Water, 0.06, 4000, (0, 0), (0.5, 0.5)),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ),
        Configuration(
            name="Spherefall (Ice)",
            geometries=[
                Circle(Phase.Ice, 0.06, 4000, (0, 0), (0.5, 0.5)),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=8.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ),
    ]

    quality = 1
    max_particles = max([c.n_particles for c in configurations])
    solver = Solver(quality=quality, max_particles=max_particles)
    renderer = Renderer(
        name="MPM - Water and Ice with Phase Transition",
        configurations=configurations,
        res=(720, 720),
        solver=solver,
    )
    renderer.run()


if __name__ == "__main__":
    main()
