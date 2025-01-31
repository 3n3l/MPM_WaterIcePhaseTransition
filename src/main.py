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
            name="Spherefall (Water)",
            geometries=[
                # Circle(Phase.Water, 0.01, 1000, (0, 0), (0.5, 0.25)),
                *[Square(Phase.Water, 0.05, 10, (0, -2), (0.5, 0.8), i) for i in range(10, 1000)],
                Square(Phase.Ice, 0.05, 500, (0, 0), (0.25, 0.25)),
                Square(Phase.Ice, 0.05, 500, (0, 0), (0.5, 0.05)),
                Square(Phase.Ice, 0.05, 500, (0, 0), (0.75, 0.05)),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ),
        # Configuration(
        #     name="Spherefall (Ice)",
        #     geometries=[
        #         Circle(Phase.Ice, 0.08, 3000, (0, 0), (0.5, 0.25)),
        #     ],
        #     E=1.4e5,  # Young's modulus (1.4e5)
        #     nu=0.2,  # Poisson's ratio (0.2)
        #     zeta=10,  # Hardening coefficient (10)
        #     theta_c=2.5e-2,  # Critical compression (2.5e-2)
        #     theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        # ),
        # Configuration(
        #     name="Spherefall (Water + Ice)",
        #     geometries=[
        #         Circle(Phase.Water, 0.08, 3000, (0, 0), (0.5, 0.5)),
        #         Circle(Phase.Ice, 0.08, 3000, (0, 0), (0.5, 0.25)),
        #     ],
        #     E=1.4e5,  # Young's modulus (1.4e5)
        #     nu=0.2,  # Poisson's ratio (0.2)
        #     zeta=10,  # Hardening coefficient (10)
        #     theta_c=2.5e-2,  # Critical compression (2.5e-2)
        #     theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        # ),
        # Configuration(
        #     name="Circle and Squares (Water + Ice)",
        #     geometries=[
        #         Circle(Phase.Water, 0.1, 15_000, (0, -3), (0.5, 0.5)),
        #         Square(Phase.Ice, 0.05, 500, (0, 0), (0.25, 0.05)),
        #         Square(Phase.Ice, 0.05, 500, (0, 0), (0.5, 0.05)),
        #         Square(Phase.Ice, 0.05, 500, (0, 0), (0.75, 0.05)),
        #     ],
        #     E=1.4e5,  # Young's modulus (1.4e5)
        #     nu=0.2,  # Poisson's ratio (0.2)
        #     zeta=10,  # Hardening coefficient (10)
        #     theta_c=2.5e-2,  # Critical compression (2.5e-2)
        #     theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        # ),
    ]

    quality = 1
    max_particles = max([c.n_particles for c in configurations])
    solver = Solver(quality=quality, max_particles=max_particles)
    renderer = Renderer(solver=solver, configurations=configurations)
    renderer.run()


if __name__ == "__main__":
    main()
