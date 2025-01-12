from Configuration import Configuration
from Solver import Solver, Phase
from Renderer import Renderer
from geometries import Circle


def main():
    quality = 1
    max_particles = 3_000 * (quality**2)
    configurations = [
        Configuration(
            name="Spherefall (slippery)",
            geometries=[Circle(Phase.Water, 0.1, 3000, (0, 0), (0.5, 0.5))],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            stickiness=2,  # Higher value means a stickier border
            friction=2,  # Higher value means the border has more friction
        ),
    ]

    print("-" * 150)
    print("[Hint] Press R to [R]eset, P|SPACE to [P]ause/un[P]ause and S|BACKSPACE to [S]tart recording!")
    print("-" * 150)

    solver = Solver(quality=quality, max_particles=max_particles)
    renderer = Renderer(solver=solver, configurations=configurations)
    renderer.run()


if __name__ == "__main__":
    main()
