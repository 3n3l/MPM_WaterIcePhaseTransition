import utils  # import first to append parent directory to path

from src.enums import Phase, Color, State, Classification, Capacity, Conductivity
from src.mpm_solver import MPM_Solver, LATENT_HEAT
from src.configurations import Configuration
from src.geometries import Circle, Rectangle

import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, debug=True)
# ti.init(arch=ti.cuda, debug=True)


@ti.kernel
def load_configuration(solver: ti.template(), configuration: ti.template()):  # pyright: ignore
    """
    Loads the chosen configuration into the MLS-MPM solver.
    ---
    Parameters:
        configuration: Configuration
    """
    solver.n_particles[None] = configuration.n_particles
    solver.stickiness[None] = configuration.stickiness
    solver.friction[None] = configuration.friction
    solver.lambda_0[None] = configuration.lambda_0
    solver.theta_c[None] = configuration.theta_c
    solver.theta_s[None] = configuration.theta_s
    solver.zeta[None] = configuration.zeta
    solver.mu_0[None] = configuration.mu_0
    solver.nu[None] = configuration.nu
    solver.E[None] = configuration.E


@ti.kernel
def reset_solver(solver: ti.template(), configuration: ti.template()):  # pyright: ignore
    """
    Resets the MLS-MPM solver to the field values of the configuration.
    ---
    Parameters:
        configuration: Configuration
    """
    solver.current_frame[None] = 0
    for p in solver.position_p:
        if p < configuration.n_particles:
            p_is_water = configuration.phase_p[p] == Phase.Water
            solver.capacity_p[p] = Capacity.Water if p_is_water else Capacity.Ice
            solver.conductivity_p[p] = Conductivity.Water if p_is_water else Conductivity.Ice
            solver.color_p[p] = Color.Water if p_is_water else Color.Ice
            solver.heat_p[p] = LATENT_HEAT if p_is_water else 0.0

            solver.activation_threshold_p[p] = configuration.activity_threshold_p[p]
            solver.velocity_p[p] = configuration.velocity_p[p]
            solver.temperature_p[p] = configuration.temperature_p[p]
            solver.activation_state_p[p] = configuration.state_p[p]
            solver.phase_p[p] = configuration.phase_p[p]

            offset_position = configuration.position_p[p] + solver.boundary_offset
            p_is_active = configuration.state_p[p] == State.Active
            solver.active_position_p[p] = offset_position if p_is_active else [0, 0]
            solver.position_p[p] = offset_position
        else:
            # TODO: this might be completely irrelevant, as only the first n_particles are used anyway?
            #       So work can be saved by just ignoring all the other particles and iterating only
            #       over the configuration.n_particles?
            solver.color_p[p] = Color.Background
            solver.capacity_p[p] = Capacity.Zero
            solver.conductivity_p[p] = Conductivity.Zero
            solver.activation_threshold_p[p] = 0
            solver.temperature_p[p] = 0
            solver.position_p[p] = [0, 0]
            solver.velocity_p[p] = [0, 0]
            solver.activation_state_p[p] = State.Inactive
            solver.phase_p[p] = Phase.Water
            solver.heat_p[p] = 0
            solver.heat_p[p] = 0
            solver.active_position_p[p] = [0, 0]

        solver.mass_p[p] = solver.particle_vol * solver.rho_0
        solver.inv_lambda_p[p] = 1 / solver.lambda_0[None]
        # solver.inv_lambda_p[p] = 1 / 9999999999.0
        solver.FE_p[p] = ti.Matrix([[1, 0], [0, 1]])
        solver.C_p[p] = ti.Matrix.zero(float, 2, 2)
        solver.JE_p[p] = 1
        solver.JP_p[p] = 1


@ti.kernel
def compute_divergence(solver: ti.template(), div: ti.types.ndarray()):  # pyright: ignore
    for i, j in solver.pressure_c:
        div[i, j] = 0
        if solver.classification_c[i, j] == Classification.Interior:
            div[i, j] += solver.velocity_x[i + 1, j] - solver.velocity_x[i, j]
            div[i, j] += solver.velocity_y[i, j + 1] - solver.velocity_y[i, j]


def main() -> None:
    configurations = [
        Configuration(
            name="Waterspout Hits Body of Water",
            geometries=[
                # TODO: width is set with boundary_offset in mind, change this to absolute values,
                #       or even find a cleaner solution for this?
                Rectangle(Phase.Water, 0.953, 0.05, 5_000, (0, 0), (0, 0), 0, 20.0),
                *[Rectangle(Phase.Water, 0.08, 0.04, 50, (0, -2), (0.45, 0.45), i, 20.0) for i in range(10, 300)],
            ],
            E=1e4,  # Young's modulus (1.4e5)
            nu=0.49,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
            ambient_temperature=20.0,
        ),
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
            name="Stationary Pool of Water",
            geometries=[
                # TODO: width is set with boundary_offset in mind, change this to absolute values,
                #       or even find a cleaner solution for this?
                Rectangle(Phase.Water, 0.953, 0.05, 5_000, (0, 0), (0, 0), 0, 20.0),
            ],
            E=1e4,  # Young's modulus (1.4e5)
            nu=0.49,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
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

    max_particles = max([c.n_particles for c in configurations])
    solver = MPM_Solver(quality=1, max_particles=max_particles)
    divergence = ti.ndarray(ti.f32, shape=(solver.n_grid, solver.n_grid))
    we_succeeded = True

    for configuration in configurations:
        print(f"NOW RUNNING: {configuration.name}")
        load_configuration(solver, configuration)
        reset_solver(solver, configuration)
        for i in range(1, 301):
            solver.current_frame[None] += 1
            for _ in range(int(2e-3 // solver.dt)):
                solver.reset_grids()
                solver.particle_to_grid()
                solver.momentum_to_velocity()
                solver.classify_cells()
                solver.compute_volumes()
                solver.pressure_solver.solve()
                solver.grid_to_particle()

            prev_min, prev_max = np.abs(np.min(divergence.to_numpy())), np.abs(np.max(divergence.to_numpy()))
            compute_divergence(solver, divergence)
            curr_min, curr_max = np.abs(np.min(divergence.to_numpy())), np.abs(np.max(divergence.to_numpy()))

            print(".", end=("\n" if i % 10 == 0 else " "), flush=True)

            if np.round(curr_min) > np.round(prev_min) or np.round(curr_max) > np.round(prev_max):
                # The solver actually increased the divergence :(
                print("\n\nDivergence increased :(")
                print(f"prev_min = {prev_min}, prev_max = {prev_max}")
                print(f"curr_min = {curr_min}, curr_max = {curr_max}")
                we_succeeded = False
                break

            if np.any(np.round(divergence.to_numpy(), 2) != 0):  # pyright: ignore
                print("\n\nDivergence too big :(")
                we_succeeded = False
                break

        if not we_succeeded:
            break

    print("\n")
    print("Divergence, min ->", np.min(divergence.to_numpy()))
    print("Divergence, max ->", np.max(divergence.to_numpy()))
    print()

    print("\033[92m:)))))))))\033[0m" if we_succeeded else "\033[91m:(\033[0m")


if __name__ == "__main__":
    main()
