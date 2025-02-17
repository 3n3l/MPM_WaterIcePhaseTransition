import utils  # import first to append parent directory to path

from src.enums import Phase, Color, State, Classification
from src.configurations import Configuration
from src.geometries import Circle, Rectangle
from src.mpm_solver import MPM_Solver

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
    for p in solver.particle_position:
        if p < configuration.n_particles:
            solver.particle_color[p] = Color.Water if configuration.p_phase[p] == Phase.Water else Color.Ice
            solver.p_activation_threshold[p] = configuration.p_activity_bound[p]
            solver.particle_position[p] = configuration.p_position[p] + solver.boundary_offset
            solver.particle_velocity[p] = configuration.p_velocity[p]
            solver.p_activation_state[p] = configuration.p_state[p]
            solver.particle_phase[p] = configuration.p_phase[p]
        else:
            # TODO: this might be completely irrelevant, as only the first n_particles are used anyway?
            #       So work can be saved by just ignoring all the other particles and iterating only
            #       over the configuration.n_particles?
            solver.particle_color[p] = Color.Background
            solver.p_activation_threshold[p] = 0
            solver.particle_position[p] = [0, 0]
            solver.particle_velocity[p] = [0, 0]
            solver.p_activation_state[p] = State.Inactive
            solver.particle_phase[p] = Phase.Water

        solver.particle_mass[p] = solver.particle_vol * solver.rho_0
        solver.particle_inv_lambda[p] = 1 / solver.lambda_0[None]
        solver.particle_FE[p] = ti.Matrix([[1, 0], [0, 1]])
        solver.particle_C[p] = ti.Matrix.zero(float, 2, 2)
        solver.p_active_position[p] = [0, 0]
        solver.particle_JE[p] = 1
        solver.particle_JP[p] = 1


@ti.kernel
def compute_divergence(solver: ti.template(), div: ti.types.ndarray()):  # pyright: ignore
    for i, j in solver.cell_pressure:
        if solver.cell_classification[i, j] == Classification.Interior:
            x_divergence = solver.face_velocity_x[i, j] - solver.face_velocity_x[i + 1, j]
            y_divergence = solver.face_velocity_y[i, j] - solver.face_velocity_y[i, j + 1]
            div[i, j] = x_divergence + y_divergence
        else:
            div[i, j] = 0


def main() -> None:
    configurations = [
        Configuration(
            name="Waterspout Hits Body of Water (Water)",
            geometries=[
                Rectangle(Phase.Water, 0.96, 0.1, 5_000, (0, 0), (0, 0)),
                *[Rectangle(Phase.Water, 0.1, 0.05, 10, (0, -1), (0.45, 0.45), i) for i in range(10, 500)],
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ),
        Configuration(
            name="Simple Spout Source (Water)",
            geometries=[
                *[Rectangle(Phase.Water, 0.05, 0.05, 10, (0, -2), (0.45, 0.85), i) for i in range(10, 500)],
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.2,  # Poisson's ratio (0.2)
            zeta=10,  # Hardening coefficient (10)
            theta_c=2.5e-2,  # Critical compression (2.5e-2)
            theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ),
        Configuration(
            name="Waterspout Hits Ice Cubes (Water, Ice)",
            geometries=[
                *[Rectangle(Phase.Water, 0.05, 0.05, 25, (2, -2), (0.1, 0.8), i) for i in range(10, 500)],
                Rectangle(Phase.Ice, 0.1, 0.1, 2000, (0, 0), (0.59, 0.0)),
                Rectangle(Phase.Ice, 0.1, 0.1, 2000, (0, 0), (0.70, 0.0)),
                Rectangle(Phase.Ice, 0.1, 0.1, 2000, (0, 0), (0.65, 0.1)),
            ],
            E=1.4e5,  # Young's modulus (1.4e5)
            nu=0.1,  # Poisson's ratio (0.2)
            zeta=20,  # Hardening coefficient (10)
            theta_c=3.5e-2,  # Critical compression (2.5e-2)
            theta_s=7.5e-3,  # Critical stretch (7.5e-3)
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

                # print("+" * 200)
                # compute_divergence(solver, divergence)
                # print("BEFORE:")
                # pressure = solver.cell_pressure.to_numpy()
                # print("P ->", np.min(pressure), np.max(pressure))
                # print("D ->", np.min(divergence.to_numpy()), np.max(divergence.to_numpy()))

                solver.pressure_solver.solve()
                # FIXME: this shouldn't be needed, but keep this here for testing purposes ???
                # solver.face_velocity_x.copy_from(solver.pressure_solver.face_velocity_x)
                # solver.face_velocity_y.copy_from(solver.pressure_solver.face_velocity_y)

                # compute_divergence(solver, divergence)
                # pressure = solver.cell_pressure.to_numpy()
                # print("AFTER:")
                # print("P ->", np.min(pressure), np.max(pressure))
                # print("D ->", np.min(divergence.to_numpy()), np.max(divergence.to_numpy()))
                # print()

                solver.grid_to_particle()

            compute_divergence(solver, divergence)
            print(".", end=("\n" if i % 10 == 0 else " "), flush=True)
            if np.any(np.round(divergence.to_numpy(), 2) != 0):  # pyright: ignore
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
