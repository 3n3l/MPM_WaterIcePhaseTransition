import utils  # import first to append parent directory to path

from src.enums import Phase, Color, State, Classification
from src.configurations import Configuration
from src.mpm_solver import MPM_Solver
from src.geometries import Square

import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)


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
            x_divergence = solver.face_velocity_x[i + 1, j] - solver.face_velocity_x[i, j]
            y_divergence = solver.face_velocity_y[i, j + 1] - solver.face_velocity_y[i, j]
            div[i, j] = x_divergence + y_divergence
        else:
            div[i, j] = 0


def main() -> None:
    configuration = Configuration(
        name="Simple Spout Source (Water)",
        geometries=[
            # *[Square(Phase.Water, 0.05, 10, (0, -2), (0.45, 0.85), i) for i in range(10, 500)],
            *[Square(Phase.Water, 0.05, 100, (0, -5.0), (0.45, 0.5), i) for i in range(10, 200)],
        ],
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
    )

    solver = MPM_Solver(quality=1, max_particles=configuration.n_particles)
    load_configuration(solver, configuration)
    reset_solver(solver, configuration)

    we_succeeded = True
    divergence = ti.ndarray(ti.f32, shape=(solver.n_grid, solver.n_grid))
    for i in range(1, 501):
        solver.substep()
        compute_divergence(solver, divergence)
        print(".", end=("\n" if i % 10 == 0 else " "), flush=True)
        if not np.any(divergence.to_numpy() > 0):
            we_succeeded = False
            break

    print()
    print(divergence.to_numpy())
    print()

    print(":)))))))))" if we_succeeded else ":(")


if __name__ == "__main__":
    main()
