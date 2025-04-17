from src.configurations import Circle, Rectangle, Configuration
from src.constants import Phase

configuration_list = [
    # Configuration(
    #     name="Melting Ice Cube",
    #     geometries=[
    #         Rectangle(Phase.Ice, 0.15, 0.15, 5000, (0, 0), (0.425, 0.0), 0, -10.0),
    #     ],
    #     E=1.4e5,  # Young's modulus (1.4e5)
    #     nu=0.2,  # Poisson's ratio (0.2)
    #     zeta=10,  # Hardening coefficient (10)
    #     theta_c=8.5e-2,  # Critical compression (2.5e-2)
    #     theta_s=5.0e-3,  # Critical stretch (7.5e-3)
    #     ambient_temperature=500.0,
    # ),
    # Configuration(
    #     name="Freezing Water Cube",
    #     geometries=[
    #         Rectangle(Phase.Water, 0.2, 0.2, 5000, (0, 0), (0.4, 0.0), 0, 10.0),
    #     ],
    #     E=1.4e5,  # Young's modulus (1.4e5)
    #     nu=0.2,  # Poisson's ratio (0.2)
    #     zeta=10,  # Hardening coefficient (10)
    #     theta_c=8.5e-2,  # Critical compression (2.5e-2)
    #     theta_s=5.0e-3,  # Critical stretch (7.5e-3)
    #     ambient_temperature=-5000.0,
    # ),
    Configuration(
        name="Waterspout Hits Pool [Water]",
        geometries=[
            Rectangle(
                lower_left=(0.0, 0.0),
                phase=Phase.Water,
                temperature=20.0,
                size=(1.0, 0.1),
                velocity=(0, 0),
            ),
            *[
                Rectangle(
                    phase=Phase.Water,
                    size=(0.04, 0.04),
                    velocity=(0, -3),
                    lower_left=(0.48, 0.48),
                    frame_threshold=i,
                    temperature=20.0,
                )
                for i in range(1, 200)
            ],
        ],
        E=5e5,  # Young's modulus (1.4e5)
        nu=0.45,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=20.0,
    ),
    # Configuration(
    #     name="Waterspout Hits Ice Cube",
    #     geometries=[
    #         *[Rectangle(Phase.Water, 0.02, 0.02, 20, (0, -2), (0.48, 0.55), i, 100000.0) for i in range(5, 300)],
    #         Rectangle(Phase.Ice, 0.1, 0.1, 3000, (0, 0), (0.45, 0.0), 0, -0.5),
    #     ],
    #     E=1.4e5,  # Young's modulus (1.4e5)
    #     nu=0.2,  # Poisson's ratio (0.2)
    #     zeta=10,  # Hardening coefficient (10)
    #     theta_c=8.5e-2,  # Critical compression (2.5e-2)
    #     theta_s=5.0e-3,  # Critical stretch (7.5e-3)
    #     ambient_temperature=10.0,
    # ),
    # Configuration(
    #     name="Waterspout Hits Ice Cubes",
    #     geometries=[
    #         *[Rectangle(Phase.Water, 0.05, 0.05, 25, (2, -2), (0.1, 0.8), i, 500.0) for i in range(10, 250)],
    #         Rectangle(Phase.Ice, 0.1, 0.1, 1_000, (0, 0), (0.59, 0.0), 0, -10.0),
    #         Rectangle(Phase.Ice, 0.1, 0.1, 1_000, (0, 0), (0.70, 0.0), 0, -10.0),
    #         Rectangle(Phase.Ice, 0.1, 0.1, 1_000, (0, 0), (0.65, 0.1), 0, -10.0),
    #     ],
    #     E=1.4e5,  # Young's modulus (1.4e5)
    #     nu=0.1,  # Poisson's ratio (0.2)
    #     zeta=20,  # Hardening coefficient (10)
    #     theta_c=3.5e-2,  # Critical compression (2.5e-2)
    #     theta_s=7.5e-3,  # Critical stretch (7.5e-3)
    #     ambient_temperature=20.0,
    # ),
    Configuration(
        name="Stationary Pool [Water]",
        geometries=[
            Rectangle(
                lower_left=(0.0, 0.0),
                phase=Phase.Water,
                temperature=20.0,
                size=(1.0, 0.15),
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
    # Configuration(
    #     name="Dropping Ice Cubes Into Body of Water",
    #     geometries=[
    #         Rectangle(Phase.Water, 0.96, 0.1, 8_000, (0, 0), (0, 0), 0, 50.0),
    #         Rectangle(Phase.Ice, 0.05, 0.05, 1_000, (0, -1), (0.25, 0.35), 10, -30.0),
    #         Rectangle(Phase.Ice, 0.05, 0.05, 1_000, (0, -1), (0.45, 0.15), 20, -30.0),
    #         Rectangle(Phase.Ice, 0.05, 0.05, 1_000, (0, -1), (0.65, 0.25), 30, -30.0),
    #     ],
    #     E=1.4e5,  # Young's modulus (1.4e5)
    #     nu=0.2,  # Poisson's ratio (0.2)
    #     zeta=10,  # Hardening coefficient (10)
    #     theta_c=2.5e-2,  # Critical compression (2.5e-2)
    #     theta_s=5.0e-3,  # Critical stretch (7.5e-3)
    #     ambient_temperature=0.0,
    # ),
    # Configuration(
    #     name="Freezing Lake",
    #     geometries=[
    #         Rectangle(Phase.Water, 0.96, 0.1, 20_000, (0, 0), (0, 0), 0, 1.0),
    #     ],
    #     E=1.4e5,  # Young's modulus (1.4e5)
    #     nu=0.2,  # Poisson's ratio (0.2)
    #     zeta=10,  # Hardening coefficient (10)
    #     theta_c=2.5e-2,  # Critical compression (2.5e-2)
    #     theta_s=5.0e-3,  # Critical stretch (7.5e-3)
    #     ambient_temperature=-500.0,
    # ),
    # Configuration(
    #     name="Freezing Waterspout",
    #     geometries=[
    #         *[Rectangle(Phase.Water, 0.05, 0.05, 10, (0, -2), (0.45, 0.85), i, 30.0) for i in range(10, 500)],
    #     ],
    #     E=1.4e5,  # Young's modulus (1.4e5)
    #     nu=0.2,  # Poisson's ratio (0.2)
    #     zeta=10,  # Hardening coefficient (10)
    #     theta_c=2.5e-2,  # Critical compression (2.5e-2)
    #     theta_s=5.0e-3,  # Critical stretch (7.5e-3)
    #     ambient_temperature=-500.0,
    # ),
    Configuration(
        name="Simple Spout Source",
        geometries=[
            Rectangle(
                phase=Phase.Water,
                size=(0.04, 0.04),
                velocity=(0, -3),
                lower_left=(0.48, 0.48),
                frame_threshold=i,
                temperature=20.0,
            )
            for i in range(1, 200)
        ],
        E=5.5e5,  # Young's modulus (1.4e5)
        nu=0.45,  # Poisson's ratio (0.2)
        zeta=1,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=20.0,
    ),
    # Configuration(
    #     name="Simple Blob Source",
    #     geometries=[
    #         *[Circle(Phase.Ice, 0.05, 1000, (5, 0), (0.1, 0.5), i, -20.0) for i in range(0, 250, 25)],
    #     ],
    #     E=1.4e5,  # Young's modulus (1.4e5)
    #     nu=0.25,  # Poisson's ratio (0.2)
    #     zeta=10,  # Hardening coefficient (10)
    #     theta_c=8.5e-2,  # Critical compression (2.5e-2)
    #     theta_s=7.5e-3,  # Critical stretch (7.5e-3)
    #     ambient_temperature=-20.0,
    # ),
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
    Configuration(
        name="Spherefall [Ice]",
        geometries=[
            Circle(
                center=(0.5, 0.35),
                velocity=(0, -2),
                phase=Phase.Ice,
                radius=0.08,
            ),
        ],
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=8.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=-20.0,
    ),
]
