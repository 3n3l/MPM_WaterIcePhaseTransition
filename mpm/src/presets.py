from src.configurations import Circle, Rectangle, Configuration
from src.constants import Phase

# Width of the bounding box, TODO: transform points to coordinates in bounding box
offset = 0.0234375

configuration_list = [
    Configuration(
        name="Melting Ice Cube [Ice -> Water]",
        geometries=[
            Rectangle(
                phase=Phase.Ice, size=(0.15, 0.15), velocity=(0, 0), lower_left=(0.425, offset), temperature=-10.0
            ),
        ],
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=1000.0,
    ),
    Configuration(
        name="Waterspout Hits Body of Water [Water]",
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
                for i in range(1, 300)
            ],
        ],
        E=5e5,  # Young's modulus (1.4e5)
        nu=0.45,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=20.0,
    ),
    Configuration(
        name="Dam Break [Water]",
        geometries=[
            Rectangle(
                lower_left=(offset, offset),
                phase=Phase.Water,
                temperature=20.0,
                size=(0.5 - offset, 0.5 - offset),
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
    Configuration(
        name="Dropping Ice Cubes Into Body of Water [Water, Ice]",
        geometries=[
            Rectangle(
                phase=Phase.Water,
                size=(0.96, 0.1),
                velocity=(0, 0),
                lower_left=(0, 0),
                frame_threshold=0,
                temperature=50.0,
            ),
            Rectangle(
                phase=Phase.Ice,
                size=(0.05, 0.05),
                velocity=(0, -1),
                lower_left=(0.25, 0.35),
                frame_threshold=10,
                temperature=-30.0,
            ),
            Rectangle(
                phase=Phase.Ice,
                size=(0.05, 0.05),
                velocity=(0, -1),
                lower_left=(0.45, 0.15),
                frame_threshold=20,
                temperature=-30.0,
            ),
            Rectangle(
                phase=Phase.Ice,
                size=(0.05, 0.05),
                velocity=(0, -1),
                lower_left=(0.65, 0.25),
                frame_threshold=30,
                temperature=-30.0,
            ),
        ],
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=0.0,
    ),
    Configuration(
        name="Freezing Lake [Water -> Ice]",
        geometries=[
            Rectangle(
                lower_left=(0.0, 0.0),
                phase=Phase.Water,
                temperature=20.0,
                size=(1.0, 0.15),
                velocity=(0, 0),
            ),
        ],
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=-500.0,
    ),
    Configuration(
        name="Simple Spout Source [Water]",
        geometries=[
            Rectangle(
                lower_left=(0.48, 0.48),
                velocity=(0, -3),
                size=(0.04, 0.04),
                frame_threshold=i,
                temperature=20.0,
                phase=Phase.Water,
            )
            for i in range(1, 300)
        ],
        E=5.5e5,  # Young's modulus (1.4e5)
        nu=0.45,  # Poisson's ratio (0.2)
        zeta=1,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=5.0e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=20.0,
    ),
    Configuration(
        name="Simple Blob Source [Ice]",
        geometries=[
            *[
                Circle(
                    velocity=(5, 0),
                    center=(0.25, 0.2 + (i * 0.005)),
                    radius=0.05,
                    temperature=20.0,
                    frame_threshold=i,
                    phase=Phase.Ice,
                )
                for i in range(10, 110, 25)
            ],
        ],
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.25,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=8.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=-20.0,
    ),
    Configuration(
        name="Spherefall [Water]",
        geometries=[
            Circle(
                center=(0.5, 0.5),
                phase=Phase.Water,
                velocity=(0, -3),
                radius=0.1,
            )
        ],
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=20,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=-20.0,
    ),
    Configuration(
        name="Spherefall [Ice]",
        geometries=[
            Circle(
                center=(0.5, 0.5),
                velocity=(0, -3),
                phase=Phase.Ice,
                radius=0.1,
            ),
        ],
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=-20.0,
    ),
    Configuration(
        name="Double Spherefall [Water, Ice]",
        geometries=[
            Circle(
                center=(0.25, 0.5),
                velocity=(0, -3),
                phase=Phase.Ice,
                radius=0.1,
            ),
            Circle(
                center=(0.75, 0.5),
                velocity=(0, -3),
                phase=Phase.Water,
                radius=0.1,
            ),
        ],
        E=1.4e5,  # Young's modulus (1.4e5)
        nu=0.2,  # Poisson's ratio (0.2)
        zeta=10,  # Hardening coefficient (10)
        theta_c=2.5e-2,  # Critical compression (2.5e-2)
        theta_s=7.5e-3,  # Critical stretch (7.5e-3)
        ambient_temperature=-20.0,
    ),
]

# Sort by length in descending order:
configuration_list.sort(key=lambda c: len(c.name), reverse=True)
