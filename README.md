## MPM - Water and Ice with Phase Transition
MLS-MPM implementation of [Augmented MPM for phase-change and varied materials](https://dl.acm.org/doi/10.1145/2601097.2601176), written in [Taichi](https://www.taichi-lang.org/).


### Installation
Dependencies are managed with Conda:
```bash
conda env create -f environment.yaml
conda activate MPM
```
This can be run on CPU or GPU (CUDA), the latter needs the cuSPARSE libraries and corresponding drivers.


### Usage
```bash
usage: main.py [-h] [-g [{GGUI,GUI}]] [-c [CONFIGURATION]] [-s [{Direct,Iterative}]] [-q [QUALITY]]

options:
  -h, --help            show this help message and exit
  -g [{GGUI,GUI}], --gui [{GGUI,GUI}]
                        Use GGUI (depends on Vulkan) or GUI system for the simulation.
  -c [CONFIGURATION], --configuration [CONFIGURATION]
                        0: Melting Ice Cube
                        1: Freezing Water Cube
                        2: Waterspout Hits Body of Water
                        3: Waterspout Hits Ice Cube
                        4: Waterspout Hits Ice Cubes
                        5: Dropping Ice Cubes Into Body of Water
                        6: Freezing Lake
                        7: Freezing Waterspout
                        8: Simple Spout Source
                        9: Simple Blob Source
                        10: Spherefall
                        11: Spherefall
  -s [{Direct,Iterative}], --solverType [{Direct,Iterative}]
                        Choose whether to use a direct or iterative solver for the pressure and heat systems.
  -q [QUALITY], --quality [QUALITY]
                        Choose a quality multiplicator for the simulation (higher is better).

Press R to reset, SPACE to pause/unpause the simulation!
```
