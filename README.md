## MPM - Water and Ice with Phase Transition
[MLS-MPM](https://dl.acm.org/doi/10.1145/3197517.3201293) implementation of [Augmented MPM for phase-change and varied materials](https://dl.acm.org/doi/10.1145/2601097.2601176), written in [Taichi](https://www.taichi-lang.org/).


### Installation
Dependencies are managed with Conda:
```bash
conda env create -f environment.yaml
conda activate MPM
```
You also need to install [cuSPARSE libraries](https://pypi.org/project/nvidia-cusparse-cu12/) and [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for the CUDA backend,
and [Vulkan Drivers](https://developer.nvidia.com/vulkan-driver) for the GGUI frontend. Both of these are optional, but result in better performance and visibility.


### Simulation
```bash
python mpm/main.py --arch=CPU     # runs the simulation on the CPU
python mpm/main.py --arch=CUDA    # runs the simulation on the GPU
```

If Vulkan is not available you can resort to Taichi's older GUI system with
```bash
python mpm/main.py --gui=GUI --configuration=3
```
Keep in mind that the simulation starts paused, pause/unpause is toggled with space.


### Options
<!-- Run the simulation with `python mpm/main.py`, or on the CUDA backend with `python mpm/main.py --arch=CUDA`, the GGUI frontend needs Vulkan, -->
<!-- if this is not available you can resort to `python mpm/main.py --gui=GUI` -->

```bash
options:
  -h, --help            show this help message and exit
  -g [{GGUI,GUI}], --gui [{GGUI,GUI}]
                        Use GGUI (depends on Vulkan) or GUI system for the simulation.
  -c [CONFIGURATION], --configuration [CONFIGURATION]
                        Available Configurations:
                        [0] -> Dropping Ice Cubes Into Body of Water [Water, Ice]
                        [1] -> Waterspout Hits Body of Water [Water]
                        [2] -> Iceballs hitting each other [Ice]
                        [3] -> Floating Ice Cube [Ice -> Water]
                        [4] -> Floating Ice Ball [Ice -> Water]
                        [5] -> Melting Ice Cube [Ice -> Water]
                        [6] -> Double Spherefall [Water, Ice]
                        [7] -> Freezing Lake [Water -> Ice]
                        [8] -> Simple Spout Source [Water]
                        [9] -> Simple Blob Source [Ice]
                        [10] -> Stationary Pool [Water]
                        [11] -> Spherefall [Water]
                        [12] -> Dam Break [Water]
                        [13] -> Spherefall [Ice]
  -s [{Direct,Iterative}], --solverType [{Direct,Iterative}]
                        Choose whether to use a direct or iterative solver for the pressure and heat systems.
  -q [QUALITY], --quality [QUALITY]
                        Choose a quality multiplicator for the simulation (higher is better).
  -a [{CPU,GPU,CUDA}], --arch [{CPU,GPU,CUDA}]
                        Choose the Taichi architecture to run on.
  -d, --debug           Turn on debugging.
```
