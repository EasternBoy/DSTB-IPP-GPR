# Distributed Informative Path Planning by Gaussian Process Regression

## Overview
This repository contains the source code for the paper:  
**"Connectivity-Preserving Distributed Informative Path Planning for Mobile Robot Networks,"**  
published in *IEEE Robotics and Automation Letters*, vol. 9, no. 3, pp. 2949–2956.

## Requirements
- **Programming Language:** Julia 1.8.5
- **GaussianProcess.jl** 0.12.5
- **LinearAlgebra v1.12.0**

## Scenarios
Two simulation scenarios are provided:
1. **10 robots** in a **200 m × 200 m** environment  
   Set M = 10 and run: `main.jl`
2. **25 robots** in a **400 m × 400 m** environment  
   Set M = 25 and run: `main.jl`

## Notes on Computation
src/parallel-computing leverages **distributed computing** to improve performance.  
Please set the number of workers according to your computational resources by adjusting the variable **`nP`** in the simulation files above.


## Simulation results for 4, 10, and 25 robots
[YouTube demo video](https://www.youtube.com/watch?v=OhAIk5bYg74)
