# Distributed Informative Path Planning by Gaussian Process Regression

## Overview
This repository contains the source code for the paper:  
**"Connectivity-Preserving Distributed Informative Path Planning for Mobile Robot Networks,"**  
published in *IEEE Robotics and Automation Letters*, vol. 9, no. 3, pp. 2949–2956.

## Requirements
- **Programming Language:** Julia 1.8.5

## Scenarios
Two simulation scenarios are provided:
1. **10 robots** in a **200 m × 200 m** environment  
   Run: `sim10robots200.jl`
2. **25 robots** in a **400 m × 400 m** environment  
   Run: `sim25robots400.jl`

## Notes on Computation
The simulations leverage **distributed computing** to improve performance.  
Please set the number of workers according to your computational resources by adjusting the variable **`nP`** in the simulation files above.
