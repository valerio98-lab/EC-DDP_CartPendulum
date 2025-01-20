# 🚀 Equality Constrained Differential Dynamic Programming (EC-DDP)
A reimplementation of **Equality Constrained Differential Dynamic Programming (EC-DDP)** for trajectory optimization with equality constraints, based on the **Augmented Lagrangian** approach proposed by **El Kazdadi et al. (ICRA 2021)**.

## 📌 Features
- 🔍 **DDP with equality constraints** using the **Augmented Lagrangian** formulation.
- 🏎 **Two strategies for handling Lagrange multipliers**:
  - **Globally constant multipliers** (classic augmented Lagrangian method).
  - **Affine multipliers w.r.t. state** (providing a feedback term for robustness).
- 📈 **Designed for optimal control in robotics** with complex dynamics.
- ⚡ **Efficient implementation** with optimized backward and forward passes.

## 📚 Repository Structure
- `ec_ddp.py` → Core implementation of **EC-DDP** with both multiplier strategies.
- `examples/` → Sample use cases for **cartpole** and **robotic arms**.
- `tests/` → Unit tests to validate algorithm correctness.

## 🔧 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/ec-ddp.git
cd ec-ddp
pip install -r requirements.txt
```

## 🔧 Quick Example
```bash
from ec_ddp import EC_DDP

# Define dynamics, cost function, and constraints
ddp_solver = EC_DDP(dynamics, cost_function, constraints)
optimal_trajectory = ddp_solver.solve(initial_state, initial_control)
```
