# Equality Constrained Differential Dynamic Programming (EC-DDP)
A reimplementation of **Equality Constrained Differential Dynamic Programming (EC-DDP)** for trajectory optimization with equality constraints, based on the **Augmented Lagrangian** approach proposed by **El Kazdadi et al. (ICRA 2021)**.

## Features
- **DDP with equality constraints** using the **Augmented Lagrangian** formulation.
- **Two strategies for handling Lagrange multipliers**:
  - **Globally constant multipliers** (classic augmented Lagrangian method).
  - **Affine multipliers w.r.t. state** (TODO).
- **Designed for optimal control in robotics** with complex dynamics.

## Repository Structure
- `ec_ddp/src/ec_ddp.py` → Core implementation of **EC-DDP** with both multiplier strategies.
- `ddp.py` → Implementation of **DDP** for comparison purposes.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/valerio98-lab/EC-DDP_CartPendulum.git
cd EC-DDP_CartPendulum
pip install -r requirements.txt
```

## Test
You can easily execute in parallel two simulations:
```bash
python main.py --model cart_pendulum
```
This command will execute an ec-ddp and ddp simulation on cart pendulum. 

Eventually it can be executed a parallel simulation on different systems:
```bash
python main.py --ddp_model cart_pendulum --ec_ddp_model pendubot
``` 

##Authors:

Valerio Belli, José del Valle Delgado, Serena Trovalusci, Leonardo Sandri
