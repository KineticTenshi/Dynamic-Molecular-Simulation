# Event-Driven Particle Simulation

This repository contains multiple event-driven simulations of ideal gases composed of particles (monomers and dimers) in 2D boxes, governed by elastic collisions. These projects aim to explore physical equilibrium through computational methods including moving barriers, steady states, and particle-wall interactions.

---

## 🧪 Physical Background

These simulations model a 2D ideal gas where particles interact solely via **elastic collisions**, either with each other or with walls. No interparticle potentials are considered — the system evolves strictly according to conservation of momentum and kinetic energy.

### Key Equations

- **System energy**:
  $
  E = \sum_{i=1}^{N} \frac{1}{2} m_i v_i^2 = N k_B T
  $

- **Chamber temperature**:
  \[
  T_j = \frac{1}{N_j k_B} \sum_{q=1}^{N_j} \frac{m_q}{2} \vec{v}_q^2
  \]

- **Wall–particle collision mechanics**:

  The velocities after elastic collision between a particle and the moving wall are given by:

  \[
  \vec{v}_p' = \frac{m_p - m_w}{m_p + m_w} \vec{v}_p + \frac{2 m_w}{m_p + m_w} \vec{v}_w
  \]

  \[
  \vec{v}_w' = \frac{m_w - m_p}{m_p + m_w} \vec{v}_w + \frac{2 m_p}{m_p + m_w} \vec{v}_p
  \]

---

## 📂 Simulations Overview

### 1. **Monomer-only simulation**

Particles are initialized in a box and allowed to collide elastically over time. Demonstrates the dynamics of a basic ideal gas in 2D with no external forces.

Launch with:

```bash
py .\event_code_classMonomer.py
```

### 2. **Monomer + Dimer simulation**

Particles consist of both monomers and dimers (two spheres bound together). Additional logic handles internal constraints and collision handling for dimers.

Launch with:

```bash
py .\event_code_classDimer.py
```

### 3. **Simulation with Moving Wall**

A vertical wall separates the box into two chambers. The wall itself has mass and can move along the x-axis via elastic interaction with particles.

#### Key objectives:
- Analyze **temperature** and **density** evolution in each chamber.
- Study the **impact of wall mass** on steady-state convergence.
- Validate **thermodynamic consistency** using entropy derivatives.

Launch with:

```bash
py .\Projet_Moving_Wall_EventDrivenSimulation.py
```

---

## 📈 Notable Physical Results

- **Equilibrium**: The simulations confirm that the system reaches a thermodynamic steady state characterized by **equal temperature** and **pressure**, but not necessarily equal **density**.

- **Wall dynamics**: Wall mass influences the time to reach equilibrium. Heavier walls dampen oscillations but slow convergence.

- **System size**: Larger particle counts smooth the evolution curves due to increased collision frequency.

- **Asymmetry**: If one chamber contains smaller/lighter particles, densities may differ at equilibrium, even if temperature and pressure converge — consistent with thermodynamic predictions from entropy maximization.

---

## ⚠️ Requirements

- Python 3.10+
- `numpy`, `matplotlib`
- `tkinter` (GUI may be used for wall visualization)

Install dependencies if needed:

```bash
pip install numpy matplotlib
```

---

## 👩‍💻 Authors

- Domitille Avalle  
- Eric EA

Project under the supervision of **Prof. Juliane Klmaser**, November 2021.

---

## 📬 Notes

This project was developed in an academic context to better understand the microscopic foundations of thermodynamics via simulation. Contributions or questions are welcome via issues or pull requests.
