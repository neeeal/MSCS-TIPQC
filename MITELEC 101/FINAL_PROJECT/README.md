# Multi-Agent Pac-Man Simulation: Conflict Resolution Study

This project implements a decentralized Multi-Agent System (MAS) within a metabolically-constrained Pac-Man environment. It is designed to conduct a comparative study on conflict resolution paradigms, specifically evaluating the trade-offs between **Static Arbitration** (Priority-Based), **Explicit Negotiation** (Alternating Offers), and **Implicit Learning** (Q-Learning).

## Project Overview

Three autonomous Pac-Man agents must cooperate to clear a maze while avoiding ghosts and managing a decaying energy supply. The simulation creates a high-contention environment to test how different protocols handle:
* **Resource Contention:** Competition for shared corridors and pellets.
* **Metabolic Constraints:** The balance between waiting (safety) and moving (feeding).
* **Fairness vs. Efficiency:** Analysis of decision latency, collision rates, and equitable resource access.

## Strategies Implemented
1.  **Priority-Based (Static Arbitration):** A rigid hierarchy resolves conflicts based on score and energy levels.
2.  **Alternating Offers (Explicit Negotiation):** Agents engage in a communicative protocol to bargain for the right of way.
3.  **Q-Learning (Implicit Learning):** Agents utilize Reinforcement Learning to adaptively learn collision avoidance policies over time.

---

## How to Use

Follow these steps to set up the simulation environment correctly.

### 1. Prerequisites
Ensure you have **Python 3.10.19** installed on your system to ensure compatibility with the simulation dependencies.

### 2. Environment Setup
It is highly recommended to use a virtual environment to isolate dependencies.

**Windows:**
```bash
# Create the virtual environment using Python 3.10.19
py -3.10 -m venv venv

# Activate the environment
.\venv\Scripts\activate
```

**macOS / Linux:**
```Bash
# Create the virtual environment using Python 3.10.19
python3.10 -m venv venv

# Activate the environment
source venv/bin/activate
```


### 3. Installation
Navigate to the project directory and install the required packages.

```Bash
cd "MITELEC 101/FINAL_PROJECT"
pip install -r requirements.txt
```
### 4. Running the Simulation
Once the dependencies are installed and the environment is active, run the main visualizer loop.

```Bash
python pacman_visualizer_loop.py
```
# Controls & Usage
When the simulation window opens, use the following keys to control the experiment:

* Press 1: Initialize Priority-Based Strategy.

* Press 2: Initialize Alternating Offers Strategy.

* Press 3: Initialize Q-Learning (RL) Strategy.

* Spacebar: Pause/Unpause the simulation.

# Data Output
The simulation automatically generates a Master Log file (CSV format) in the root directory upon completion of the trial batch (e.g., MITELEC101_Master_Log_Q_LEARNING.csv). This log contains frame-by-frame and summary metrics including:

* Efficiency: Average Turns per Round.

* Stability: Total Conflicts Detected.

* Fairness: Jain's Fairness Index.