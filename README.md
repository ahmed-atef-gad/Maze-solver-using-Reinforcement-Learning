# Reinforcement Learning Maze Solver

This project implements a maze-solving agent using Q-learning reinforcement learning.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone this repository or download the source code.
2. Navigate to the project directory:
3. Install the required dependencies:

## Components

- `maze_env/`: Contains the maze environment implementation
  - `maze.py`: Defines the maze and reward structure
  - `render.py`: Visualizes the maze and agent
- `agents/`: Contains the Q-learning agent
  - `q_learning.py`: Implements the Q-learning algorithm
- `train.py`: Main script for training the agent
- `evaluate.py`: Script for evaluating the trained agent
- `visualize.py`: Script for visualizing the Q-table
- `configs/`: Configuration files
  - `default.yaml`: Default training parameters

## Key Algorithms

- **Q-learning**: A value-based reinforcement learning algorithm that learns optimal policies by estimating the value of state-action pairs
- **Epsilon-greedy Strategy**: Balances exploration and exploitation
- **Reward Shaping**: Uses distance-based rewards to guide learning
- **Experience Replay**: Reuses past experiences to improve learning efficiency

## Usage
1. install the required dependencies:

````bash
pip install -r requirements.txt
````

2. Train the agent:

````bash
python train.py
````

   This will train the agent according to the parameters in `configs/default.yaml` and save the Q-table and training metrics to the `results/` directory.

3. Evaluate the trained agent:
````bash
python evaluate.py
````
   This will load the trained Q-table and visualize the agent navigating through the maze.

4. Visualize the Q-table:
````bash
python visualize.py
````
   This will generate a visualization of the learned Q-values, showing the policy the agent has learned.

## Configuration

You can modify the learning parameters in `configs/default.yaml`:

- `learning_rate`: How quickly the agent updates its Q-values
- `discount_factor`: How much future rewards are valued
- `epsilon`: Initial exploration rate
- `epsilon_decay`: How quickly exploration decreases
- `min_epsilon`: Minimum exploration rate
- `episodes`: Number of training episodes

## Results

Training results are saved in the `results/` directory:

- Q-table values: The learned action values for each state
- Reward history: Total rewards per episode
- Steps history: Number of steps per episode
- Exploration rate: Epsilon value over time
- Exploration/exploitation decisions: Count of explore vs. exploit actions

## Performance

On a standard CPU, training with 50,000-100,000 episodes takes approximately 15-30 minutes depending on your system's performance. The agent typically learns to solve the maze efficiently after 20,000-30,000 episodes.
