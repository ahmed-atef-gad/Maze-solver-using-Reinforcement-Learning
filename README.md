# Reinforcement Learning Maze Solver

This project implements maze-solving agents using reinforcement learning algorithms, including Q-learning and Policy Gradient methods. The environment features moving obstacles for added complexity.

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
- `agents/`: Contains the reinforcement learning agents
  - `q_learning.py`: Implements the Q-learning algorithm
  - `policy_gradient.py`: Implements the Policy Gradient algorithm with neural networks
- `train.py`: Main script for training the agent
- `evaluate.py`: Script for evaluating the trained agent
- `visualize.py`: Script for visualizing the Q-table
- `configs/`: Configuration files
  - `default.yaml`: Default training parameters

## Key Algorithms

### Q-learning

- A value-based reinforcement learning algorithm that learns optimal policies by estimating the value of state-action pairs
- Uses an epsilon-greedy strategy to balance exploration and exploitation
- Stores values in a Q-table for each state-action pair

### Policy Gradient

- A policy-based reinforcement learning algorithm that directly learns the policy function
- Uses neural networks to approximate the policy
- Optimizes the policy using gradient ascent on the expected rewards
- Handles continuous state spaces more effectively than Q-learning

### Common Features

- **Moving Obstacles**: Adds complexity with dynamic elements in the environment
- **Reward Shaping**: Uses distance-based rewards to guide learning
- **Collision Prediction**: Agents learn to predict and avoid collisions with moving obstacles

## Usage

1. install the required dependencies:

````bash
pip install -r requirements.txt
````

2. Train an agent:

````bash
python train.py
````

   This will train the specified agent according to the parameters in `configs/default.yaml` and save the model and training metrics to the `results/` directory.

3. Evaluate the trained agent:

````bash
python evaluate.py
````

   This will load the trained model and visualize the agent navigating through the maze.

4. Visualize the agent's learned policy:

````bash
python visualize.py 
````

   This will generate a visualization of the learned policy, showing how the agent navigates through the maze. For Q-learning, it displays the Q-values, while for Policy Gradient, it shows the action probabilities from the neural network.

## Configuration

You can modify the learning parameters in `configs/default.yaml`:

- `discount_factor`: How much future rewards are valued (0.99)
- `q_learning_episodes`: Number of training episodes for Q-learning (150,000)
- `pg_episodes`: Number of training episodes for Policy Gradient (1,000)
- `learning_rate`: How quickly the agent updates its Q-values (0.2)
- `epsilon`: Initial exploration rate (0.8)
- `epsilon_decay`: How quickly exploration decreases (0.995)
- `min_epsilon`: Minimum exploration rate (0.01)
- `pg_learning_rate`: Learning rate for Policy Gradient (0.01)
- `hidden_dim`: Dimension of hidden layers in neural networks (32)
- `batch_size`: Number of samples per gradient update (128)
- `memory_size`: Size of the experience replay buffer (10,000)
- `update_frequency`: Frequency of policy updates (10)

## Results

Training results are saved in the `results/` directory, with separate folders for each agent:

- `results/q_learning/`: Contains results specific to the Q-learning agent
- `results/policy_gradient/`: Contains results specific to the Policy Gradient agent

- Q-table values (Q-learning): The learned action values for each state
- Policy network (Policy Gradient): The trained neural network model
- Reward history: Total rewards per episode
- Steps history: Number of steps per episode
- Exploration rate: Epsilon value over time
- Exploration/exploitation decisions: Count of explore vs. exploit actions
