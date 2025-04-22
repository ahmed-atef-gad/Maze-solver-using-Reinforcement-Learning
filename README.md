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
python train.py --agent q_learning
# or
python train.py --agent policy_gradient
````

   This will train the specified agent according to the parameters in `configs/default.yaml` and save the model and training metrics to the `results/` directory.

3. Evaluate the trained agent:

````bash
python evaluate.py --agent q_learning
# or
python evaluate.py --agent policy_gradient
````

   This will load the trained model and visualize the agent navigating through the maze.

4. Visualize the agent's learned policy:

````bash
# For Q-learning agent
python visualize.py --agent q_learning

# For Policy Gradient agent
python visualize.py --agent policy_gradient
````

   This will generate a visualization of the learned policy, showing how the agent navigates through the maze. For Q-learning, it displays the Q-values, while for Policy Gradient, it shows the action probabilities from the neural network.

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

- Q-table values (Q-learning): The learned action values for each state
- Policy network (Policy Gradient): The trained neural network model
- Reward history: Total rewards per episode
- Steps history: Number of steps per episode
- Exploration rate: Epsilon value over time
- Exploration/exploitation decisions: Count of explore vs. exploit actions

## Performance

On a standard CPU, training with 50,000-100,000 episodes takes approximately 15-30 minutes depending on your system's performance. The agent typically learns to solve the maze efficiently after 20,000-30,000 episodes.
