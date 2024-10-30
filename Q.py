import numpy as np
import random
from typing import Tuple, Dict

class GridWorld:
    def __init__(self):
        self.grid_size = 2
        self.state = (0, 0)  # Start position
        self.treasure = (1, 1)  # Treasure position
        
    def reset(self) -> Tuple[int, int]:
        self.state = (0, 0)
        return self.state
    
    def step(self, action: str) -> Tuple[Tuple[int, int], float, bool]:
        # Current position
        x, y = self.state
        
        # Move according to action
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.grid_size - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.grid_size - 1, y + 1)
            
        self.state = (x, y)
        
        # Check if we found treasure
        done = self.state == self.treasure
        reward = 10 if done else -1  # 10 for treasure, -1 for each move
        
        return self.state, reward, done

class QLearning:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = {}
        self.lr = learning_rate  # α (alpha)
        self.gamma = discount_factor  # γ (gamma)
        self.epsilon = epsilon  # For exploration
        self.actions = ['up', 'down', 'left', 'right']
        
        # Initialize Q-table with zeros
        for i in range(2):
            for j in range(2):
                self.q_table[f'state_{i}_{j}'] = {
                    action: 0 for action in self.actions
                }
    
    def get_state_key(self, state: Tuple[int, int]) -> str:
        return f'state_{state[0]}_{state[1]}'
    
    def choose_action(self, state: Tuple[int, int]) -> str:
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        state_key = self.get_state_key(state)
        return max(self.q_table[state_key].items(), key=lambda x: x[1])[0]
    
    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Get the maximum Q-value for the next state
        next_max_q = max(self.q_table[next_state_key].values())
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Q-learning formula
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        
        # Update Q-table
        self.q_table[state_key][action] = new_q

def train(episodes=1000):
    env = GridWorld()
    agent = QLearning()
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Choose and take action
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Learn from the experience
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            
        if episode % 100 == 0:
            print(f"Episode {episode}: Q-table:")
            for state in agent.q_table:
                print(f"{state}: {agent.q_table[state]}")

# Run training
if __name__ == "__main__":
    train()