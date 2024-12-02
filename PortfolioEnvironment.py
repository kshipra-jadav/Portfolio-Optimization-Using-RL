import numpy as np

class PortfolioEnvironment:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.max_steps = len(data) - 1

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step]

    def step(self, action):
        self.current_step += 1
        done = self.current_step == self.max_steps
        obs = self.data.iloc[self.current_step]

        # Execute the action: Buy (1), Hold (0), or Sell (-1)
        if action == 0:
            # Hold the position
            reward = 0
        elif action == 1:
            # Buy
            reward = obs['close'] * 0.01 if not np.isnan(obs['close']) else 0  # Adjust the percentage as needed
        elif action == 2:
            # Sell
            reward = -obs['close'] * 0.01 if not np.isnan(obs['close']) else 0  # Adjust the percentage as needed
        else:
            raise ValueError("Invalid action")

        return obs, reward, done