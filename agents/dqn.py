import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

class DQN_Agent:

    def __init__(self, env, name='DQN', path='./output', load_saved_model=True):
        self.env = env
        self.name = name
        self.model_path = os.path.join(path, name)
        if load_saved_model and os.path.exists(self.model_path):
           self.model = DQN.load(self.model_path, env=env)
        else:
            self.model = DQN('CnnPolicy', env, verbose=1)
    
    def train(self, n_timestep=40_000, eval=True):
        self.model.learn(total_timesteps=n_timestep)
        self.model.save(self.model_path)
        if eval:
            mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10)
            print(f'Evaluation ==> mean_reward: {mean_reward}, std_reward: {std_reward}')

    def get_env(self):
        return self.model.get_env()     

    def predict(self, obs, lstm_state=None, start=False):
        action, _ = self.model.predict(np.array(obs))
        return action, _



