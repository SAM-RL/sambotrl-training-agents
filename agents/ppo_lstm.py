import os
import torch as th
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO

class PPO_LSTM_Agent:

    def __init__(self, env, name='PPO_LSTM', path='./output', load_saved_model=True):
        self.env = env
        self.name = name
        self.model_path = os.path.join(path, name)
        self.checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f"./output/checkpoints/{name}/",
            name_prefix=self.name,
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        if load_saved_model and os.path.exists(self.model_path+'.zip'):
            self.model = RecurrentPPO.load(self.model_path, env=env)
        else:        
            self.model = RecurrentPPO('MultiInputLstmPolicy', env, verbose=1, tensorboard_log='./output/tensorboard')
    
    def train(self, n_timestep=40_000, eval=True):
        self.model.learn(total_timesteps=n_timestep, progress_bar=False, callback=[self.checkpoint_callback])
        self.model.save(self.model_path)
        if eval:
            mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10)
            print(f'Evaluation [{self.name}] ==> mean_reward: {mean_reward}, std_reward: {std_reward}')        

    def predict(self, obs, lstm_state=None, start=False):
        action, hidden_state  = self.model.predict(obs, state=lstm_state, episode_start=start)
        return action, hidden_state



