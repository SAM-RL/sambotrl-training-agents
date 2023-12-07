import gym
import gym_field
import argparse
import numpy as np
import agents

DEFAULT_ENV = 'field-9-actions-image-v0'
DEFAULT_MODEL = 'ppo_lstm'
DEFAULT_EXPERIMENT_NAME = 'dqn-no-loc-9-action-with-grad-reward'
DEFAULT_ENV_OUTPUT_DIR = "gym_output"
DEFAULT_N_SOURCE = 2
DEFAULT_N_STEPS = 40_000
DEFAULT_TEST_EPISODES = 5
ADV_DIFF_PARAMS = {"vx": -0.6, "vy": 0.8}

model_dict = {
    'dqn': agents.DQN_Agent,
    'ppo_lstm': agents.PPO_LSTM_Agent,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, choices=['dqn', 'ppo_lstm'], help="model to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV, help="environment name")
    parser.add_argument("-n", "--name", default=DEFAULT_EXPERIMENT_NAME, help="experiment name")
    parser.add_argument("-s", "--steps", default=DEFAULT_N_STEPS, help="training timesteps", type=int)

    args = parser.parse_args()
    
    # setup environment
    env = gym.make(args.env)
    
    # train & evaluate stable-baseline3 model
    model = model_dict[args.model](env)
    model.train(n_timestep=args.steps)

    # evaluate
    for episode_num in range(DEFAULT_TEST_EPISODES):
        obs = env.reset()
        hidden_state = None
        done = True # use this signal to reset LSTM state
        total_reward = 0.0
        steps = 0
        while True:
            action, hidden_state = model.predict(obs, lstm_state=hidden_state, start=done)
            obs, reward, done, observation = env.step(action.item())
            if done:
                env.view_env_state(episode_num, steps)
                break
            steps += 1

