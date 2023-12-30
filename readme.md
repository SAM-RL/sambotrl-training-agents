## Training SambotRL Agent with PPO-LSTM

### Training & Evaluation
Available agents include: DQN, PPO, PPO-LSTM. To train PPO-LSTM model, please run:
```
python3 train.py -m ppo_lstm -s 400_000
```
To evaluate existing model, please run:
```
python3 evaluate.py -m ppo_lstm
```

Trained LSTM-PPO model can be found in ``` ./output/``` directory. You can check ```conda_environment.yaml``` for all required dependencies in order to train our script in your environment.

### Overview
**1.  Proximal Policy Optimization**:

Proximal Policy Optimization (PPO) introduced by OpenAI in 2017, has been one of the most popular SOTA models in Reinforcement Learning for many years. Unlike existing Policy Gradient methods, PPO successfully balances sample complexity, implementation simplicity, and flexibility in fine-tuning. PPO has some benefits of Trust Region Policy Optimization (TRPO) - one of its predecessors, which aims to take the biggest improvement step on a policy being trained without accidentally stepping too far away from it and leading to performance collapse.
Key features of PPO includes:
- <ins>Clipped Surrogate Objective</ins>: modified objective for TRPO to address the issue of excessively large policy updates. The main objective incorporates a hyperparameter epsilon, which penalizes changes to the policy. The objective involves clipping the probability ratio to remove incentives for moving it outside a specific interval and taking the minimum of the clipped and unclipped objective.
- <ins>Adaptive KL Penalty Coefficient</ins>: additional trick that can be added to the clipped surrogate objective, which makes it possible to dynamically adapt the penalty coefficient to achieve a target KL divergence value in each policy update. In implementation, during each policy update, the KL-penalized objective is optimized using mini-batch SGD for several epochs, and the updated penalty coefficient β is used for the next update.

**2.  PPO-LSTM**:

LSTM (Long Short-Term Memory) is a recurrent neural network (RNN) architecture widely used in Deep Learning that is capable of capturing long-term dependencies and solving sequence-prediction tasks. PPO-LSTM is an RL model that combines both Proximal Policy Optimization (PPO) and Long Short-Term Memory (LSTM) in the agent policy network architecture, allowing the agent to utilize temporal state information from the environment for decision-making.

**3.  PPO-LSTM SambotRL Agent**:

In this project we trained our agent with PPO-LSTM model from [Stable-Baseline-3 Contrib](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html), with the following configurations:

- **Action Space**: action space is a discreet space with 9 possible action ("left", "right", "up", "down", "stay", \
            "up-left", "up-right", "down-left", "down-right")
- **Observation Space**: we embedded both field concentration information and image of agent exploration history as input to the network model, based on StableBaseline3's [MultiInputPolicies](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html). Our model input state is now a (259,1) vector which includes concentration information at the current location [concentration, gradient_x, gradient_y] and a (256,1) feature vector as output from a CNN feature extractor. In the actual implementation, since the MultiInputPolicies class already has its own internal CNN Feature Extractor, we can feed the agent history field as an image directly to our model.

Here is the shape of our observation space multi-input dictionary:
```
observation_space = spaces.Dict(
            {
                "concentration": spaces.Box(
                    low=[0.0, -100.0, -100.0],
                    high=[25.0, 100.0, 100.0],
                    dtype=np.float32,
                ),
                "agent-history-field": spaces.Box(
                    low=0,
                    high=255,
                    shape=(100, 100, 1),
                    dtype=np.uint8,
                ),
            }
        ) 
```
![Alt observation_space](images/observation_space.png?raw=true "Observation Space")
- **Reward Function**: we incorporate both per-step reward function & terminal reward function: 
	- <ins>Per-step reward</ins> is currently based on the improvement of concentration values inside the formation at the current location
	- <ins>Terminal reward</ins> is reward added when terminal state is reached. It is based on the similarity between the final mapping result w.r.t the actual field

**4. Additional Notes**:

Changes made to SambotRL's original gym environment:

- Diffusion Field & Agent Field are now their own separate classes (please check spatial_diffusion_field.py & agent_field.py) to improve readability & maintainability.
- Per-step diffusion field update for the field environment has been replaced with loading pre-recorded snapshots of environment state. This allows us to avoid calculating new updates to the environment for each timestep, which helps speed up training time.

### Results
(🔴 "Red Star" denotes agent starting position)
![Alt training_result_1](images/demo_result_1.png?raw=true "Result 1")
![Alt training_result_2](images/demo_result_2.png?raw=true "Result 2")

