from collections import OrderedDict
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import time
import scipy.io as io
import random
import os
import pickle
from spatial_diffusion_field import SpatialDiffusionField, AgentField
from typing import Any, Dict, Optional, Tuple, Union

ACTION_MAP = {
    0:"left",1:"right",2:"up",3:"down", \
    4:"stay",5:"up-left",6:"up-right",7:"down-left",8:"down-right"
}

ACTIONS = ["left", "right", "up", "down", "stay", \
            "up-left", "up-right", "down-left", "down-right"]

ACTION_OFFSET = [[-1,0],[1,0],[0,-1],[0,1],[0,0],[-1,-1],[1,-1],[-1,1],[1,1]]

class SpatialTemporalFieldHER(gym.Env):
    """
    This environment is similar to the regular field, except the state vector contains the gradient
    and concentration values, but does not contain location, and has 9 actions
    """

    def __init__(
            self,
            data_path='./envs.pickle',
            output_dir='./output',
            max_num_steps=399,
            viewscope_size=5,
            detect_source_radius=8,
          ):
        # Open AI Gym stuff
        metadata = {'render.modes': ['human']}
        super(SpatialTemporalFieldHER,
              self).__init__()
        
        with open(data_path, 'rb') as handle:
            self.data = pickle.load(handle)
            self.fields_data = self.data['fields']
                    
        self.max_num_steps = max_num_steps
        self.viewscope_size = viewscope_size

        # Load Diffusion Field & Agent Field
        self.env_field = SpatialDiffusionField(fields_data=self.fields_data)
        self.agent_field = AgentField(params_dict=self.env_field.params)

        # Agent Field related params/variables
        self.num_steps = 0
        self.agent_position = None
        self.agent_trajectory = []

        # Statistics variables
        self.rewards = []
        self.mapping_errors = []
        self.concentrations = []
        self.gradients_0 = []
        self.gradients_1 = []
        # Action Space
        self.action_space_map = ACTION_MAP
        self.actions = ACTIONS
        self.action_space = spaces.Discrete(9)

        # Environment Observation space
        self.image_size = np.array([100, 100, 1])
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0,
                    high=255,
                    shape=self.image_size,
                    dtype=np.uint8,
                ),
                "achieved_goal": spaces.Box(
                    low=0,
                    high=100,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "desired_goal": spaces.Box(
                    low=0,
                    high=100,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )
        
        self.output_dir = output_dir
        self.init_mapping_error = np.sum(self.env_field.field)


    def calculate_gradients(self, r):
        dz_dx = (self.agent_field.vs_field[r[1], r[0] + 1] -
                 self.agent_field.vs_field[r[1], r[0] - 1]) / (2 * self.agent_field.dx)
        dz_dy = (self.agent_field.vs_field[r[1] + 1, r[0]] -
                 self.agent_field.vs_field[r[1] - 1, r[0]]) / (2 * self.agent_field.dy)

        return np.array([dz_dx, dz_dy])

    def step(self, action_id):
        # Ensure action is a valid action and exists in Agent's action space
        assert self.action_space.contains(action_id), "Action is invalid!"
      
        # Get the next state
        (hit_wall, next_position) = self.get_next_position(action_id)
        if (hit_wall):
            # Stay at the same place if hitting boundary
            next_position = self.agent_position
        
        # Update field state
        self.env_field.step()
        self.agent_field.update(env_field=self.env_field.field, pos=next_position)

        # Update Mapping error
        curr_mapping_error = self.calculate_mapping_error()
        self.mapping_errors.append(curr_mapping_error)

        # Update number of steps
        self.num_steps += 1

        # Get any observations
        observations = {"location": next_position}

        # Update agent variables
        self.agent_position = next_position
        self.agent_trajectory.append(self.agent_position)

        # Get concentration
        self.concentration = self.env_field.field[self.agent_position[1], self.agent_position[0]]
        self.agent_gradients = self.calculate_gradients(self.agent_position)

        # Check for termination criteria
        done = False
        if (self.num_steps >= self.max_num_steps):
            done = True
        
        achieved_goal = self.calculate_achieved_goal()
        reward = self.compute_reward([achieved_goal], [100], None)
        self.rewards.append(reward)

        # Record field values
        self.concentrations.append(self.concentration)
        self.gradients_0.append(self.agent_gradients[0])
        self.gradients_1.append(self.agent_gradients[1])
        
        next_state = OrderedDict(
            [
                ("observation", self.agent_field.get_viewscope_image()),
                ("achieved_goal", achieved_goal),
                ("desired_goal", 100),
            ]
        )

        # Return reward, next_state, done, observations
        return (next_state, reward, done, observations)

    def reset(self):
        # Reset agent related params
        self.num_steps = 0
        self.agent_position = self.choose_random_start_position()
        self.agent_trajectory = []
        self.agent_gradients = [0.0, 0.0]
        self.concentration = 0
        self.rewards = []
        self.mapping_errors = []
        self.concentrations = []
        self.gradients_0 = []
        self.gradients_1 = []
        self.env_field.reset()
        self.agent_field.reset()
        self.agent_field.update(self.env_field.field, self.agent_position)
        # Return the first state
        # Get concentration
        self.concentration = self.env_field.field[self.agent_position[1],
                                            self.agent_position[0]]

        # Get gradients
        self.agent_gradients = self.calculate_gradients(self.agent_position)

        # Record field values
        self.concentrations.append(self.concentration)
        self.gradients_0.append(self.agent_gradients[0])
        self.gradients_1.append(self.agent_gradients[1])

        achieved_goal = self.calculate_achieved_goal()
        next_state = OrderedDict(
            [
                ("observation", self.agent_field.get_viewscope_image()),
                ("achieved_goal", achieved_goal),
                ("desired_goal", 100),
            ]
        )
        return next_state
    
    def compute_reward(
        self, achieved_goal: Union[int, np.ndarray], desired_goal: Union[int, np.ndarray], _info: Optional[Dict[str, Any]]
    ) -> np.float32:
        diff = np.abs(achieved_goal[0] - desired_goal[0])
        return -(diff > 20).astype(np.float32)
    
    def calculate_achieved_goal(self):
        return np.sum(self.agent_field.vs_field)/np.sum(self.env_field.field)*100
    
    def calculate_mapping_error(self):
        return np.sum(np.abs(self.agent_field.vs_field - self.env_field.field))

    def choose_random_start_position(self):
        possible_starts = [[80, 90], [55,55], [75, 75], [40, 60], [80, 20], [20, 80]]
        return random.choice(possible_starts)

    def get_next_position(self, action_id):
        # Create a deepcopy of current state
        next_state = copy.deepcopy(self.agent_position)
        next_state = np.add(next_state, ACTION_OFFSET[action_id])

        # Check for collisions
        hit_wall = False
        if ((next_state[0] < (0 + self.agent_field.vs_hsize) or
             next_state[0] >= (self.agent_field.size - self.agent_field.vs_hsize)) or
            ((next_state[1] < (0 + self.agent_field.vs_hsize) or
              next_state[1] >= (self.agent_field.size - self.agent_field.vs_hsize)))):
            # If the view scope is out of the field, hit_wall is set to True
            hit_wall = True

        return (hit_wall, next_state)
    
        
    
    def view_env_state(self, episode_num, timestep, save=False, path=None):
        cmap_color = "Blues"
        fig_learning, fig_learning_axes = plt.subplots(2, 3, figsize=(20, 15))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        fig_learning_axes[0, 0].set_title("Environment Field End State")
        fig_learning_axes[0, 0].set_aspect("equal")

        fig_learning_axes[0, 1].set_title("Agent Field End State")
        fig_learning_axes[0, 1].set_aspect("equal")

        fig_learning_axes[0, 2].set_title("Concentration")
        fig_learning_axes[1, 2].set_xlim([0, self.max_num_steps])

        fig_learning_axes[1, 0].set_title("Mapping Error")
        fig_learning_axes[1, 0].set_xlim([0, self.max_num_steps])

        fig_learning_axes[1, 1].set_title("gradient_x")
        fig_learning_axes[1, 1].set_xlim([0, self.max_num_steps])

        fig_learning_axes[1, 2].set_title("gradient_y")
        fig_learning_axes[1, 2].set_xlim([0, self.max_num_steps])

        # Plot 1: Environment End state
        fig_learning_axes[0, 0].imshow(
            self.env_field.field, cmap=cmap_color)

        traj_r = [position[0] for position in self.agent_trajectory]
        traj_c = [position[1] for position in self.agent_trajectory]
        fig_learning_axes[0, 0].plot(traj_r, traj_c, '.', color='black')

        view_scope_box = patches.Rectangle(
            (self.agent_trajectory[-1][0] - 5,
             self.agent_trajectory[-1][1] - 5), 11, 11,
            linewidth=2, edgecolor='r', facecolor='none')
        fig_learning_axes[0, 0].add_patch(view_scope_box)

        if (path is not None):
            path_traj_r = [position[0] for position in path]
            path_traj_c = [position[1] for position in path]
            fig_learning_axes[0, 0].plot(
                path_traj_r, path_traj_c, '.', color='red')

        fig_learning_axes[0, 0].plot(
            self.agent_trajectory[0][0], self.agent_trajectory[0][1], '*', color='red')
        # Plot 2: Agent Field End state
        fig_learning_axes[0, 1].imshow(
            self.agent_field.vs_field, cmap=cmap_color)

        fig_learning_axes[0, 1].plot(traj_r, traj_c, '.', color='black')

        fig_learning_axes[0, 1].plot(
            self.agent_trajectory[0][0], self.agent_trajectory[0][1], '*', color='red')
        # Plot 3: Concentrations at center
        fig_learning_axes[0, 2].plot(self.concentrations, '.-')
        # Plot 4: Mapping Error
        fig_learning_axes[1, 0].plot(self.mapping_errors, '.-')
        # Plot 5: Mapping Error
        fig_learning_axes[1, 1].plot(self.gradients_0, '.-')
        # Plot 6: Coverage percentage
        fig_learning_axes[1, 2].plot(self.gradients_1, '.-')

        # Add Episode number to top of image
        fig_learning.suptitle(
            "Test Episode: " + str(episode_num) + ", at timestep: " + str(timestep))

        # Save image to directory
        if save:
            fig_file_name = "test_episode_" + str(episode_num) + ".png"
            plt.savefig(os.path.join(self.path_to_output_dir, fig_file_name))
        else:
            plt.show()

        plt.close()


    def render(self, mode="human"):
        pass
