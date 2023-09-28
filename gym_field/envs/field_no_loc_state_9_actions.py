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


class SpatialTemporalFieldNoLocStateWithGradRewardConcNineActions(gym.Env):
    """
    This environment is similar to the regular field, except the state vector contains the gradient
    and concentration values, but does not contain location, and has 9 actions
    """

    def __init__(
            self,
            learning_experiment_name,
            output_dir,
            field_size=[100, 100],
            max_num_steps=300,
            view_scope_half_side=5,
            testing_field=None,
            num_sources=2,
            with_coverage_field=True
            coverage_field_size=[5, 5],
            adv_diff_params={}):
        # Open AI Gym stuff
        metadata = {'render.modes': ['human']}
        super(SpatialTemporalFieldNoLocStateWithGradRewardConcNineActions,
              self).__init__()

        # Set advection diffusion parameters
        self.dx = adv_diff_params.get("dx", 0.8)
        self.dy = adv_diff_params.get("dy", 0.8)
        self.vx = adv_diff_params.get("vx", 0.7)  # 3: -0.7, 4: 0.7, 2: -0.6
        self.vy = adv_diff_params.get("vy", -0.4)  # 3: -0.3, 4: -0.4, 2: 0.8
        self.dt = adv_diff_params.get("dt", 0.1)
        self.k = adv_diff_params.get("k", 1.0)

        print("Advection Diffusion parameters: ")
        print("dx, dy, vx, vy, dt, k: " +
              str((self.dx, self.dy, self.vx, self.vy, self.dt, self.k)))
        
        # NEW PARAM
        self.with_coverage_field = with_coverage_field
        self.coverage_field_size = coverage_field_size
        
        # Set field grid params/variables
        self.field_size = field_size
        self.field_area = self.field_size[0] * self.field_size[1]
        self.max_num_steps = max_num_steps
        self.view_scope_half_side = view_scope_half_side

        # Environment's field params
        self.testing_field = testing_field
        self.num_sources = num_sources
        self.env_curr_field = None
        self.source_template = self.load_source()
        self.template_peak_conc = np.max(self.source_template)

        if self.testing_field is not None:
            self.env_curr_field = self.create_test_field(self.testing_field)
        else:
            print("Num sources: " + str(num_sources))
            if (self.num_sources == 2):
                self.env_curr_field = self.create_field()
            elif (self.num_sources == 3):
                self.env_curr_field = self.create_field_3_sources()
            elif (self.num_sources == 4):
                self.env_curr_field = self.create_field_4_sources()

        self.env_prev_field = np.zeros(self.field_size)

        # Agent Field related params/variables
        self.num_steps = 0
        self.agent_start_position = None
        self.agent_position = None
        self.agent_curr_field = np.zeros(self.field_size)
        self.agent_field_visited = np.zeros(self.field_size)
        self.agent_trajectory = []
        self.actions_list = []
        self.curr_view_scope = np.zeros(
            [2 * self.view_scope_half_side + 1, 2 * self.view_scope_half_side + 1])
        self.agent_gradients = [0.0, 0.0]

        # Statistics variables
        self.agent_coverage = []
        self.rewards = []
        self.mapping_errors = []
        self.concentrations = []
        self.gradients_0 = []
        self.gradients_1 = []

        # TTOO DDOO: Environment Observation space. This includes x, y, conc, grad_x, grad_y as state
        low = np.array([0.0, -100.0, -100.0])
        high = np.array([25.0, 100.0, 100.0])
        if self.with_coverage_field:
            flatten_coverage_fields_length = (self.coverage_field_size*self.coverage_field_size)*2
            low = np.concatenate([low, (-1) * np.ones(flatten_coverage_fields_length)])
            high = np.concatenate([high, np.ones(flatten_coverage_fields_length)])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Agent Action related params/variables
        self.action_space_map = {}
        self.actions = ["left", "right", "up", "down", "stay",
                        "up-left", "up-right", "down-left", "down-right"]
        self.action_space = spaces.Discrete(9)
        for action_id, action in enumerate(self.actions):
            self.action_space_map[action_id] = action

        # Misc. params
        self.output_dir = output_dir
        self.learning_experiment_name = learning_experiment_name

        # Make output_dir/learning_experiment_name folder if it doesn't exist already
        self.path_to_output_dir = os.path.join(
            self.output_dir, self.learning_experiment_name)
        if not os.path.exists(self.path_to_output_dir):
            os.makedirs(self.path_to_output_dir)
            print("Output directory " + self.path_to_output_dir + " created.")
        else:
            print("Output directory " +
                  self.path_to_output_dir + " already existed!")

        # Initial mapping error
        self.init_mapping_error = np.sum(self.env_curr_field)

    def create_field(self):
        cwd = os.getcwd()
        loaded_mat = io.loadmat(cwd + "/u.mat")
        u = loaded_mat.get('u')

        u_1 = u.copy()
        source_1_c_shift = 13

        for r in range(0, self.field_size[0]):
            for c in range(0, self.field_size[1]):
                u_1[r, c] = u[r % self.field_size[0],
                              (c - source_1_c_shift) % self.field_size[1]]

        reverse_u = u.copy()

        for r in range(0, self.field_size[0]):
            for c in range(0, self.field_size[1]):
                reverse_u[r, c] = u[self.field_size[0] - r - 1,
                                    self.field_size[1] - c - 1]

        reverse_u_1 = u.copy()
        source_2_r_shift = 10
        source_2_c_shift = 2

        # fmt: off
        for r in range(0, self.field_size[0]):
            for c in range(0, self.field_size[1]):
                reverse_u_1[r, c] = reverse_u[(r + source_2_r_shift) % self.field_size[0],
                                              (c + source_2_c_shift) % self.field_size[1]]
        # fmt: on
        combined_field = u_1 + reverse_u_1
        return combined_field

    def load_source(self):
        cwd = os.getcwd()
        loaded_mat = io.loadmat(cwd + "/u.mat")
        u = loaded_mat.get('u')

        # Find center of source template
        max_loc = np.unravel_index(np.argmax(u), u.shape)

        # Move the center to [50, 50]
        centered = u.copy()
        row_shift = 50 - max_loc[0]
        col_shift = 50 - max_loc[1]

        for r in range(0, self.field_size[0]):
            for c in range(0, self.field_size[1]):
                centered[r, c] = u[(r - row_shift) % self.field_size[0],
                                   (c - col_shift) % self.field_size[1]]
        return centered

    def add_source(self, source_location, concentration):
        # Make a copy of the template
        new_source = self.source_template.copy()

        # Find the factor to reduce concentration by and scale the array
        conc_reduction_factor = concentration / self.template_peak_conc
        new_source *= conc_reduction_factor

        # Move source to desired location
        new_source_copy = new_source.copy()
        row_shift = source_location[0] - 50
        col_shift = source_location[1] - 50

        for r in range(0, self.field_size[0]):
            for c in range(0, self.field_size[1]):
                new_source[r, c] = new_source_copy[(r - row_shift) % self.field_size[0],
                                                   (c - col_shift) % self.field_size[1]]

        # Add the new source to existing field
        return new_source

    def create_field_3_sources(self):
        print("Creating 3 source field...")
        field = np.zeros(self.field_size)

        source1 = [45, 30]
        conc1 = 20

        source2 = [20, 60]
        conc2 = 25

        source3 = [80, 66]
        conc3 = 17

        field += self.add_source(source1, conc1)
        field += self.add_source(source2, conc2)
        field += self.add_source(source3, conc3)
        return field

    def create_field_4_sources(self):
        print("Creating 4 source field...")
        field = np.zeros(self.field_size)

        # Field-3
        source1 = [45, 30]
        conc1 = 20

        source2 = [20, 60]
        conc2 = 25

        source3 = [80, 66]
        conc3 = 17

        source4 = [65, 25]
        conc4 = 22

        field += self.add_source(source1, conc1)
        field += self.add_source(source2, conc2)
        field += self.add_source(source3, conc3)
        field += self.add_source(source4, conc4)
        return field

    def create_test_field(self, num):
        field = np.zeros(self.field_size)
        if (num == 1):
            print("Creating testing field-1")
            # Testing field-1
            source1 = [30, 30]
            conc1 = 25

            # source2 = [50, 60]
            # conc2 = 25

            source3 = [75, 66]
            conc3 = 17

            source4 = [65, 25]
            conc4 = 22

            field += self.add_source(source1, conc1)
            # field += self.add_source(source2, conc2)
            field += self.add_source(source3, conc3)
            field += self.add_source(source4, conc4)
            return field
        else:
            print("Creating testing field-2")
            # Testing field-1
            source1 = [20, 30]
            conc1 = 23

            source2 = [45, 70]
            conc2 = 25

            source3 = [75, 35]
            conc3 = 20

            field += self.add_source(source1, conc1)
            field += self.add_source(source2, conc2)
            field += self.add_source(source3, conc3)
            return field

    def update_env_field(self):
        updated_u = self.env_curr_field.copy()
        u_k = self.env_curr_field.copy()

        dx = self.dx
        dy = self.dy
        dt = self.dt
        vx = self.vx
        vy = self.vy
        k = self.k

        # fmt: off
        for i in range(1, self.field_size[0] - 1):
            for j in range(1, self.field_size[1] - 1):
                updated_u[j, i] = u_k[j, i] + k * (dt / dx ** 2) * \
                    ((u_k[j + 1, i] + u_k[j - 1, i] +
                      u_k[j, i + 1] + u_k[j, i - 1] - 4 * u_k[j, i])) + \
                    vx * (dt / dx) * ((u_k[j + 1, i] - u_k[j, i])) + vy * (dt / dy) * \
                    (u_k[j, i + 1] - u_k[j, i])
        # fmt: on                                                                                                                                                                                          i])

        # self.env_prev_field = self.env_curr_field
        self.env_curr_field = updated_u

    def update_field(self, field, params):
        updated_u = field.copy()
        u_k = field.copy()

        dx = params["dx"]
        dy = params["dy"]
        dt = params["dt"]
        vx = params["vx"]
        vy = params["vy"]
        k = params["k"]

        # fmt: off
        for i in range(1, self.field_size[0] - 1):
            for j in range(1, self.field_size[1] - 1):
                updated_u[j, i] = u_k[j, i] + k * (dt / dx ** 2) * \
                    ((u_k[j + 1, i] + u_k[j - 1, i] +
                      u_k[j, i + 1] + u_k[j, i - 1] - 4 * u_k[j, i])) + \
                    vx * (dt / dx) * ((u_k[j + 1, i] - u_k[j, i])) + vy * (dt / dy) * \
                    (u_k[j, i + 1] - u_k[j, i])
        # fmt: on                                                                                                                                                                                          i])

        # self.env_prev_field = self.env_curr_field
        return updated_u

    def calculate_gradients(self, r):
        dz_dx = (self.agent_curr_field[r[0] + 1, r[1]] -
                 self.agent_curr_field[r[0] - 1, r[1]]) / (2 * self.dx)
        dz_dy = (self.agent_curr_field[r[0], r[1] + 1] -
                 self.agent_curr_field[r[0], r[1] - 1]) / (2 * self.dy)

        return np.array([dz_dx, dz_dy])

    def step(self, action_id):
        # Ensure action is a valid action and exists in Agent's action space
        assert self.action_space.contains(
            action_id), "Action %r (%s) is invalid!" % (action_id, type(action_id))

        action = self.action_space_map[action_id]
        assert action in self.actions, "%s (%s) invalid" % (
            action, type(action))

        self.actions_list.append(action_id)

        # Get the next state
        (hit_wall, next_position) = self.get_next_position(action)
        if (hit_wall):
            # Terminate the episode, and return a large negative reward
            reward = -1000
            self.rewards.append(reward)
            # Get concentration
            concentration = self.env_curr_field[self.agent_position[0],
                                                self.agent_position[1]]
            # Get gradients
            self.agent_gradients = self.calculate_gradients(
                self.agent_position)

            # Record field values
            self.concentrations.append(concentration)
            self.gradients_0.append(self.agent_gradients[0])
            self.gradients_1.append(self.agent_gradients[1])

            observation = {"location": self.agent_position}

            return ([concentration, self.agent_gradients[0], self.agent_gradients[1]], reward, True, observation)

        # Update field state
        self.update_env_field()

        # Update agent's view of the field
        frac_coverage_improvement = self.update_agent_field_and_coverage(
            next_position)

        # Update Mapping error
        curr_mapping_error = self.calculate_mapping_error()
        self.mapping_errors.append(curr_mapping_error)

        # Get gradients
        self.agent_gradients = self.calculate_gradients(self.agent_position)

        # Get the new reward
        reward = self.calculate_reward_3(
            next_position, self.agent_gradients)

        # Update number of steps
        self.num_steps += 1

        # Check for termination criteria
        done = False
        if (self.num_steps >= self.max_num_steps):
            done = True
            reward += 1000

        self.rewards.append(reward)

        # Get any observations
        observations = {"location": next_position}

        # Update agent variables
        self.agent_position = next_position
        self.agent_trajectory.append(self.agent_position)

        # Get concentration
        concentration = self.env_curr_field[self.agent_position[0],
                                            self.agent_position[1]]

        # Record field values
        self.concentrations.append(concentration)
        self.gradients_0.append(self.agent_gradients[0])
        self.gradients_1.append(self.agent_gradients[1])
        
        next_state = [concentration, self.agent_gradients[0], self.agent_gradients[1]]
        if self.with_coverage_field:
            visited_field = (self.env_curr_field > 0).astype(float)
            global_field, local_field = self.get_ego_coverage_fields(visited_field, \
                                                                self.agent_position, self.coverage_field_size)
            next_state = np.concatenate([next_state, global_field.flatten(), local_field.flatten()])

        # Return reward, next_state, done, observations
        return (next_state, reward, done, observations)

    def reset(self):
        # Reset agent related params
        self.num_steps = 0
        self.agent_position = self.choose_random_start_position()
        self.agent_curr_field = np.zeros(self.field_size)
        self.agent_field_visited = np.zeros(self.field_size)
        self.agent_trajectory = []
        self.curr_view_scope = np.zeros(
            [2 * self.view_scope_half_side + 1, 2 * self.view_scope_half_side + 1])
        self.agent_gradients = [0.0, 0.0]

        # Reset environment related params
        if self.testing_field is not None:
            self.env_curr_field = self.create_test_field(self.testing_field)
        else:
            # print("Num sources: " + str(self.num_sources))
            if (self.num_sources == 2):
                self.env_curr_field = self.create_field()
            elif (self.num_sources == 3):
                self.env_curr_field = self.create_field_3_sources()
            elif (self.num_sources == 4):
                self.env_curr_field = self.create_field_4_sources()

        # Reset stats
        self.agent_coverage = []
        self.rewards = []
        self.mapping_errors = []
        self.concentrations = []
        self.gradients_0 = []
        self.gradients_1 = []

        # Return the first state
        # Get concentration
        concentration = self.env_curr_field[self.agent_position[0],
                                            self.agent_position[1]]

        # Get gradients
        self.agent_gradients = self.calculate_gradients(self.agent_position)

        # Record field values
        self.concentrations.append(concentration)
        self.gradients_0.append(self.agent_gradients[0])
        self.gradients_1.append(self.agent_gradients[1])

        return [concentration, self.agent_gradients[0], self.agent_gradients[1]]

    def choose_random_start_position(self):
        # return [np.random.randint(self.view_scope_half_side + 1, self.field_size[0] - self.view_scope_half_side - 1),
        #         np.random.randint(self.view_scope_half_side + 1, self.field_size[1] - self.view_scope_half_side - 1)]
        # return [np.random.randint(8, 13),
        #         np.random.randint(30, 35)]
        # TODO (Deepak): Change this
        # possible_starts = [[8, 30], [11, 35], [10, 34], [
        #     80, 80], [60, 80], [80, 60], [60, 70], [40, 40]]
        # possible_starts = [[36, 83]]
        possible_starts = [[80, 90]]
        return random.choice(possible_starts)

    def get_next_position(self, action):
        # Create a deepcopy of current state
        next_state = copy.deepcopy(self.agent_position)

        # Only update the next_state if the action changes the position, else stay
        if action == "left":
            next_state[1] = next_state[1] - 1
        elif action == "right":
            next_state[1] = next_state[1] + 1
        elif action == "up":
            next_state[0] = next_state[0] - 1
        elif action == "down":
            next_state[0] = next_state[0] + 1
        elif action == "stay":
            pass
        elif action == "up-left":
            next_state[0] = next_state[0] - 1
            next_state[1] = next_state[1] - 1
        elif action == "up-right":
            next_state[0] = next_state[0] - 1
            next_state[1] = next_state[1] + 1
        elif action == "down-left":
            next_state[0] = next_state[0] + 1
            next_state[1] = next_state[1] - 1
        elif action == "down-right":
            next_state[0] = next_state[0] + 1
            next_state[1] = next_state[1] + 1

        # Check for collisions
        hit_wall = False
        if ((next_state[0] < (0 + self.view_scope_half_side) or
             next_state[0] >= (self.field_size[0] - self.view_scope_half_side)) or
            ((next_state[1] < (0 + self.view_scope_half_side) or
              next_state[1] >= (self.field_size[1] - self.view_scope_half_side)))):
            # If the view scope is out of the field, hit_wall is set to True
            hit_wall = True

        return (hit_wall, next_state)
    
    def get_ego_coverage_fields(self, field, pos=[50,50], output_shape=(5,5)):
        field_half_size = field.shape[0] // 2
        block_shape = (field.shape[0] // output_shape[0], field.shape[1] // output_shape[1])
        padded_field = np.pad(field, (field_half_size, field_half_size), 'constant', constant_values=1)
        ego_field = padded_field[pos[0]:pos[0]+field_half_size*2, \
                                 pos[1]:pos[1]+field_half_size*2]
        blocks = ego_field.reshape((ego_field.shape[0]//block_shape[0], block_shape[0], \
                                 ego_field.shape[1]//block_shape[1], block_shape[1]))
        global_coverage_field = blocks.sum(axis=(1,3)) / (block_shape[0] * block_shape[1])
        local_coverage_field = padded_field[pos[0]+field_half_size-(output_shape[0]//2) \
                                           :pos[0]+field_half_size+(output_shape[0]//2 \
                                                                  +output_shape[0]%2), \
                                           pos[1]+field_half_size-(output_shape[1]//2) \
                                           :pos[1]+field_half_size+(output_shape[1]//2 \
                                                                  +output_shape[1]%2)]
    return global_coverage_field, local_coverage_field

    def calculate_reward_1(self, next_state, frac_coverage_improvement):
        prev_mapping_error = 0
        if len(self.mapping_errors) != 0:
            prev_mapping_error = self.mapping_errors[-1]
        else:
            prev_mapping_error = self.init_mapping_error

        curr_mapping_error = self.calculate_mapping_error()

        self.mapping_errors.append(curr_mapping_error)

        mapping_error_improvement = prev_mapping_error - curr_mapping_error

        reward = (0.0001 * mapping_error_improvement) + \
            (2 * frac_coverage_improvement)
        return reward

    def calculate_reward_2(self, next_state, frac_coverage_improvement):
        """
        Assuming that the reward is only proportional to what is being copied by the viewscope.
        Coverage not considered.
        """
        return 1e-2 * np.sum(self.curr_view_scope)

    def calculate_reward_3(self, next_state, gradients):
        """
        Assuming that the reward is only proportional to what is being copied by the viewscope.
        Coverage not considered.
        """
        reward_from_vs = 1e-2 * np.sum(self.curr_view_scope)

        if reward_from_vs < 10:
            return reward_from_vs
        else:
            sum_sq_grad = (gradients[0] ** 2) + (gradients[1] ** 2)
            reward_from_grad = 20 * np.exp(-5 * sum_sq_grad)
            return reward_from_vs + reward_from_grad

    def normalize(self, field):
        max_val = field.max()
        min_val = field.min()
        field_normalized = (field - min_val) / (max_val - min_val)
        return field_normalized

    def calculate_mapping_error(self):
        return np.sum(np.abs(self.agent_curr_field - self.env_curr_field))

    def update_agent_field_and_coverage(self, next_state):
        vs_min_row = next_state[0] - self.view_scope_half_side
        vs_max_row = next_state[0] + self.view_scope_half_side + 1

        vs_min_col = next_state[1] - self.view_scope_half_side
        vs_max_col = next_state[1] + self.view_scope_half_side + 1

        # Count prev_visited
        prev_visited = np.count_nonzero(self.agent_field_visited)

        # self.curr_view_scope = np.zeros(
        #     [vs_max_row - vs_min_row, vs_max_col - vs_min_col])

        # for r in range(vs_min_row, vs_max_row):
        #     for c in range(vs_min_col, vs_max_col):
        #         self.agent_curr_field[r, c] = self.env_curr_field[r, c]
        #         self.agent_field_visited[r, c] = 1
        self.agent_curr_field[vs_min_row:vs_max_row, vs_min_col:vs_max_col] = \
            self.env_curr_field[vs_min_row:vs_max_row, vs_min_col:vs_max_col]

        self.agent_field_visited[vs_min_row:vs_max_row,
                                 vs_min_col:vs_max_col] = 1

        self.curr_view_scope = self.agent_curr_field[vs_min_row:vs_max_row,
                                                     vs_min_col:vs_max_col]

        # Update agent field state
        # Only need this in testing mode

        # testing field-1
        # vx: 0.65,    0.64
        # vy: -0.45,   -0.46
        # testing field-2
        # vx: -0.8,      -0.81
        # vy: 0.4,      0.42

        field_params = {
            "dx": 0.8,
            "dy": 0.8,
            "vx": -0.56,  # 3: -0.7, 4: 0.7, 2: -0.6
            "vy": 0.84,    # 3: -0.3, 4: -0.4, 2: 0.8
            "dt": 0.1,
            "k": 1.0
        }

        # Estimated params
        # 4: 0.68, -0.44
        # 3: -0.71, -0.27
        # 2: -0.56, 0.84

        self.agent_curr_field = self.update_field(
            self.agent_curr_field, field_params)

        # Count curr_visited
        curr_visited = np.count_nonzero(self.agent_field_visited)
        frac_coverage_improvement = float(
            curr_visited) - float(prev_visited) / float(self.field_area)

        # Record coverage percentage
        self.agent_coverage.append(
            (float(curr_visited) * 100.0) / float(self.field_area))
        return frac_coverage_improvement

    def save_episode_state(self, episode_num):
        fig_learning, fig_learning_axes = plt.subplots(2, 3, figsize=(15, 10))
        fig_learning_axes[0, 0].set_title("Environment Field End State")
        fig_learning_axes[0, 0].set_aspect("equal")

        fig_learning_axes[0, 1].set_title("Agent Field End State")
        fig_learning_axes[0, 1].set_aspect("equal")

        fig_learning_axes[0, 2].set_title("States visited")
        fig_learning_axes[0, 2].set_aspect("equal")

        fig_learning_axes[1, 0].set_title("Reward")
        # fig_learning_axes[1, 0].set_aspect("equal", adjustable='box')
        fig_learning_axes[1, 0].set_xlim([0, self.max_num_steps])

        fig_learning_axes[1, 1].set_title("Mapping Error")
        # fig_learning_axes[1, 1].set_aspect("equal", adjustable='box')
        fig_learning_axes[1, 1].set_xlim([0, self.max_num_steps])

        fig_learning_axes[1, 2].set_title("Field Coverage")
        # fig_learning_axes[1, 2].set_aspect("equal", adjustable='box')
        fig_learning_axes[1, 2].set_xlim([0, self.max_num_steps])
        fig_learning_axes[1, 2].set_ylim([0, 100])

        # Plot 1: Environment End state
        fig_learning_axes[0, 0].imshow(
            self.env_curr_field.T, cmap="Blues")

        traj_r = [position[0] for position in self.agent_trajectory]
        traj_c = [position[1] for position in self.agent_trajectory]
        fig_learning_axes[0, 0].plot(traj_r, traj_c, '.', color='black')

        fig_learning_axes[0, 0].plot(
            self.agent_trajectory[0][0], self.agent_trajectory[0][1], '*', color='red')

        # Plot 2: Agent Field End state
        fig_learning_axes[0, 1].imshow(
            self.agent_curr_field.T, cmap="Blues")

        fig_learning_axes[0, 1].plot(traj_r, traj_c, '.', color='black')

        fig_learning_axes[0, 1].plot(
            self.agent_trajectory[0][0], self.agent_trajectory[0][1], '*', color='red')

        # Plot 3: Visited
        fig_learning_axes[0, 2].imshow(
            self.agent_field_visited.T, cmap="Blues")

        fig_learning_axes[0, 2].plot(traj_r, traj_c, '.', color='black')

        fig_learning_axes[0, 2].plot(
            self.agent_trajectory[0][0], self.agent_trajectory[0][1], '*', color='red')

        # Plot 4: Reward
        fig_learning_axes[1, 0].plot(self.rewards, '.-')

        # Plot 5: Mapping Error
        fig_learning_axes[1, 1].plot(self.mapping_errors, '.-')

        # Plot 6: Coverage percentage
        fig_learning_axes[1, 2].plot(self.agent_coverage, '.-')

        # Add Episode number to top of image
        fig_learning.suptitle(
            "Episode number: " + str(episode_num) + ", Num timesteps: " + str(self.num_steps))

        # Save image to directory
        fig_file_name = "episode_" + str(episode_num) + ".png"
        plt.savefig(os.path.join(self.path_to_output_dir, fig_file_name))

        plt.close()

    def view_testing_episode_state(self, episode_num, timestep, save=False, path=None):

        if self.testing_field == None:
            cmap_color = "Blues"
        else:
            cmap_color = "Greens"

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
            self.env_curr_field.T, cmap=cmap_color)

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
            self.agent_curr_field.T, cmap=cmap_color)

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

    def view_field_state_only(self):
        if self.testing_field == None:
            cmap_color = "Blues"
        else:
            cmap_color = "Greens"

        fig = plt.figure(figsize=(8, 8))

        pos = plt.imshow(self.env_curr_field.T, cmap=cmap_color)
        cbar = plt.colorbar(pos, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=16)

        title_str = "Time step: " + \
            str(self.num_steps) + " Params (vx, vy): " + str((self.vx, self.vy))
        plt.title(title_str)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.show()

    def test_update_field_in_loop(self):
        fig = plt.figure(figsize=(8, 8))
        plt.ion()
        plt.show()
        self.agent_position = self.agent_start_position
        for i in range(100):
            plt.title("Step: " + str(i))

            plt.imshow(self.env_curr_field.T, cmap="Blues")
            traj_r = [position[0] for position in self.agent_trajectory]
            traj_c = [position[1] for position in self.agent_trajectory]
            plt.plot(traj_r, traj_c, '.', color='black')

            if (i % 2 == 0):
                (_, next_state) = self.get_next_position("right")
                self.agent_position = next_state
            else:
                (_, next_state) = self.get_next_position("down")
                self.agent_position = next_state

            self.agent_trajectory.append(next_state)

            plt.draw()
            plt.pause(0.001)
            self.update_env_field()
        plt.ioff()
        plt.close()

    def render(self, mode="human"):
        pass
