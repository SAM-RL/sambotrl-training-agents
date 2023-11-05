import numpy as np
DEFAULT_FIELD_PARAMS = {
    'size': 100,
    'dx': 0.8,
    'dy': 0.8,
    'dt': 0.15,
    'vx': 0.0,
    'vy': 0.0,
    'k': 1,
}

VISITED_MASK = -1
DETECTED_SOURCE_MASK = -5
BORDER_MASK = -3

class AgentField():
    def __init__(
            self,
            ego_size=15,
            fov_masks=None,
            source_mask_radius=5,
            viewscope_half_size=5,
            params_dict=DEFAULT_FIELD_PARAMS,
            ):
        self.ego_size = ego_size
        self.mask_radius = source_mask_radius
        self.vs_hsize = viewscope_half_size
        self.load_params(params_dict)
        self.vs_field = np.zeros((self.size,self.size))
        self.visited_field = np.zeros((self.size,self.size))
        self.viewscope = np.zeros([2 * self.vs_hsize + 1, 2 * self.vs_hsize + 1])
        self.fov_masks = fov_masks
        self.fov_masks_sum = np.sum(fov_masks,axis=(1,2))
        
    def load_params(self, params_dict):
        self.size = params_dict["size"]
        self.dx = params_dict["dx"]
        self.dy = params_dict["dy"]
        self.vx = params_dict["vx"]
        self.vy = params_dict["vy"]
        self.dt = params_dict["dt"]
        self.k = params_dict["k"]

    def update(self, env_field, pos):
        min_x, max_x = pos[0] - self.vs_hsize, pos[0] + self.vs_hsize + 1
        min_y, max_y = pos[1] - self.vs_hsize, pos[1] + self.vs_hsize + 1
        self.vs_field[min_x:max_x, min_y:max_y] = env_field[min_x:max_x, min_y:max_y]
        self.viewscope = self.vs_field[min_x:max_x, min_y:max_y]
        self.update_viewscope_field()
        self.visited_field[pos[0], pos[1]] = np.minimum(VISITED_MASK, self.visited_field[pos[0], pos[1]])
        
    def mask_detected_src(self, src_pos):
        min_x, max_x = src_pos[0] - self.mask_radius, src_pos[0] + self.mask_radius + 1
        min_y, max_y = src_pos[1] - self.mask_radius, src_pos[1] + self.mask_radius + 1
        self.visited_field[min_x:max_x, min_y:max_y] = np.minimum(DETECTED_SOURCE_MASK, self.visited_field[min_x:max_x, min_y:max_y])

    def compute_fov_vector(self, pos):
        field_hsize = self.size // 2
        padded_field = np.pad(self.visited_field, (field_hsize, field_hsize), 'constant', constant_values=BORDER_MASK)
        ego_field = padded_field[pos[0]+field_hsize-(self.ego_size//2):pos[0]+field_hsize+(self.ego_size//2+self.ego_size%2), \
                                        pos[1]+field_hsize-(self.ego_size//2):pos[1]+field_hsize+(self.ego_size//2+self.ego_size%2)]
        fov_vector = np.sum(ego_field*self.fov_masks, axis=(1,2)) / self.fov_masks_sum
        return fov_vector, ego_field

    def update_viewscope_field(self):
        updated_u = self.vs_field.copy()
        u_k = self.vs_field.copy()
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                updated_u[j, i] = u_k[j, i] + self.k * (self.dt / self.dx ** 2) * \
                    ((u_k[j + 1, i] + u_k[j - 1, i] +
                        u_k[j, i + 1] + u_k[j, i - 1] - 4 * u_k[j, i])) + \
                    self.vx * (self.dt / self.dx) * ((u_k[j + 1, i] - u_k[j, i])) + self.vy * (self.dt / self.dy) * \
                    (u_k[j, i + 1] - u_k[j, i])
        self.vs_field = updated_u
    

    def reset(self):
        self.vs_field = np.zeros((self.size,self.size))
        self.visited_field = np.zeros((self.size,self.size))
        self.viewscope = np.zeros([2 * self.vs_hsize + 1, 2 * self.vs_hsize + 1])
