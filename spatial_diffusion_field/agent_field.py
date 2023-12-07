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

class AgentField():
    def __init__(
            self,
            ego_size=15,
            source_mask_radius=5,
            viewscope_half_size=5,
            params_dict=DEFAULT_FIELD_PARAMS,
            ):
        self.ego_size = ego_size
        self.mask_radius = source_mask_radius
        self.vs_hsize = viewscope_half_size
        self.load_params(params_dict)
        self.vs_field_improvement = 0
        self.vs_field_raw = np.zeros((self.size,self.size))
        self.vs_field = np.zeros((self.size,self.size))
        self.visited_field = np.zeros((self.size,self.size))
        self.viewscope = np.zeros([2 * self.vs_hsize + 1, 2 * self.vs_hsize + 1])
        self.curr_pos = np.array([0,0])
        
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
        self.vs_field_improvement = np.abs(np.sum(env_field[min_y:max_y, min_x:max_x]) - np.sum(self.vs_field_raw[min_y:max_y, min_x:max_x]))
        self.vs_field_raw[min_y:max_y, min_x:max_x] = env_field[min_y:max_y, min_x:max_x]        
        self.vs_field[min_y:max_y, min_x:max_x] = env_field[min_y:max_y, min_x:max_x]        
        self.curr_pos = pos

        # In training, we can comment this to speed up training
        # self.viewscope = self.vs_field[min_y:max_y, min_x:max_x]
        # self.update_viewscope_field()
        
        self.visited_field[min_y:max_y, min_x:max_x] = np.minimum(VISITED_MASK, self.visited_field[min_y:max_y, min_x:max_x])

    def get_viewscope_image(self, show_agent=True):
        img = (self.vs_field_raw/25*210+30).astype('uint8')
        img[self.visited_field!=VISITED_MASK]=0
        img[self.curr_pos[1]-1:self.curr_pos[1]+2,self.curr_pos[0]-1:self.curr_pos[0]+2]=255
        if (show_agent):
            img[self.curr_pos[1]-self.vs_hsize:self.curr_pos[1]+self.vs_hsize+1,self.curr_pos[0]-self.vs_hsize]=255
            img[self.curr_pos[1]-self.vs_hsize:self.curr_pos[1]+self.vs_hsize+1,self.curr_pos[0]+self.vs_hsize]=255
            img[self.curr_pos[1]-self.vs_hsize,self.curr_pos[0]-self.vs_hsize:self.curr_pos[0]+self.vs_hsize+1]=255
            img[self.curr_pos[1]+self.vs_hsize,self.curr_pos[0]-self.vs_hsize:self.curr_pos[0]+self.vs_hsize+1]=255
        return img[..., np.newaxis]

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
        self.vs_field_raw = np.zeros((self.size,self.size))
        self.vs_field_improvement = 0
        self.visited_field = np.zeros((self.size,self.size))
        self.viewscope = np.zeros([2 * self.vs_hsize + 1, 2 * self.vs_hsize + 1])
