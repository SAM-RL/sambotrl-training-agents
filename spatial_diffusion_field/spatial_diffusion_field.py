import random
DEFAULT_FIELD_PARAMS = {
    'size': 100,
    'dx': 0.8,
    'dy': 0.8,
    'dt': 0.15,
    'vx': 0.0,
    'vy': 0.0,
    'k': 1,
}
class SpatialDiffusionField():
    def __init__(
            self,
            field_name=None,
            fields_data={},
            preload=True,
            randomize=False
            ):
        self.n_step = 0
        self.preload = preload
        self.randomize = randomize
        self.fields_data = fields_data
        self.field_name = field_name
        if field_name is not None:
            self.field = fields_data[field_name]['snapshots'][0]
            self.load_params(fields_data[field_name]['metadata'])
            self.sources = fields_data[self.field_name]['sources']
        else:
            self.field_name = next(iter(fields_data))
            self.field = fields_data[self.field_name]['snapshots'][0]
            self.load_params(fields_data[self.field_name]['metadata'])
            self.sources = fields_data[self.field_name]['sources']

    def load_params(self, params_dict):
        self.params = params_dict
        self.size = params_dict["size"]
        self.dx = params_dict["dx"]
        self.dy = params_dict["dy"]
        self.vx = params_dict["vx"]
        self.vy = params_dict["vy"]
        self.dt = params_dict["dt"]
        self.k = params_dict["k"]

    def step(self):
        self.n_step += 1
        if self.preload:
            self.field = self.fields_data[self.field_name]['snapshots'][self.n_step]
        else: 
            self.field = self.update_field()       

    def update_field(self):
        updated_u = self.field.copy()
        u_k = self.field.copy()
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                updated_u[j, i] = u_k[j, i] + self.k * (self.dt / self.dx ** 2) * \
                    ((u_k[j + 1, i] + u_k[j - 1, i] +
                        u_k[j, i + 1] + u_k[j, i - 1] - 4 * u_k[j, i])) + \
                    self.vx * (self.dt / self.dx) * ((u_k[j + 1, i] - u_k[j, i])) + self.vy * (self.dt / self.dy) * \
                    (u_k[j, i + 1] - u_k[j, i])
        return updated_u

    def reset(self):
        if self.randomize:
            self.field_name = random.choice(list(self.fields_data.keys()))
            self.field = self.fields_data[self.field_name]['snapshots'][0]
            self.load_params(self.fields_data[self.field_name]['metadata'])
            self.sources = self.fields_data[self.field_name]['sources']
        else:
            self.field = self.fields_data[self.field_name]['snapshots'][0]
            self.sources = self.fields_data[self.field_name]['sources']
        self.n_step = 0