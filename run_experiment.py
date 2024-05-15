import pickle
from lib.diffusion_field_env import SpatialTemporalDiffusionEnvironment, SpatialTemporalDiffusionField, SpatialTemporalDiffusionEnvGymWrapper 
from lib.agent_handlers import ClusterBasedExplorationHandler, GradientSourceHeadingHandler
from lib.utils import plot_experiment

selected_maps = ['2-src-6','3-src-4']
envs = []
for name in selected_maps:
    with open(f'./data/envs/{name}.dat', 'rb') as handle:
        envs.append(pickle.load(handle))

if __name__ == "__main__":
    env_wrapper = SpatialTemporalDiffusionEnvGymWrapper(envs=envs)
    INITIAL_POS = [90,90]

    state, obsr  = env_wrapper.reset(pos=INITIAL_POS, env_id=0)
    steps = 0
    z_thresh = 2
    exploration_handler = ClusterBasedExplorationHandler(pos=INITIAL_POS)
    src_heading_handler = GradientSourceHeadingHandler()
    explore_mode = True
    new_target_reached = True
    traj = []
    src_found = 0
    while True:
        if explore_mode:
            action_id, dest_reached = exploration_handler.get_next_action(state, obsr)
            new_target_reached = new_target_reached or dest_reached 
            state, reward, done, obsr = env_wrapper.step(action_id)
            if state[0] > z_thresh and new_target_reached:
                explore_mode = False
        else:
            action_id = src_heading_handler.get_next_action(state, obsr)    
            state, reward, done, obsr = env_wrapper.step(action_id)
            if "source_found" in obsr.keys() and src_found<2:
                src_found += 1
                explore_mode = True
                new_target_reached = False
                exploration_handler.reset(field_mask=obsr["source_mask"])
        traj.append(obsr["location"])
        if done:
            plot_experiment(env_wrapper.env.env_field, env_wrapper.agent_field.vs_field, traj, env_wrapper.get_metrics(), save=True)
            # plt.imshow(env.agent_field.visited_field)
            break
        steps += 1 
