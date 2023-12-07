from gym.envs.registration import register

register(
    id='field-no-loc-state-with-grad-reward-9-actions-v0',
    entry_point='gym_field.envs:SpatialTemporalFieldNoLocStateWithGradRewardConcNineActions'
)
register(
    id='field-9-actions-v0',
    entry_point='gym_field.envs:SpatialTemporalFieldNineActions'
)
register(
    id='field-9-actions-image-v0',
    entry_point='gym_field.envs:SpatialTemporalFieldImageStateNineActions'
)
