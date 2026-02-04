from gymnasium.envs.registration import register

register(
    id = 'idrlenv/bipedal_agg-v0',
    entry_point = 'idrlenv.envs.bipedal_walker_agg:BipedalWalkerAgg',
    max_episode_steps=1500
)