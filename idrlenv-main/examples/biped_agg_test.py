import gymnasium as gym
from idrlenv.envs.bipedal_walker_agg import BipedalWalkerAgg
# Create the environment

def test_bipedal_agg_v0():
    # env = gym.make(id='idrlenv/bipedalAgg-v0', n_contents=2, render_mode='human')
    env = gym.make(id="idrlenv/bipedal_agg-v0", n_contents=4, render_mode='human')
    observation, info = env.reset()
    count=0

    while count<1800:
        # Render the current state of the environment (optional)
        env.render()
        
        # Take a random action
        action = env.action_space.sample()
        
        # Step the environment forward by one timestep
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Check if the episode has ended
        if terminated or truncated:
            print("Episode finished. Resetting environment.")
            observation, info = env.reset()
        count+=1
        
if __name__ == "__main__":
    test_bipedal_agg_v0()
