import gymnasium as gym
from idrlenv.envs.centipede import centipede


def test_centipede_v0():
# Create the environment
    env = gym.make(id="centipede_v0", n_contents=4, render_mode='human')
    # n_contents 用于指定聚合体中单体的数量，>=1.
    
    observation, info = env.reset()
    count=0
    while count<1800:
        env.render()
        
        action = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished. Resetting environment.")
            observation, info = env.reset()
        count += 1

if __name__=="__main__":
    test_centipede_v0()