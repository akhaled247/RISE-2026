from stable_baselines3 import PPO
import gymnasium as gym
from numpy import uint8
import specbench
import safety_gymnasium
from gymnasium.wrappers import FlattenObservation

def make_env(env_name, render_mode=None):
    if env_name.startswith("Letter"):
        env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    elif env_name.startswith("Panda"):
        env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
    elif env_name.startswith("Point") or env_name.startswith("Car") or env_name.startswith("Ant"):
        from specbench.envs.zones.safety_gym_wrapper_ma import SafetyGymWrapperMA
        from specbench.envs.zones.safety_gym_wrapper_ma_sro import SafetyGymWrapperMASAR
        from specbench.envs.zones.safety_gym_wrapper import SafetyGymWrapper
        import safety_gymnasium
        env = safety_gymnasium.make(env_name, disable_env_checker=True, render_mode=render_mode)
        env = SafetyGymWrapperMASAR(env) if "SAR" in env_name else SafetyGymWrapperMA(env) if "MA" in env_name else SafetyGymWrapper(env)
    else:
        # env = gym.make(env_name, disable_env_checker=True, render_mode=render_mode)
        try:
            import safety_gymnasium
            env = safety_gymnasium.make(env_name, disable_env_checker=True, render_mode=render_mode)
        except Exception as e:
            raise ValueError(f"Unknown environment name: {env_name}")
    return env

seed = 0

# 1. Initialize the standard Gymnasium environment
env_name = 'PointLTLMASAR2-v0'
steps = 750

print(f"="*40)
render_mode = "human" if 'Vision' not in env_name else None
env = make_env(env_name, render_mode=None)

# 2. Instantiate the PPO Agent 
# "MlpPolicy" is used for feature vectors (like positions and velocities)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    device="cpu"
)
# 3. Train the agent
model.learn(total_timesteps=50000)

# 4. Evaluate the trained agent
env = make_env(env_name, render_mode="human")
obs, info = env.reset(see=seed)
total_reward = 0
episodes = 1
for episode in range(episodes):
    obs, info = env.reset()
    episode_reward = 0
    done = False
    i = 0
    while not done or i>steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        done = terminated or truncated
        i+=1

    print(f"Episode {episode+1}: {episode_reward}")

env.close()
