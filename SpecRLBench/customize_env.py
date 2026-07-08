import gymnasium as gym
import specbench
import safety_gymnasium

env_config = {
    'env_id': 'CustomizedLTL-v0',


    # the list of atomic propositions, choose from
    # [black, blue, green, cyan, red, magenta, yellow, white, orange, purple, lime, teal, pink, brown, navy, gray]
    'atomic_propositions': ['green', 'yellow', 'blue', 'magenta'], 

    # the number of zones per color, 
    # make sure to match the atomic propositions list, 
    'number_of_zones_per_color': {
        'green': 1,
        'yellow': 2,
        'blue': 3,
        'magenta': 4
    },

    # size of each zone (assume same size for all zones of the same color),
    'size_of_zones': {
        'green': 0.3,
        'yellow': 0.4,
        'blue': 0.1,
        'magenta': 0.2,
    },

    # number of moving zones per color
    'number_of_moving_zones': {
        'green': 1,
        'yellow': 1,
        'blue': 2,
        'magenta': 2,
    },

    # keepout distance for each zone, used when sampling zone placements,
    # the size of each zone and the keepout distance together determine if zones can overlap or not
    'keepout_distances': {
        'green': 0.3,
        'yellow': 0.1,
        'blue': 0.2,
        'magenta': 0.4,
    },

    # other config parameters

    # agent type, choose from 'Point', 'Car', and 'Ant'
    'agent_name': 'Point',

    # maximum number of steps per episode
    'max_episode_steps': 1000,

    # whether to use partial observability (for LiDAR, redundant when using vision)
    # if True, the agent will only observe the zones within a certain distance
    'partial_observability': False,

    # whether to flatten the observation into a 1D vector, 
    # make sure to keep false when using vision-based observation
    'observation_flatten': False,

    # whether to use vision-based observation (RGB images), if False, use LiDAR-based observation
    'use_vision': False,

}


from specbench.envs.zones.safety_gym_register import register_helper

# first register
register_helper(env_config=env_config)

# then make the environment and test it out
env = safety_gymnasium.make(env_config['env_id'], render_mode='human')

obs, info = env.reset(seed=0)

for _ in range(500):
    action = env.action_space.sample()
    env.step(action)
    env.render()

env.close()