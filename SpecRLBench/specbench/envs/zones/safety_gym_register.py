import copy

from gymnasium import make as gymnasium_make
from gymnasium import register as gymnasium_register
from safety_gymnasium.utils.registration import make, register


def register_helper(env_config):
    """Register a environment to both Safety-Gymnasium and Gymnasium registry."""
    env_name, dash, version = env_config['env_id'].partition('-')
    # tmp_config = copy.deepcopy(env_config)
    # ap_config = tmp_config['ap_related_parameters']
    # tmp_config.pop('env_id')
    # tmp_config.pop('ap_related_parameters')
    config = {'config': env_config, 'task_id': env_config['env_id']}

    # print(f"register_helper, config = {config}")

    register(
        id=env_config['env_id'],
        entry_point='safety_gymnasium.builder:Builder',
        kwargs=config,
        max_episode_steps=env_config["max_episode_steps"],
    )
    gymnasium_register(
        id=f'{env_name}Gymnasium{dash}{version}',
        entry_point='safety_gymnasium.wrappers.gymnasium_conversion:make_gymnasium_environment',
        kwargs={'env_id': f'{env_name}Gymnasium{dash}{version}', **config},
        max_episode_steps=env_config["max_episode_steps"],
    )

