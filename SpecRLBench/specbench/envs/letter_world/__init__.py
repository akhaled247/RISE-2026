from gymnasium.envs.registration import register
from specbench.envs.letter_world.letter_types import Letter


register(
    id='LetterLTL0-v0',
    entry_point='specbench.envs.letter_world.letter_env:LetterSafetyEnv',
    kwargs=dict(
        grid_size=7,
        letters=[Letter(c) for c in "aabbccddeeffgghhiijjkkll"],
        use_fixed_map=False,
        use_agent_centric_view=True,
    )
)


register(
    id='LetterLTL0-v0.partial',
    entry_point='specbench.envs.letter_world.letter_env:LetterSafetyEnv',
    kwargs=dict(
        grid_size=15,
        letters=[Letter(c) for c in "aabbccddeeffgghhiijjkkll"],
        use_fixed_map=False,
        use_agent_centric_view=True,
        obs_grid_size=7,
    )
)
