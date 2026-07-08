import os

from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

ENV_IDS = []

for task in ["Reach", "LTLReach",
             "PickAndPlace", "LTLPickAndPlace",
             "Slide", "Push", "Stack", "Flip"]:
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""

            env_id_l0 = f"Panda{task}0{control_suffix}{reward_suffix}-v0"
            env_id_l0_partial = env_id_l0 + '.partial'
            env_id_l0_vision = f"Panda{task}0{control_suffix}{reward_suffix}Vision-v0"

            env_id_l1 = f"Panda{task}1{control_suffix}{reward_suffix}-v0"
            env_id_l1_partial = env_id_l1 + '.partial'
            env_id_l1_vision = f"Panda{task}1{control_suffix}{reward_suffix}Vision-v0"

            max_episode_steps = 100 if task == "Stack" else 50
            max_episode_steps = 200 if "LTL" in task else max_episode_steps
            # default closest point observation
            register(
                id=env_id_l0,
                entry_point=f"panda_gym.envs:Panda{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type, 
                        "obs_use_ee_only": True},
                max_episode_steps=max_episode_steps,
            )
            ENV_IDS.append(env_id_l0)

            # closest point observation with partial observability
            register(
                id=env_id_l0_partial,
                entry_point=f"panda_gym.envs:Panda{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type, 
                        "partial_observability": True, "obs_use_ee_only": True},
                max_episode_steps=max_episode_steps,
            )
            ENV_IDS.append(env_id_l0_partial)

            # closest point observation with both ee and arm states
            register(
                id=env_id_l1,
                entry_point=f"panda_gym.envs:Panda{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type, 
                        "obs_use_ee_only": False},
                max_episode_steps=max_episode_steps,
            )
            ENV_IDS.append(env_id_l1)

            # closest point observation with partial observability and both ee and arm states
            register(
                id=env_id_l1_partial,
                entry_point=f"panda_gym.envs:Panda{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type, 
                        "partial_observability": True, "obs_use_ee_only": False},
                max_episode_steps=max_episode_steps,
            )
            ENV_IDS.append(env_id_l1_partial)

            # rgbd camera observation
            register(
                id=env_id_l0_vision,
                entry_point=f"panda_gym.envs:Panda{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type, 
                        "observe_vision": True, "obs_use_ee_only": True},
                max_episode_steps=max_episode_steps,
            )
            ENV_IDS.append(env_id_l0_vision)

            # rgbd camera observation with both ee and arm states
            register(
                id=env_id_l1_vision,
                entry_point=f"panda_gym.envs:Panda{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type, 
                        "observe_vision": True, "obs_use_ee_only": False},
                max_episode_steps=max_episode_steps,
            )
            ENV_IDS.append(env_id_l1_vision)

