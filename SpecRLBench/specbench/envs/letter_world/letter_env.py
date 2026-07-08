import pickle
import random
from typing import Any, Literal

import numpy as np
import gymnasium as gym
import pygame
from gymnasium import spaces
from gymnasium.core import ObsType, ActType, RenderFrame
from gymnasium.wrappers import TimeLimit
import os

from specbench.utils.ltl.logic import FrozenAssignment, Assignment
from . letter_types import *


class LetterEnv(gym.Env):
    """
    This environment is a grid with randomly located letters on it.
    We ensure that there is a clean path to any of the letters (a path that includes no passing by any letter).
    Note that stepping outside the map causes the agent to appear on the other side.
    """
    metadata = {'render_modes': ['human', 'rgb_array', 'path'], 'render_fps': 10}

    def __init__(
            self,
            grid_size: int,
            letters: list[Letter],
            use_fixed_map: bool,
            use_agent_centric_view: bool,
            render_mode: str | None = None,
            map: dict[tuple[int, int], str] = None,
            show_window: bool = False,
            save_dir: str = "/projectnb/pnn/zjguo/code/mtrl/deep-ltl/letter_paths"
    ):
        if use_agent_centric_view and grid_size % 2 == 0:
            raise ValueError("Agent-centric view is only available for odd grid-sizes")
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.letters = letters
        self.use_fixed_map = use_fixed_map
        self.use_agent_centric_view = use_agent_centric_view
        self.letter_types = sorted(list(set([l.char for l in letters])))
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size, len(self.letter_types) + 1),
                                            dtype=np.uint8)
        self.map = map
        self.agent = (0, 0)
        self.locations = [(i, j) for i in range(grid_size) for j in range(grid_size) if (i, j) != (0, 0)]
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.original_obs = None

        self.rng = random.Random()
        if render_mode is not None:
            self.renderer = LetterEnvRenderer(self.grid_size, render_mode=render_mode,
                                             render_fps=self.metadata['render_fps'], show_window=show_window, save_dir=save_dir)

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        di, dj = self.actions[action]
        agent_i = (self.agent[0] + di + self.grid_size) % self.grid_size
        agent_j = (self.agent[1] + dj + self.grid_size) % self.grid_size
        self.agent = agent_i, agent_j

        # Letters step after agent
        # Process each letter: remove from old position, step, add to new position
        for letter in self.letters:
            prev_loc = (letter.x, letter.y)
            if prev_loc in self.map and self.map[prev_loc] == letter.char:
                del self.map[prev_loc]
            letter.step(self)
            new_loc = (letter.x, letter.y)
            self.map[new_loc] = letter.char

        if self.render_mode == "human":
            self._render_frame()
        self.original_obs = self._get_observation()
        return self.original_obs, 0.0, False, False, {'propositions': self.get_active_propositions()}

    def wait_for_input(self) -> bool:
        if self.render_mode not in ["human", "path"]:
            return False
        return self.renderer.wait_for_input()

    def _get_observation(self) -> ObsType:
        obs = np.zeros(shape=(self.grid_size, self.grid_size, len(self.letter_types) + 1), dtype=np.uint8)

        # Getting agent-centric view (if needed)
        c_map, agent = self.map, self.agent
        if self.use_agent_centric_view:
            c_map, agent = self._get_centric_map()

        # adding objects
        for loc in c_map:
            letter_id = self.letter_types.index(c_map[loc])
            obs[loc[0], loc[1], letter_id] = 1

        # adding agent
        obs[agent[0], agent[1], len(self.letter_types)] = 1
        return obs

    def _get_centric_map(self):
        center = self.grid_size // 2
        agent = (center, center)
        delta = center - self.agent[0], center - self.agent[1]
        c_map = {}
        for loc in self.map:
            new_loc_i = (loc[0] + delta[0] + self.grid_size) % self.grid_size
            new_loc_j = (loc[1] + delta[1] + self.grid_size) % self.grid_size
            c_map[(new_loc_i, new_loc_j)] = self.map[loc]
        return c_map, agent

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)
        if not self.use_fixed_map:
            self.map = None
        # Sampling a new map
        while self.map is None:
            # Sampling a random map
            self.map = {}
            self.rng.shuffle(self.locations)
            for i, letter in enumerate(self.letters):
                loc = self.locations[i]
                self.map[loc] = letter.char
                # Update Letter coordinates and reset their state
                letter.x = loc[0]
                letter.y = loc[1]
                letter.reset()
            # Checking that the map is valid
            if _is_valid_map(self.map, self.grid_size, self.actions):
                break
            self.map = None

        # # sample_agent
        # while True:
        #     self.agent = (self.rng.randint(0, self.grid_size - 1), self.rng.randint(0, self.grid_size - 1))
        #     # print(f"agent = {self.agent}")
        #     if self.agent not in self.map:
        #         break

        # Locating the agent into (0,0)
        self.agent = (0, 0)
        self.original_obs = self._get_observation()
        return self.original_obs, {'propositions': set()}

    def print(self):
        c_map, agent = self.map, self.agent
        if self.use_agent_centric_view:
            c_map, agent = self._get_centric_map()
        print("*" * (self.grid_size + 2))
        for i in range(self.grid_size):
            line = "*"
            for j in range(self.grid_size):
                if (i, j) == agent:
                    line += "A"
                elif (i, j) in c_map:
                    line += c_map[(i, j)]
                else:
                    line += " "
            print(line + "*")
        print("*" * (self.grid_size + 2))
        print("Active:", self.get_active_propositions())

    def print_features(self):
        obs = self._get_observation()
        print("*" * (self.grid_size + 2))
        for i in range(self.grid_size):
            line = "*"
            for j in range(self.grid_size):
                if np.max(obs[i, j, :]) > 0:
                    line += str(np.argmax(obs[i, j, :]))
                else:
                    line += " "
            print(line + "*")
        print("*" * (self.grid_size + 2))

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array" or self.renderer.show_window:
            return self._render_frame()

    def render_path(self, actions: list[int]):
        return self.renderer.render_path(self, [self.actions[a] for a in actions])

    def _render_frame(self):
        return self.renderer.render(self)

    def close(self):
        if self.render_mode is not None:
            self.renderer.close()

    def get_active_propositions(self) -> set[str]:
        if self.agent in self.map:
            letter = self.map[self.agent]
            return {letter}
        return set()

    def get_propositions(self) -> list[str]:
        return self.letter_types

    def get_possible_assignments(self) -> list[Assignment]:
        return Assignment.zero_or_one_propositions(set(self.get_propositions()))

    def save_world_info(self, path: str):
        with open(path, 'wb+') as f:
            pickle.dump(self.map, f)

    def load_world_info(self, path: str):
        with open(path, 'rb') as f:
            self.map = pickle.load(f)
        self.use_fixed_map = True


class LetterSafetyEnv(LetterEnv):
    """
    This environment is a grid with randomly located letters on it.
    We ensure that there is a clean path to any of the letters (a path that includes no passing by any letter).
    Note that stepping outside the map causes the agent to appear on the other side.
    """
    metadata = {'render_modes': ['human', 'rgb_array', 'path'], 'render_fps': 10}

    def __init__(
        self,
        grid_size: int,
        letters: str,
        use_fixed_map: bool,
        use_agent_centric_view: bool,
        render_mode: str | None = None,
        map: dict[tuple[int, int], str] = None,
        obs_grid_size: int = None,
        show_window: bool = False,
        save_dir: str = "/projectnb/pnn/zjguo/code/mtrl/deep-ltl/letter_paths"
    ):
        super().__init__(
            grid_size=grid_size,
            letters=letters,
            use_fixed_map=use_fixed_map,
            use_agent_centric_view=use_agent_centric_view,
            render_mode=render_mode,
            map=map,
            show_window=show_window,
            save_dir=save_dir
        )
        self.obs_grid_size = obs_grid_size if obs_grid_size is not None else grid_size
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.obs_grid_size, self.obs_grid_size, len(self.letter_types) + 1),
            dtype=np.uint8
        )

    def _get_observation(self) -> ObsType:
        obs = super()._get_observation()
        if self.obs_grid_size < self.grid_size:
            delta = (self.grid_size - self.obs_grid_size) // 2
            obs = obs[delta:delta+self.obs_grid_size, delta:delta+self.obs_grid_size, :]
        return obs

    def print_map(self):
        map = [[" ", " ", " ", " ", " ", " ", " "] for i in range(self.grid_size)]
        # print(map)
        for loc in self.map:
            map[loc[0]][loc[1]] = self.map[loc]
        map[self.agent[0]][self.agent[1]] = "A"
        for row in map:
            print(row)
        # print(self.map)


def _is_valid_map(map, grid_size, actions):
    open_list = [(0, 0)]
    closed_list = set()
    while open_list:
        s = open_list.pop()
        closed_list.add(s)
        if s not in map:
            for di, dj in actions:
                si = (s[0] + di + grid_size) % grid_size
                sj = (s[1] + dj + grid_size) % grid_size
                if (si, sj) not in closed_list and (si, sj) not in open_list:
                    open_list.append((si, sj))
    return len(closed_list) == grid_size * grid_size


class LetterEnvNoWrapping(gym.Env):
    """
    This environment is a grid with randomly located letters on it.
    We ensure that there is a clean path to any of the letters (a path that includes no passing by any letter).
    Note that stepping outside the map causes the agent to appear on the other side.
    """
    metadata = {'render_modes': ['human', 'rgb_array', 'path'], 'render_fps': 10}

    def __init__(
            self,
            grid_size: int,
            letters: str,
            use_fixed_map: bool,
            use_agent_centric_view: bool,
            render_mode: str | None = None,
            map: dict[tuple[int, int], str] = None
    ):
        if use_agent_centric_view and grid_size % 2 == 0:
            raise ValueError("Agent-centric view is only available for odd grid-sizes")
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.letters = letters
        self.use_fixed_map = use_fixed_map
        self.use_agent_centric_view = use_agent_centric_view
        self.letter_types = sorted(list(set(letters)))
        self.action_space = spaces.Discrete(4)
        # self.obs_grid_size = grid_size # 3 # grid_size # 5
        # self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_grid_size, self.obs_grid_size, len(self.letter_types) + 1),
        #                                     dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size, len(self.letter_types) + 1),
                                            dtype=np.uint8)
        self.map = map
        self.agent = (0, 0)
        self.locations = [(i, j) for i in range(grid_size) for j in range(grid_size) if (i, j) != (0, 0)]
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]#, (0, 0)]
        self.original_obs = None
        self.rng = random.Random()
        if render_mode is not None:
            self.renderer = LetterEnvRenderer(self.grid_size, render_mode=render_mode,
                                              render_fps=self.metadata['render_fps'])

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        di, dj = self.actions[action]
        # agent_i = (self.agent[0] + di + self.grid_size) % self.grid_size
        # agent_j = (self.agent[1] + dj + self.grid_size) % self.grid_size
        agent_i = self.agent[0] + di
        agent_j = self.agent[1] + dj
        agent_i = min(max(agent_i, 0), self.grid_size-1)
        agent_j = min(max(agent_j, 0), self.grid_size-1)

        self.agent = agent_i, agent_j

        if self.render_mode == "human":
            self._render_frame()
        self.original_obs = self._get_observation()
        return self.original_obs, 0.0, False, False, {'propositions': self.get_active_propositions()}

    def wait_for_input(self) -> bool:
        if self.render_mode not in ["human", "path"]:
            return False
        return self.renderer.wait_for_input()

    def _get_observation(self) -> ObsType:
        obs = np.zeros(shape=(self.grid_size, self.grid_size, len(self.letter_types) + 1), dtype=np.uint8)

        # Getting agent-centric view (if needed)
        c_map, agent = self.map, self.agent
        # if self.use_agent_centric_view:
        #     c_map, agent = self._get_centric_map()

        # adding objects
        for loc in c_map:
            letter_id = self.letter_types.index(c_map[loc])
            obs[loc[0], loc[1], letter_id] = 1

        # adding agent
        obs[agent[0], agent[1], len(self.letter_types)] = 1
        # delta = (self.grid_size - self.obs_grid_size) // 2
        # obs = obs[delta:delta+self.obs_grid_size, delta:delta+self.obs_grid_size, :]
        return obs

    # def _get_centric_map(self):
    #     center = self.grid_size // 2
    #     agent = (center, center)
    #     delta = center - self.agent[0], center - self.agent[1]
    #     c_map = {}
    #     for loc in self.map:
    #         new_loc_i = (loc[0] + delta[0] + self.grid_size) % self.grid_size
    #         new_loc_j = (loc[1] + delta[1] + self.grid_size) % self.grid_size
    #         c_map[(new_loc_i, new_loc_j)] = self.map[loc]
    #     return c_map, agent

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)
        if not self.use_fixed_map:
            self.map = None
        # Sampling a new map
        while self.map is None:
            # Sampling a random map
            self.map = {}
            self.rng.shuffle(self.locations)
            for i in range(len(self.letters)):
                self.map[self.locations[i]] = self.letters[i]
            # Checking that the map is valid
            if _is_valid_map_v2(self.map, self.grid_size, self.actions):
                break
            self.map = None

        # Locating the agent into (0,0)
        self.agent = (0, 0)
        self.original_obs = self._get_observation()
        return self.original_obs, {'propositions': set()}

    def print(self):
        c_map, agent = self.map, self.agent
        if self.use_agent_centric_view:
            c_map, agent = self._get_centric_map()
        print("*" * (self.grid_size + 2))
        for i in range(self.grid_size):
            line = "*"
            for j in range(self.grid_size):
                if (i, j) == agent:
                    line += "A"
                elif (i, j) in c_map:
                    line += c_map[(i, j)]
                else:
                    line += " "
            print(line + "*")
        print("*" * (self.grid_size + 2))
        print("Active:", self.get_active_propositions())

    def print_features(self):
        obs = self._get_observation()
        print("*" * (self.grid_size + 2))
        for i in range(self.grid_size):
            line = "*"
            for j in range(self.grid_size):
                if np.max(obs[i, j, :]) > 0:
                    line += str(np.argmax(obs[i, j, :]))
                else:
                    line += " "
            print(line + "*")
        print("*" * (self.grid_size + 2))

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array" or self.renderer.show_window:
            return self._render_frame()

    def render_path(self, actions: list[int]):
        return self.renderer.render_path(self, [self.actions[a] for a in actions])

    def _render_frame(self):
        return self.renderer.render(self)

    def close(self):
        if self.render_mode is not None:
            self.renderer.close()

    def get_active_propositions(self) -> set[str]:
        if self.agent in self.map:
            letter = self.map[self.agent]
            return {letter}
        return set()

    def get_propositions(self) -> list[str]:
        return self.letter_types

    # def get_possible_assignments(self) -> list[Assignment]:
    #     return Assignment.zero_or_one_propositions(set(self.get_propositions()))

    def save_world_info(self, path: str):
        with open(path, 'wb+') as f:
            pickle.dump(self.map, f)

    def load_world_info(self, path: str):
        with open(path, 'rb') as f:
            self.map = pickle.load(f)
        self.use_fixed_map = True


def _is_valid_map_v2(map, grid_size, actions):
    open_list = [(0, 0)]
    closed_list = set()
    while open_list:
        s = open_list.pop()
        closed_list.add(s)
        if s not in map:
            for di, dj in actions:
                # si = (s[0] + di + grid_size) % grid_size
                # sj = (s[1] + dj + grid_size) % grid_size
                si = s[0] + di
                sj = s[1] + dj
                si = min(max(si, 0), grid_size-1)
                sj = min(max(sj, 0), grid_size-1)
                if (si, sj) not in closed_list and (si, sj) not in open_list:
                    open_list.append((si, sj))
    return len(closed_list) == grid_size * grid_size


class LetterEnvRenderer:
    def __init__(self, grid_size: int, render_mode: Literal["human", "path", "rgb_array"] = "human",
                 render_fps=1, cell_size=120, save_dir="/projectnb/pnn/zjguo/code/mtrl/deep-ltl/letter_paths", show_window=False):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen_size = self.grid_size * self.cell_size
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.save_dir = save_dir  # Optional: where to save frames
        self.show_window = show_window

        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font(None, int(self.cell_size * 0.75))

        self.bg_color = (240, 240, 240)
        self.cell_color = (200, 200, 200)
        self.agent_color = (0, 128, 0)
        self.border_color = (0, 0, 0)
        self.arrow_color = (47, 79, 79, 160)
        # self.arrow_color = (0, 128, 0, 255)

        self.frame_count = 0
        os.makedirs(self.save_dir, exist_ok=True)

        self.window = None
        if self.show_window:
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('LetterEnvRenderer Frame')


    def render(self, env):
        canvas = self.draw_canvas(env)
        return self.update(canvas)

    def draw_canvas(self, env):
        canvas = pygame.Surface((self.screen_size, self.screen_size))
        canvas.fill(self.bg_color)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                if (i, j) == env.agent:
                    bg_color = self.agent_color
                    fg_color = (255, 255, 255)
                elif (i, j) in env.map:
                    bg_color = self.cell_color
                    fg_color = (0, 0, 0)
                else:
                    bg_color = self.bg_color
                pygame.draw.rect(canvas, bg_color, rect)
                if (i, j) in env.map:
                    text_surface = self.font.render(env.map[(i, j)], True, fg_color)
                    canvas.blit(text_surface, text_surface.get_rect(center=rect.center))
                pygame.draw.rect(canvas, self.border_color, rect, 2)
        return canvas

    def update(self, canvas):
        array = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), (1, 0, 2))

        if self.save_dir:
            filename = os.path.join(self.save_dir, f"frame_{self.frame_count:04d}.png")
            pygame.image.save(canvas, filename)
            self.frame_count += 1

        if self.show_window and self.window:
            self.window.blit(canvas, (0, 0))
            pygame.display.update()

        if self.render_mode == "rgb_array":
            return array
        else:
            return None

    def get_action_from_window(self, str_to_action):
        """
        Wait for a keypress in the window and return the corresponding action index.
        Returns None if quit or unknown key.
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    key = event.unicode.lower()
                    if key in str_to_action:
                        return str_to_action[key]
                    elif key == 'q':
                        return None

    def render_path(self, env, actions: list[tuple[int, int]]):
        canvas = self.draw_canvas(env)
        path_canvas = pygame.Surface((self.screen_size, self.screen_size), pygame.SRCALPHA)
        path_canvas.fill((0, 0, 0, 0))

        # current_pos = (0, 0)
        current_pos = (1, 3)
        for action in actions:
            next_pos = ((current_pos[0] + action[0]) % env.grid_size, (current_pos[1] + action[1]) % env.grid_size)
            self.draw_arrow(path_canvas, current_pos, next_pos)
            current_pos = next_pos

        canvas.blit(path_canvas, (0, 0))
        return self.update(canvas)

    def draw_arrow(self, canvas, start_pos, end_pos):
        sx = start_pos[1] * self.cell_size + self.cell_size // 2
        sy = start_pos[0] * self.cell_size + self.cell_size // 2
        ex = end_pos[1] * self.cell_size + self.cell_size // 2
        ey = end_pos[0] * self.cell_size + self.cell_size // 2

        def draw_segment(sx, sy, ex, ey):
            pygame.draw.line(canvas, self.arrow_color, (sx, sy), (ex, ey), 15)
            angle = np.arctan2(ey - sy, ex - sx)
            ah = 10
            ah_angle = np.pi / 6
            head = [
                (ex, ey),
                (ex - ah * np.cos(angle - ah_angle), ey - ah * np.sin(angle - ah_angle)),
                (ex - ah * np.cos(angle + ah_angle), ey - ah * np.sin(angle + ah_angle))
            ]
            pygame.draw.polygon(canvas, self.arrow_color, head, width=15)

        if abs(start_pos[1] - end_pos[1]) > 1:
            if start_pos[1] > end_pos[1]:
                draw_segment(sx, sy, self.screen_size, sy)
                draw_segment(0, ey, ex, ey)
            else:
                draw_segment(sx, sy, 0, sy)
                draw_segment(self.screen_size, ey, ex, ey)
        elif abs(start_pos[0] - end_pos[0]) > 1:
            if start_pos[0] > end_pos[0]:
                draw_segment(sx, sy, sx, self.screen_size)
                draw_segment(ex, 0, ex, ey)
            else:
                draw_segment(sx, sy, sx, 0)
                draw_segment(ex, self.screen_size, ex, ey)
        else:
            draw_segment(sx, sy, ex, ey)

    def wait_for_input(self) -> bool:
        return False  # no actual input handling in headless mode

    def close(self):
        pygame.quit()


def main():
    # commands
    str_to_action = {"w": 0, "s": 1, "a": 2, "d": 3}
    grid_size = 7
    letter_chars = "aabbccddee"
    use_fixed_map = False
    use_agent_centric_view = True
    timeout = 10

    # Create Letter objects
    letters = [Letter(char=c) for c in letter_chars]
    # Add a moving letter for testing
    letters.append(MovingLetter(char='z', dx=1, dy=0))
    # Add a bouncing letter for testing
    letters.append(BouncingLetter(char='x', dx=0, dy=1))
    # Add a random moving letter for testing
    letters.append(RandomMovingLetter(char='y'))

    # play the game!
    game = LetterSafetyEnv(grid_size, letters, use_fixed_map, use_agent_centric_view, render_mode="human", show_window=True, save_dir="./letter_paths")
    game = TimeLimit(game, timeout)
    while True:
        # Episode
        game.reset()
        while True:
            game.render()
            if hasattr(game, 'renderer') and getattr(game.renderer, 'show_window', False):
                action = game.renderer.get_action_from_window(str_to_action)
                if action is None:
                    print("Quit or unknown key pressed.")
                    return
            else:
                print("\nAction? ", end="")
                a = input()
                print()
                action = str_to_action[a] if a in str_to_action else None
                if action is None:
                    print("Forbidden action")
                    continue
            obs, reward, term, trunc, _ = game.step(action)
            if term or trunc:
                break
        print("Episode ended.")
        game.render()

if __name__ == "__main__":
    main()
