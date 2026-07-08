from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple
from xml.sax.saxutils import escape
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


DEFAULT_CONFIG: dict[str, Any] = {
    "agent_num": 2,
    "zone_colors": ["green", "yellow", "blue", "magenta"],
    "zone_size": [0.5, 0.5, 0.5, 0.5],
    "zone_num": [2, 2, 2, 2],
    "zone_keepout": 0.4,
    "agent_radius": 0.28,
    "agent_keepout": 0.5,
    "placement_extents": [-2.8, -2.8, 2.8, 2.8],
    "max_layout_retries": 10_000,
    "max_place_retries": 1_000,
    "lidar_bin_dim": 16,
    "lidar_exp_gain": 0.5,
    "lidar_alias": True,
    "max_steps": 1000,
    "frameskip": 10,
    "fixed_layout_seed": None,
}

COLOR_IDS = {
    "green": 0,
    "yellow": 1,
    "blue": 2,
    "magenta": 3,
    "black": 4,
    "cyan": 5,
    "red": 6,
    "white": 7,
    "orange": 8,
    "purple": 9,
    "lime": 10,
    "teal": 11,
    "pink": 12,
    "brown": 13,
    "navy": 14,
    "gray": 15,
}

COLOR_RGBA = {
    "green": (0.0, 1.0, 0.0, 0.25),
    "yellow": (1.0, 1.0, 0.0, 0.25),
    "blue": (0.0, 0.0, 1.0, 0.25),
    "magenta": (1.0, 0.0, 1.0, 0.25),
    "black": (0.0, 0.0, 0.0, 0.25),
    "cyan": (0.0, 1.0, 1.0, 0.25),
    "red": (1.0, 0.0, 0.0, 0.25),
    "white": (1.0, 1.0, 1.0, 0.25),
    "orange": (1.0, 0.5, 0.0, 0.25),
    "purple": (0.5, 0.0, 0.5, 0.25),
    "lime": (0.5, 1.0, 0.0, 0.25),
    "teal": (0.0, 0.5, 0.5, 0.25),
    "pink": (1.0, 0.75, 0.8, 0.25),
    "brown": (0.6, 0.3, 0.0, 0.25),
    "navy": (0.0, 0.0, 0.5, 0.25),
    "gray": (0.5, 0.5, 0.5, 0.25),
}

COLOR_NAMES_BY_ID = {idx: color for color, idx in COLOR_IDS.items()}


class Box(NamedTuple):
    """Tiny JAX-friendly Box space descriptor.

    This keeps the environment independent from ``dfa_gym.spaces`` while still
    exposing the same useful ``low/high/shape/dtype`` attributes.
    """

    low: float
    high: float
    shape: tuple[int, ...]
    dtype: Any = jnp.float32

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.uniform(
            key,
            shape=self.shape,
            minval=self.low,
            maxval=self.high,
            dtype=self.dtype,
        )


@dataclass(frozen=True)
class LayoutSpec:
    """Static-shape layout sampling inputs.

    All counts are Python integers/tuples so callers can JIT functions that
    close over this spec.  Numeric values are JAX arrays so the sampler itself
    stays on the JAX side.
    """

    agent_num: int
    agent_keepout: float
    zone_colors: tuple[str, ...]
    zone_color_ids: tuple[int, ...]
    zone_size: tuple[float, ...]
    zone_num: tuple[int, ...]
    zone_keepout: float
    placement_extents: tuple[float, float, float, float]
    max_layout_retries: int
    max_place_retries: int

    @property
    def zone_total(self) -> int:
        return sum(self.zone_num)

    @property
    def object_total(self) -> int:
        return self.zone_total + self.agent_num


class Layout(NamedTuple):
    """A sampled environment layout.

    ``object_xy`` and ``object_keepout`` follow the original sampling order:
    all zone objects first, then all agents.  Convenience views for zones and
    agents are included because XML generation will usually want them directly.
    """

    valid: jax.Array
    object_xy: jax.Array
    object_keepout: jax.Array
    zone_xy: jax.Array
    zone_size: jax.Array
    zone_color_id: jax.Array
    agent_xy: jax.Array
    agent_rot: jax.Array


class SafeMultiAgentMJXEnvState(NamedTuple):
    """JAX state for the fixed-topology MJX environment."""

    model: Any
    data: Any
    layout: Layout
    time: int


def make_layout_spec(config: dict[str, Any] | None = None) -> LayoutSpec:
    """Normalize a user config into a static-shape ``LayoutSpec``.

    This helper is Python-side by design.  It validates flexible user input
    once, then the actual sampling happens in ``sample_layout`` with JAX only.
    """

    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)

    zone_colors = tuple(merged["zone_colors"])
    zone_num = tuple(int(x) for x in merged["zone_num"])
    zone_size = tuple(float(x) for x in merged["zone_size"])

    if len(zone_colors) != len(zone_num):
        raise ValueError("zone_colors and zone_num must have the same length.")
    if len(zone_colors) != len(zone_size):
        raise ValueError("zone_colors and zone_size must have the same length.")
    unknown = [color for color in zone_colors if color not in COLOR_IDS]
    if unknown:
        raise ValueError(f"Unknown zone colors: {unknown}")
    if int(merged["agent_num"]) <= 0:
        raise ValueError("agent_num must be positive.")

    return LayoutSpec(
        agent_num=int(merged["agent_num"]),
        agent_keepout=float(merged["agent_keepout"]),
        zone_colors=zone_colors,
        zone_color_ids=tuple(COLOR_IDS[color] for color in zone_colors),
        zone_size=zone_size,
        zone_num=zone_num,
        zone_keepout=float(merged["zone_keepout"]),
        placement_extents=tuple(float(x) for x in merged["placement_extents"]),
        max_layout_retries=int(merged["max_layout_retries"]),
        max_place_retries=int(merged["max_place_retries"]),
    )


def _zone_arrays(spec: LayoutSpec) -> tuple[jax.Array, jax.Array, jax.Array]:
    color_ids: list[int] = []
    sizes: list[float] = []
    keepouts: list[float] = []

    for color_id, size, num in zip(spec.zone_color_ids, spec.zone_size, spec.zone_num):
        color_ids.extend([color_id] * num)
        sizes.extend([size] * num)
        keepouts.extend([size + 0.5 * spec.zone_keepout] * num)

    return (
        jnp.asarray(color_ids, dtype=jnp.int32),
        jnp.asarray(sizes, dtype=jnp.float32),
        jnp.asarray(keepouts, dtype=jnp.float32),
    )


def _sample_xy(key: jax.Array, extents: jax.Array, keepout: jax.Array) -> jax.Array:
    low = extents[:2] + keepout
    high = extents[2:] - keepout
    return jax.random.uniform(key, shape=(2,), minval=low, maxval=high)


def _valid_against_placed(
    xy: jax.Array,
    keepout: jax.Array,
    placed_xy: jax.Array,
    placed_keepout: jax.Array,
    placed_mask: jax.Array,
) -> jax.Array:
    distances = jnp.linalg.norm(placed_xy - xy, axis=-1)
    required = placed_keepout + keepout
    ok_or_unplaced = jnp.logical_or(~placed_mask, distances >= required)
    return jnp.all(ok_or_unplaced)


def _place_one_object(
    key: jax.Array,
    extents: jax.Array,
    keepout: jax.Array,
    placed_xy: jax.Array,
    placed_keepout: jax.Array,
    placed_mask: jax.Array,
    max_place_retries: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rejection-sample one object with JAX control flow."""

    def cond_fun(state):
        attempt, _key, _xy, valid = state
        return jnp.logical_and(attempt < max_place_retries, ~valid)

    def body_fun(state):
        attempt, key_i, _xy, _valid = state
        key_i, subkey = jax.random.split(key_i)
        xy = _sample_xy(subkey, extents, keepout)
        valid = _valid_against_placed(xy, keepout, placed_xy, placed_keepout, placed_mask)
        return attempt + 1, key_i, xy, valid

    init_xy = jnp.zeros((2,), dtype=jnp.float32)
    init = (jnp.asarray(0), key, init_xy, jnp.asarray(False))
    _, out_key, xy, valid = jax.lax.while_loop(cond_fun, body_fun, init)
    return out_key, xy, valid


def _sample_layout_once(key: jax.Array, spec: LayoutSpec) -> Layout:
    key, rot_key = jax.random.split(key)
    zone_color_id, zone_size, zone_keepout = _zone_arrays(spec)
    placement_extents = jnp.asarray(spec.placement_extents, dtype=jnp.float32)

    object_keepout = jnp.concatenate(
        [
            zone_keepout,
            jnp.full((spec.agent_num,), spec.agent_keepout, dtype=jnp.float32),
        ],
    )
    object_xy = jnp.zeros((spec.object_total, 2), dtype=jnp.float32)
    placed_mask = jnp.zeros((spec.object_total,), dtype=jnp.bool_)

    valid = jnp.asarray(True)
    key_i = key
    for idx in range(spec.object_total):
        key_i, subkey = jax.random.split(key_i)
        key_i, xy, placed = _place_one_object(
            subkey,
            placement_extents,
            object_keepout[idx],
            object_xy,
            object_keepout,
            placed_mask,
            spec.max_place_retries,
        )
        object_xy = object_xy.at[idx].set(xy)
        placed_mask = placed_mask.at[idx].set(placed)
        valid = jnp.logical_and(valid, placed)

    agent_rot = jax.random.uniform(
        rot_key,
        shape=(spec.agent_num,),
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )

    return Layout(
        valid=valid,
        object_xy=object_xy,
        object_keepout=object_keepout,
        zone_xy=object_xy[: spec.zone_total],
        zone_size=zone_size,
        zone_color_id=zone_color_id,
        agent_xy=object_xy[spec.zone_total :],
        agent_rot=agent_rot,
    )


def sample_layout(key: jax.Array, spec: LayoutSpec) -> Layout:
    """Sample a valid layout using only JAX randomness/control flow.

    This mirrors the original Safety Gymnasium order: zones are placed first,
    then agents are placed while respecting keepout against zones and other
    agents.
    """

    def cond_fun(state):
        attempt, _key, layout = state
        return jnp.logical_and(attempt < spec.max_layout_retries, ~layout.valid)

    def body_fun(state):
        attempt, key_i, _layout = state
        key_i, subkey = jax.random.split(key_i)
        layout = _sample_layout_once(subkey, spec)
        return attempt + 1, key_i, layout

    key, subkey = jax.random.split(key)
    init_layout = _sample_layout_once(subkey, spec)
    init = (jnp.asarray(1), key, init_layout)
    _, _, layout = jax.lax.while_loop(cond_fun, body_fun, init)
    return layout


# def make_sample_layout_fn(spec: LayoutSpec):
#     """Return a JIT-compiled single-layout sampler for a static spec."""

#     return jax.jit(partial(sample_layout, spec=spec))


# def make_batched_sample_layout_fn(spec: LayoutSpec):
#     """Return a JIT-compiled batched sampler over a batch of PRNG keys."""

#     return jax.jit(jax.vmap(partial(sample_layout, spec=spec)))


def _fmt(values) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def rot2quat(theta: jax.Array) -> jax.Array:
    """Return a quaternion for a rotation about the Z axis."""

    half = theta / 2.0
    zeros = jnp.zeros_like(half)
    return jnp.stack([jnp.cos(half), zeros, zeros, jnp.sin(half)], axis=-1)


def _build_fixed_topology_point_xml(config: dict[str, Any], spec: LayoutSpec) -> str:
    """Build one fixed-topology MuJoCo XML string.

    Runtime reset samples a layout and writes agent positions into ``qpos``.
    Zone positions are kept in JAX layout state for now, so their XML bodies are
    only topology/render placeholders.
    """

    # if str(config["agent_name"]) != "Point":
    #     raise NotImplementedError("Only agent_name='Point' is implemented so far.")

    xml: list[str] = []
    xml.append("<mujoco model=\"safe_multi_agent_mjx\">")
    xml.append("  <size njmax=\"3000\" nconmax=\"1000\"/>")
    xml.append("  <option timestep=\"0.002\"/>")
    xml.append("  <default>")
    xml.append("    <geom condim=\"6\" density=\"1\"/>")
    xml.append("    <joint damping=\"0.001\"/>")
    xml.append("    <motor ctrlrange=\"-1 1\" ctrllimited=\"true\" forcerange=\"-.05 .05\" forcelimited=\"true\"/>")
    xml.append("    <velocity ctrlrange=\"-1 1\" ctrllimited=\"true\" forcerange=\"-.05 .05\" forcelimited=\"true\"/>")
    xml.append("    <site size=\"0.032\" type=\"sphere\"/>")
    xml.append("  </default>")
    xml.append("  <worldbody>")
    xml.append("    <light cutoff=\"100\" diffuse=\"1 1 1\" dir=\"0 0 -1\" directional=\"true\" pos=\"0 0 0.5\" castshadow=\"false\"/>")
    xml.append("    <camera name=\"fixednear\" pos=\"0 -2 2\" zaxis=\"0 -1 1\"/>")
    xml.append("    <camera name=\"fixedfar\" pos=\"0 -5 10\" zaxis=\"0 -.5 1\"/>")
    xml.append("    <geom name=\"floor\" size=\"3.5 3.5 0.1\" type=\"plane\" condim=\"6\" rgba=\"1 1 1 1\"/>")

    # wall_size = float(config.get("wall_size", 3.5))
    # wall_loc = float(config.get("wall_locate_factor", 3.5))
    # wall_rgba = "0.2 0.2 0.2 0.9"
    # wall_defs = [
    #     ("ltl_wall0", (wall_loc, 0.0, 0.25), (0.05, wall_size, 0.3), 0.0),
    #     ("ltl_wall1", (-wall_loc, 0.0, 0.25), (0.05, wall_size, 0.3), 0.0),
    #     ("ltl_wall2", (0.0, wall_loc, 0.25), (0.05, wall_size, 0.3), 1.57079632679),
    #     ("ltl_wall3", (0.0, -wall_loc, 0.25), (0.05, wall_size, 0.3), 1.57079632679),
    # ]
    # for name, pos, size, rot in wall_defs:
    #     xml.append(f"    <body name=\"{name}\" pos=\"{_fmt(pos)}\" euler=\"0 0 {rot:.8g}\">")
    #     xml.append(f"      <geom name=\"{name}\" type=\"box\" size=\"{_fmt(size)}\" rgba=\"{wall_rgba}\" group=\"1\"/>")
    #     xml.append("    </body>")

    for i in range(spec.agent_num):
        color = (0.7412, 0.0431, 0.1843, 1.0)
        xml.append(f"    <body name=\"agent_{i}\" pos=\"0 0 .1\">")
        xml.append(f"      <camera name=\"vision_{i}\" pos=\"0 0 .15\" xyaxes=\"0 -1 0 .4 0 1\" fovy=\"90\"/>")
        xml.append(f"      <joint type=\"slide\" axis=\"1 0 0\" name=\"x_{i}\" damping=\"0.01\"/>")
        xml.append(f"      <joint type=\"slide\" axis=\"0 1 0\" name=\"y_{i}\" damping=\"0.01\"/>")
        xml.append(f"      <joint type=\"hinge\" axis=\"0 0 1\" name=\"z_{i}\" damping=\"0.005\"/>")
        xml.append(f"      <geom name=\"agent_{i}\" type=\"sphere\" size=\".1\" friction=\"1 0.01 0.01\" rgba=\"{_fmt(color)}\"/>")
        xml.append(f"      <geom name=\"pointarrow_{i}\" pos=\"0.1 0 0\" size=\"0.05 0.05 0.05\" type=\"box\" rgba=\"{_fmt(color)}\"/>")
        xml.append(f"      <site name=\"agent_site_{i}\" rgba=\"1 0 0 .1\"/>")
        xml.append("    </body>")

    zone_idx = 0
    for color_name, size, num in zip(spec.zone_colors, spec.zone_size, spec.zone_num):
        rgba = COLOR_RGBA[color_name]
        for _ in range(num):
            name = escape(f"zone_{color_name}_{zone_idx}")
            xml.append(f"    <body name=\"{name}\" pos=\"0 0 0.02\">")
            xml.append(f"      <joint type=\"slide\" axis=\"1 0 0\" name=\"{name}_x\" damping=\"0\"/>")
            xml.append(f"      <joint type=\"slide\" axis=\"0 1 0\" name=\"{name}_y\" damping=\"0\"/>")
            xml.append(f"      <geom name=\"{name}\" type=\"cylinder\" size=\"{size:.8g} 0.01\" rgba=\"{_fmt(rgba)}\" contype=\"0\" conaffinity=\"0\" group=\"2\"/>")
            xml.append("    </body>")
            zone_idx += 1

    xml.append("  </worldbody>")
    xml.append("  <sensor>")
    for i in range(spec.agent_num):
        xml.append(f"    <accelerometer site=\"agent_site_{i}\" name=\"accelerometer_{i}\"/>")
        xml.append(f"    <velocimeter site=\"agent_site_{i}\" name=\"velocimeter_{i}\"/>")
        xml.append(f"    <gyro site=\"agent_site_{i}\" name=\"gyro_{i}\"/>")
        xml.append(f"    <magnetometer site=\"agent_site_{i}\" name=\"magnetometer_{i}\"/>")
        xml.append(f"    <subtreecom body=\"agent_{i}\" name=\"subtreecom_{i}\"/>")
        xml.append(f"    <subtreelinvel body=\"agent_{i}\" name=\"subtreelinvel_{i}\"/>")
        xml.append(f"    <subtreeangmom body=\"agent_{i}\" name=\"subtreeangmom_{i}\"/>")
    xml.append("  </sensor>")
    xml.append("  <actuator>")
    for i in range(spec.agent_num):
        xml.append(f"    <motor gear=\"0.3 0 0 0 0 0\" site=\"agent_site_{i}\" name=\"x_{i}\"/>")
        xml.append(f"    <velocity gear=\"0.3\" joint=\"z_{i}\" name=\"z_{i}\"/>")
    xml.append("  </actuator>")
    xml.append("</mujoco>")
    return "\n".join(xml)



class SafeMultiAgentMJXEnv:
    """TokenEnv-compatible shell for the future MJX multi-agent environment."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        self.spec = make_layout_spec(self.config)
        self.num_agents = self.spec.agent_num
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.n_tokens = len(self.spec.zone_colors)
        
        self.max_steps_in_episode = int(self.config["max_steps"])
        self.frameskip = int(self.config["frameskip"])
        self.fixed_layout_seed = self.config.get("fixed_layout_seed")
        self.agent_radius = float(self.config["agent_radius"])
        self.action_dim = 2
        # self.action_shape = (self.action_dim,)
        self.sensor_names = ("accelerometer", "velocimeter", "magnetometer", "gyro")
        self.sensor_dim = 3
        self.lidar_bin_dim = int(self.config["lidar_bin_dim"])
        self.lidar_exp_gain = float(self.config["lidar_exp_gain"])
        self.lidar_alias = bool(self.config["lidar_alias"])
        self.lidar_zone_colors = tuple(sorted(set(self.spec.zone_colors)))
        self.lidar_zone_color_ids = jnp.asarray(
            [COLOR_IDS[color] for color in self.lidar_zone_colors],
            dtype=jnp.int32,
        )
        self.lidar_group_count = len(self.lidar_zone_colors) + 1
        self.sensor_obs_dim = len(self.sensor_names) * self.sensor_dim
        self.lidar_obs_dim = self.lidar_bin_dim * self.lidar_group_count
        self.obs_shape = (self.sensor_obs_dim + self.lidar_obs_dim,)
        self.action_spaces = {
            agent: Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=jnp.float32)
            for agent in self.agents
        }
        self.observation_spaces = {
            agent: Box(low=-jnp.inf, high=jnp.inf, shape=self.obs_shape, dtype=jnp.float32)
            for agent in self.agents
        }

        # self.sample_layout = make_sample_layout_fn(self.spec)
        # self.sample_layout_batched = make_batched_sample_layout_fn(self.spec)

        self.xml_string = _build_fixed_topology_point_xml(self.config, self.spec)
        self.mj_model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.model = mjx.put_model(self.mj_model)
        self.base_data = mjx.make_data(self.model)

        self.agent_qposadr = jnp.asarray(
            [
                [
                    int(self.mj_model.jnt_qposadr[self.mj_model.joint(f"x_{i}").id]),
                    int(self.mj_model.jnt_qposadr[self.mj_model.joint(f"y_{i}").id]),
                    int(self.mj_model.jnt_qposadr[self.mj_model.joint(f"z_{i}").id]),
                ]
                for i in range(self.num_agents)
            ],
            dtype=jnp.int32,
        )
        zone_names = [
            f"zone_{color_name}_{idx}"
            for idx, color_name in enumerate(
                color
                for color, num in zip(self.spec.zone_colors, self.spec.zone_num)
                for _ in range(num)
            )
        ]
        self.zone_color_ids = jnp.asarray(
            [
                COLOR_IDS[color]
                for color, num in zip(self.spec.zone_colors, self.spec.zone_num)
                for _ in range(num)
            ],
            dtype=jnp.int32,
        )
        self.zone_sizes = jnp.asarray(
            [
                size
                for size, num in zip(self.spec.zone_size, self.spec.zone_num)
                for _ in range(num)
            ],
            dtype=jnp.float32,
        )
        self.zone_qposadr = jnp.asarray(
            [
                [
                    int(self.mj_model.jnt_qposadr[self.mj_model.joint(f"{name}_x").id]),
                    int(self.mj_model.jnt_qposadr[self.mj_model.joint(f"{name}_y").id]),
                ]
                for name in zone_names
            ],
            dtype=jnp.int32,
        )
        self.agent_body_ids = jnp.asarray(
            [int(self.mj_model.body(f"agent_{i}").id) for i in range(self.num_agents)],
            dtype=jnp.int32,
        )
        self.zone_body_ids = jnp.asarray(
            [int(self.mj_model.body(name).id) for name in zone_names],
            dtype=jnp.int32,
        )
        sensor_adr = []
        for i in range(self.num_agents):
            per_agent = []
            for sensor_name in self.sensor_names:
                sensor_id = int(self.mj_model.sensor(f"{sensor_name}_{i}").id)
                dim = int(self.mj_model.sensor_dim[sensor_id])
                if dim != self.sensor_dim:
                    raise ValueError(
                        f"Expected {sensor_name}_{i} dim {self.sensor_dim}, got {dim}."
                    )
                per_agent.append(int(self.mj_model.sensor_adr[sensor_id]))
            sensor_adr.append(per_agent)
        self.agent_sensor_adr = jnp.asarray(sensor_adr, dtype=jnp.int32)
        self.sensor_offsets = jnp.arange(self.sensor_dim, dtype=jnp.int32)

        self.init_state = None

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: jax.Array,
    ) -> Tuple[Dict[str, jax.Array], SafeMultiAgentMJXEnvState, Layout]:
        """Sample a new layout and reset MJX data without rebuilding the model."""

        if self.fixed_layout_seed is not None:
            key = jax.random.PRNGKey(self.fixed_layout_seed)
        state = self.sample_init_state(key)
        obs = self.get_obs(state=state)
        return obs, state# , state.layout

    def step(
        self,
        key: jax.Array,
        state: SafeMultiAgentMJXEnvState,
        actions: Dict[str, jax.Array],
        reset_state: Optional[SafeMultiAgentMJXEnvState] = None,
    ) -> Tuple[
        Dict[str, jax.Array],
        SafeMultiAgentMJXEnvState,
        Dict[str, jax.Array],
        Dict[str, jax.Array],
        Dict,
    ]:
        """TokenEnv/MultiAgentEnv-compatible step with auto-reset."""

        key, key_reset = jax.random.split(key)
        obs_st, state_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is None:
            obs_re, state_re, _layout_re = self.reset(key_reset)
        else:
            state_re = reset_state
            obs_re = self.get_obs(state_re)

        next_state = jax.tree.map(
            lambda reset_v, step_v: jax.lax.select(dones["__all__"], reset_v, step_v),
            state_re,
            state_st,
        )
        obs = jax.tree.map(
            lambda reset_v, step_v: jax.lax.select(dones["__all__"], reset_v, step_v),
            obs_re,
            obs_st,
        )
        return obs, next_state, rewards, dones, infos

    # @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: jax.Array,
        state: SafeMultiAgentMJXEnvState,
        actions: Dict[str, jax.Array],
    ) -> Tuple[
        Dict[str, jax.Array],
        SafeMultiAgentMJXEnvState,
        Dict[str, jax.Array],
        Dict[str, jax.Array],
        Dict,
    ]:
        del key

        action_matrix = jnp.stack(
            [jnp.asarray(actions[agent], dtype=jnp.float32) for agent in self.agents],
            axis=0,
        )
        ctrl = action_matrix.reshape((-1,))

        data = state.data.replace(ctrl=ctrl)

        def step_once(_idx, data_i):
            return mjx.step(state.model, data_i)

        data = jax.lax.fori_loop(0, self.frameskip, step_once, data)
        data = mjx.forward(state.model, data)

        new_state = SafeMultiAgentMJXEnvState(
            model=state.model,
            data=data,
            layout=state.layout,
            time=state.time + 1,
        )

        collisions = self.agent_collision_mask(new_state)
        _rewards = jnp.where(
            collisions,
            jnp.full((self.num_agents,), -1.0, dtype=jnp.float32),
            jnp.zeros((self.num_agents,), dtype=jnp.float32),
        )
        rewards = {agent: _rewards[i] for i, agent in enumerate(self.agents)}

        timeout = new_state.time >= self.max_steps_in_episode
        _dones = jnp.logical_or(
            collisions,
            jnp.full((self.num_agents,), timeout, dtype=bool),
        )
        dones = {agent: _dones[i] for i, agent in enumerate(self.agents)}
        dones.update({"__all__": jnp.logical_or(jnp.any(collisions), timeout)})

        obs = self.get_obs(state=new_state)
        info = {}
        return obs, new_state, rewards, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def agent_collision_mask(self, state: SafeMultiAgentMJXEnvState) -> jax.Array:
        agent_xy = state.data.xpos[self.agent_body_ids, :2]
        dist = jnp.linalg.norm(agent_xy[:, None, :] - agent_xy[None, :, :], axis=-1)
        not_self = ~jnp.eye(self.num_agents, dtype=bool)
        pair_collision = jnp.logical_and(dist <= self.agent_radius, not_self)
        return jnp.any(pair_collision, axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def zone_label_matrix(self, state: SafeMultiAgentMJXEnvState) -> jax.Array:
        agent_xy = state.data.xpos[self.agent_body_ids, :2]
        zone_xy = state.data.xpos[self.zone_body_ids, :2]
        dist = jnp.linalg.norm(agent_xy[:, None, :] - zone_xy[None, :, :], axis=-1)
        in_zone = dist <= self.zone_sizes[None, :]

        per_color = []
        for color_id in self.lidar_zone_color_ids:
            mask = self.zone_color_ids == color_id
            per_color.append(jnp.any(jnp.logical_and(in_zone, mask[None, :]), axis=1))
        return jnp.stack(per_color, axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def label_f(self, state: SafeMultiAgentMJXEnvState) -> Dict[str, jax.Array]:
        matches_any = self.zone_label_matrix(state)
        has_match = jnp.any(matches_any, axis=1)
        label_idx = jnp.argmax(matches_any, axis=1)
        agent_zone_matches = jnp.where(has_match, label_idx, -1)
        return {
            agent: agent_zone_matches[agent_idx]
            for agent_idx, agent in enumerate(self.agents)
        }

    def _pseudo_lidar(
        self,
        agent_xy: jax.Array,
        agent_theta: jax.Array,
        object_xy: jax.Array,
        object_mask: jax.Array,
    ) -> jax.Array:
        delta = object_xy - agent_xy[None, :]
        c = jnp.cos(agent_theta)
        s = jnp.sin(agent_theta)
        ego_x = delta[:, 0] * c + delta[:, 1] * s
        ego_y = -delta[:, 0] * s + delta[:, 1] * c

        dist = jnp.linalg.norm(jnp.stack([ego_x, ego_y], axis=-1), axis=-1)
        angle = jnp.mod(jnp.arctan2(ego_y, ego_x), 2.0 * jnp.pi)
        bin_size = (2.0 * jnp.pi) / float(self.lidar_bin_dim)
        bins = jnp.floor(angle / bin_size).astype(jnp.int32)
        bins = jnp.clip(bins, 0, self.lidar_bin_dim - 1)

        sensor = jnp.exp(-self.lidar_exp_gain * dist)
        sensor = jnp.where(object_mask, sensor, 0.0)

        lidar = jnp.zeros((self.lidar_bin_dim,), dtype=jnp.float32)
        lidar = lidar.at[bins].max(sensor)

        if self.lidar_alias:
            alias = (angle - bins.astype(jnp.float32) * bin_size) / bin_size
            plus_bins = (bins + 1) % self.lidar_bin_dim
            minus_bins = (bins - 1) % self.lidar_bin_dim
            lidar = lidar.at[plus_bins].max(sensor * alias)
            lidar = lidar.at[minus_bins].max(sensor * (1.0 - alias))
        return lidar

    @partial(jax.jit, static_argnums=(0,))
    def get_lidar_obs(self, state: SafeMultiAgentMJXEnvState) -> jax.Array:
        zone_xy = state.data.xpos[self.zone_body_ids, :2]
        agent_xy = state.data.xpos[self.agent_body_ids, :2]
        xmat = state.data.xmat[self.agent_body_ids].reshape((self.num_agents, 3, 3))
        agent_theta = jnp.arctan2(xmat[:, 1, 0], xmat[:, 0, 0])

        zone_lidar = []
        for color_id in self.lidar_zone_color_ids:
            mask = self.zone_color_ids == color_id
            per_agent = jax.vmap(
                lambda xy, theta: self._pseudo_lidar(xy, theta, zone_xy, mask),
            )(agent_xy, agent_theta)
            zone_lidar.append(per_agent)

        other_mask = ~jnp.eye(self.num_agents, dtype=bool)
        other_lidar = jax.vmap(
            lambda xy, theta, mask: self._pseudo_lidar(xy, theta, agent_xy, mask),
        )(agent_xy, agent_theta, other_mask)

        lidar_groups = jnp.stack(zone_lidar + [other_lidar], axis=1)
        return lidar_groups.reshape((self.num_agents, -1))

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: SafeMultiAgentMJXEnvState) -> Dict[str, jax.Array]:
        """Read per-agent proprioceptive sensors from MJX data."""

        sensor_values = state.data.sensordata[
            self.agent_sensor_adr[:, :, None] + self.sensor_offsets[None, None, :]
        ]
        sensor_obs = sensor_values.reshape((self.num_agents, -1))
        lidar_obs = self.get_lidar_obs(state)
        obs = jnp.concatenate([sensor_obs, lidar_obs], axis=-1)
        return {agent: obs[i] for i, agent in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def sample_init_state(self, key: jax.Array) -> SafeMultiAgentMJXEnvState:
        layout = sample_layout(key, self.spec)

        body_pos = self.model.body_pos.at[self.agent_body_ids, :2].set(layout.agent_xy)
        body_quat = self.model.body_quat.at[self.agent_body_ids].set(rot2quat(layout.agent_rot))
        model = self.model.replace(body_pos=body_pos, body_quat=body_quat)

        qpos = jnp.zeros_like(self.base_data.qpos)
        qpos = qpos.at[self.agent_qposadr[:, 0]].set(0.0)
        qpos = qpos.at[self.agent_qposadr[:, 1]].set(0.0)
        qpos = qpos.at[self.agent_qposadr[:, 2]].set(0.0)
        qpos = qpos.at[self.zone_qposadr[:, 0]].set(layout.zone_xy[:, 0])
        qpos = qpos.at[self.zone_qposadr[:, 1]].set(layout.zone_xy[:, 1])

        data = self.base_data.replace(
            qpos=qpos,
            qvel=jnp.zeros_like(self.base_data.qvel),
            ctrl=jnp.zeros_like(self.base_data.ctrl),
        )
        data = mjx.forward(model, data)

        return SafeMultiAgentMJXEnvState(
            model=model,
            data=data,
            layout=layout,
            time=jnp.asarray(0, dtype=jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def set_layout(
        self,
        state: SafeMultiAgentMJXEnvState,
        agent_xy: jax.Array,
        agent_rot: jax.Array,
        zone_xy: jax.Array,
    ) -> SafeMultiAgentMJXEnvState:
        """Overwrite an existing state with explicit layout arrays."""

        object_xy = jnp.concatenate([zone_xy, agent_xy], axis=0)
        object_keepout = jnp.concatenate(
            [
                self.zone_sizes + 0.5 * float(self.config["zone_keepout"]),
                jnp.full((self.num_agents,), self.spec.agent_keepout, dtype=jnp.float32),
            ],
        )
        layout = Layout(
            valid=jnp.asarray(True),
            object_xy=object_xy,
            object_keepout=object_keepout,
            zone_xy=zone_xy,
            zone_size=self.zone_sizes,
            zone_color_id=self.zone_color_ids,
            agent_xy=agent_xy,
            agent_rot=agent_rot,
        )

        body_pos = state.model.body_pos.at[self.agent_body_ids, :2].set(agent_xy)
        body_quat = state.model.body_quat.at[self.agent_body_ids].set(rot2quat(agent_rot))
        model = state.model.replace(body_pos=body_pos, body_quat=body_quat)

        qpos = state.data.qpos
        qpos = qpos.at[self.agent_qposadr[:, 0]].set(0.0)
        qpos = qpos.at[self.agent_qposadr[:, 1]].set(0.0)
        qpos = qpos.at[self.agent_qposadr[:, 2]].set(0.0)
        qpos = qpos.at[self.zone_qposadr[:, 0]].set(zone_xy[:, 0])
        qpos = qpos.at[self.zone_qposadr[:, 1]].set(zone_xy[:, 1])

        data = state.data.replace(
            qpos=qpos,
            qvel=jnp.zeros_like(state.data.qvel),
            ctrl=jnp.zeros_like(state.data.ctrl),
        )
        data = mjx.forward(model, data)

        return SafeMultiAgentMJXEnvState(
            model=model,
            data=data,
            layout=layout,
            time=jnp.asarray(0, dtype=jnp.int32),
        )

    def observation_space(self, agent: str):# -> Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str):# -> Space:
        return self.action_spaces[agent]

    # @property
    # def name(self) -> str:
    #     return type(self).__name__

    # @property
    # def agent_classes(self) -> dict[str, list[str]]:
    #     return {"agents": [self.agent_name for _ in self.agents]}
