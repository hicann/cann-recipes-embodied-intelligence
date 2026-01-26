# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2025 Shanghai Jiao Tong University
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a DiffusionPolicy in simulation using Ascend OM for the UNet.

This script is the DP analogue of act/eval_act_ascend.py, and is also inspired by
(dp legacy) dp/eval/eval.py.

Design:
- Uses standard LeRobot env creation and rollout loops.
- Loads a PyTorch DiffusionPolicy (for encoders, scheduler logic, and queueing).
- Replaces the policy's UNet forward with an Ascend OM-backed module.
- Computes rollout metrics and robust success_rate parsing (final_info + per-step flags).

Expected OM model:
- A UNet exported with inputs (sample, t, global_cond) and output (model_output).
"""

import json
import logging
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

# Ascend/ACL runtime dependencies (available on the target Ascend machine).
# Keep optional so developers can still import this file locally.
_ASCEND_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover
    import acl  # type: ignore
    import acllite_utils as utils  # type: ignore
    import constants as const  # type: ignore
    from acllite_model import AclLiteModel  # type: ignore
    from acllite_resource import resource_list  # type: ignore
except Exception as exc:  # pragma: no cover
    _ASCEND_IMPORT_ERROR = exc
    acl = None  # type: ignore

    class _DummyUtils:  # pragma: no cover
        @staticmethod
        def display_time(fn):
            return fn

        @staticmethod
        def check_ret(*_args, **_kwargs):
            return None

    utils = _DummyUtils()  # type: ignore
    const = SimpleNamespace(SUCCESS=0)  # type: ignore
    AclLiteModel = None  # type: ignore
    resource_list = None  # type: ignore

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, check_env_attributes_and_types, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging, inside_slurm

LOGGER = logging.getLogger(__name__)


def _as_bool(x: object) -> bool | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray) and x.shape == ():
        try:
            return bool(x.item())
        except Exception as err:
            LOGGER.debug("Failed to convert scalar ndarray to bool: %s", err)
            return None
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        try:
            return bool(x.item())
        except Exception as err:
            LOGGER.debug("Failed to convert tensor to bool: %s", err)
            return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(x)
    return None


def _find_success_in_mapping(obj: object, *, max_depth: int = 4) -> bool | None:
    if max_depth <= 0:
        return None
    if not isinstance(obj, Mapping):
        return None

    for key in ("is_success", "success"):
        if key in obj:
            b = _as_bool(obj.get(key))
            if b is not None:
                return b

    for v in obj.values():
        if isinstance(v, Mapping):
            found = _find_success_in_mapping(v, max_depth=max_depth - 1)
            if found is not None:
                return found

    return None


def _extract_successes(
    *,
    info: Mapping[str, object],
    terminated: np.ndarray,
    truncated: np.ndarray,
    num_envs: int,
) -> list[bool]:
    done_step = np.asarray(terminated) | np.asarray(truncated)
    done_ix_global = np.where(done_step.reshape(-1).astype(bool))[0]

    per_step_successes: list[bool] | None = None

    def _combine_final(final_successes: list[bool]) -> list[bool]:
        if per_step_successes is None:
            return final_successes
        out = [False] * num_envs
        for i in range(num_envs):
            out[i] = bool((final_successes[i] or per_step_successes[i]) and bool(done_step[i]))
        return out

    def _flat_values(x: object) -> np.ndarray | None:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            try:
                return x.detach().cpu().reshape(-1).numpy()
            except Exception:
                return None
        if isinstance(x, np.ndarray):
            return x.reshape(-1)
        if isinstance(x, (list, tuple)):
            return np.asarray(list(x)).reshape(-1)
        try:
            return np.asarray([x]).reshape(-1)
        except Exception:
            return None

    def _done_env_indices() -> np.ndarray:
        m = info.get("_final_info")
        if isinstance(m, np.ndarray):
            mm = np.asarray(m).astype(bool).reshape(-1)
            if mm.shape[0] == num_envs:
                return np.where(mm)[0]
        return done_ix_global

    def _assign_with_mask(values: object, mask: object) -> list[bool] | None:
        vv = _flat_values(values)
        mm = _flat_values(mask)
        if vv is None or mm is None:
            return None
        try:
            mm_bool = np.asarray(mm).astype(bool).reshape(-1)
        except Exception:
            return None
        out = [False] * num_envs

        if mm_bool.shape[0] == num_envs:
            if vv.shape[0] == num_envs:
                for env_i in range(num_envs):
                    if not bool(mm_bool[env_i]):
                        continue
                    b = _as_bool(vv[env_i])
                    out[env_i] = bool((b if b is not None else False) and bool(done_step[env_i]))
                return out

            idx = np.where(mm_bool)[0]
            if vv.shape[0] != idx.shape[0]:
                return None
            for j, env_i in enumerate(idx.tolist()):
                b = _as_bool(vv[j])
                out[env_i] = bool((b if b is not None else False) and bool(done_step[env_i]))
            return out

        done_ix = done_ix_global
        if mm_bool.shape[0] == done_ix.shape[0]:
            local_idx = np.where(mm_bool)[0]
            if vv.shape[0] == done_ix.shape[0]:
                vv_sel = vv[local_idx]
            elif vv.shape[0] == local_idx.shape[0]:
                vv_sel = vv
            else:
                return None
            for j, local_j in enumerate(local_idx.tolist()):
                env_i = int(done_ix[local_j])
                b = _as_bool(vv_sel[j])
                out[env_i] = bool((b if b is not None else False) and bool(done_step[env_i]))
            return out

        return None

    def _assign_direct(values: object) -> list[bool] | None:
        vv = _flat_values(values)
        if vv is None:
            return None
        if vv.shape[0] != num_envs:
            return None
        out = [False] * num_envs
        for i in range(num_envs):
            b = _as_bool(vv[i])
            out[i] = bool((b if b is not None else False) and bool(done_step[i]))
        return out

    for key in ("is_success", "success"):
        if key in info:
            masked = _assign_with_mask(info.get(key), info.get(f"_{key}"))
            if masked is not None:
                per_step_successes = masked
                break
            direct = _assign_direct(info.get(key))
            if direct is not None:
                per_step_successes = direct
                break

    final_info = info.get("final_info")
    if isinstance(final_info, np.ndarray):
        try:
            final_info = list(final_info)
        except Exception:
            final_info = None

    if isinstance(final_info, (list, tuple)):
        done_ix = _done_env_indices()
        if len(final_info) == num_envs:
            out: list[bool] = []
            for i, entry in enumerate(final_info):
                if entry is None or not bool(done_step[i]):
                    out.append(False)
                    continue
                if isinstance(entry, Mapping):
                    found = _find_success_in_mapping(entry)
                    out.append(bool(found) if found is not None else False)
                else:
                    out.append(False)
            return _combine_final(out)
        if len(final_info) == len(done_ix):
            out = [False] * num_envs
            for j, env_i in enumerate(done_ix.tolist()):
                entry = final_info[j]
                if entry is None or not bool(done_step[env_i]):
                    continue
                if isinstance(entry, Mapping):
                    found = _find_success_in_mapping(entry)
                    out[env_i] = bool(found) if found is not None else False
            return _combine_final(out)

    if isinstance(final_info, Mapping):
        done_ix = _done_env_indices()
        for key in ("is_success", "success"):
            if key not in final_info:
                continue
            vv = _flat_values(final_info.get(key))
            if vv is None:
                continue

            inner_mask = final_info.get(f"_{key}")
            if inner_mask is not None:
                mm = _flat_values(inner_mask)
                if mm is not None:
                    mm_bool = np.asarray(mm).astype(bool).reshape(-1)
                    if mm_bool.shape[0] == done_ix.shape[0] and vv.shape[0] == done_ix.shape[0]:
                        done_ix2 = done_ix[mm_bool]
                        vv2 = vv[mm_bool]
                    else:
                        done_ix2 = done_ix
                        vv2 = vv
                else:
                    done_ix2 = done_ix
                    vv2 = vv
            else:
                done_ix2 = done_ix
                vv2 = vv

            if vv2.shape[0] != done_ix2.shape[0]:
                continue

            out = [False] * num_envs
            for j, env_i in enumerate(done_ix2.tolist()):
                b = _as_bool(vv2[j])
                out[env_i] = bool((b if b is not None else False) and bool(done_step[env_i]))
            return _combine_final(out)

        found = _find_success_in_mapping(final_info)
        if found is not None and bool(done_step.any()):
            out = [False] * num_envs
            for env_i in _done_env_indices().tolist():
                out[env_i] = bool(found) and bool(done_step[env_i])
            return _combine_final(out)

    if per_step_successes is not None:
        return per_step_successes

    return [False] * num_envs


def _unwrap_single_vec_env(envs: object) -> gym.vector.VectorEnv:
    if isinstance(envs, gym.vector.VectorEnv):
        return envs

    if isinstance(envs, Mapping):
        if len(envs) != 1:
            raise ValueError(f"Expected exactly 1 env suite, got: {list(envs.keys())}")
        suite_envs = next(iter(envs.values()))
        if isinstance(suite_envs, gym.vector.VectorEnv):
            return suite_envs
        if isinstance(suite_envs, Mapping):
            if len(suite_envs) != 1:
                raise ValueError(f"Expected exactly 1 task env, got: {list(suite_envs.keys())}")
            vec = next(iter(suite_envs.values()))
            if isinstance(vec, gym.vector.VectorEnv):
                return vec
            raise TypeError(f"Unexpected task env type: {type(vec)}")
        raise TypeError(f"Unexpected suite env container type: {type(suite_envs)}")

    if isinstance(envs, Sequence) and not isinstance(envs, (str, bytes)):
        if len(envs) != 1:
            raise ValueError(f"Expected a single env, got {len(envs)} items")
        return _unwrap_single_vec_env(envs[0])

    raise TypeError(f"Unsupported make_env() return type: {type(envs)}")


class AclLiteResource:
    def __init__(self, device_id: int = 0) -> None:
        self.device_id = int(device_id)
        self.context = None
        self.stream = None
        self.run_mode = None

    def init(self) -> None:
        if acl is None:  # pragma: no cover
            raise RuntimeError(
                "Ascend ACL runtime is not available in this environment. "
                f"Original import error: {_ASCEND_IMPORT_ERROR}"
            )

        ret = acl.init()
        ret = acl.rt.set_device(self.device_id)
        utils.check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        utils.check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        utils.check_ret("acl.rt.create_stream", ret)

        self.run_mode, ret = acl.rt.get_run_mode()
        utils.check_ret("acl.rt.get_run_mode", ret)

    def __del__(self):
        try:
            if resource_list is not None:
                resource_list.destroy()
        except Exception as err:
            LOGGER.warning("Failed to destroy ACL resource list: %s", err)
        if acl is None:
            return
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)


def _resolve_om_model_path(om_model_path: Path | None, *, policy_dir: Path) -> Path:
    if om_model_path is None:
        raise ValueError("Missing --om_model_path (Path to OM model)")

    p = Path(om_model_path).expanduser()

    # 逻辑修改：优先检查当前路径（无论是绝对路径还是相对路径）
    # 如果当前路径下存在该文件，直接使用，不再强拼 policy_dir
    if p.exists():
        return p.resolve()

    # 如果当前路径找不到，再尝试去 policy_dir 下面找（兼容旧逻辑）
    resolved_in_policy = (policy_dir / p).resolve()
    if resolved_in_policy.exists():
        return resolved_in_policy

    # 都找不到才报错
    raise FileNotFoundError(
        f"OM model path not found.\n"
        f"Checked location 1 (CWD): {p.resolve()}\n"
        f"Checked location 2 (Policy Dir): {resolved_in_policy}"
    )


def _find_policy_postprocessor_dir(policy_dir: Path) -> Path | None:
    direct = policy_dir / "policy_postprocessor.json"
    if direct.exists():
        return policy_dir

    migrated = policy_dir.parent / f"{policy_dir.name}_migrated" / "policy_postprocessor.json"
    if migrated.exists():
        return migrated.parent

    return None


def _find_policy_preprocessor_dir(policy_dir: Path) -> Path | None:
    direct = policy_dir / "policy_preprocessor.json"
    if direct.exists():
        return policy_dir

    migrated = policy_dir.parent / f"{policy_dir.name}_migrated" / "policy_preprocessor.json"
    if migrated.exists():
        return migrated.parent

    return None


def _maybe_load_policy_preprocessor(policy_dir: Path) -> PolicyProcessorPipeline | None:
    pre_dir = _find_policy_preprocessor_dir(policy_dir)
    if pre_dir is None:
        return None
    try:
        return PolicyProcessorPipeline.from_pretrained(pre_dir, config_filename="policy_preprocessor.json")
    except Exception as err:
        logging.warning(
            "Failed to load policy_preprocessor.json from %s. Error: %s",
            pre_dir,
            err,
        )
        return None


def _maybe_load_policy_postprocessor(policy_dir: Path) -> PolicyProcessorPipeline | None:
    post_dir = _find_policy_postprocessor_dir(policy_dir)
    if post_dir is None:
        return None
    try:
        return PolicyProcessorPipeline.from_pretrained(post_dir, config_filename="policy_postprocessor.json")
    except Exception as err:
        logging.warning(
            "Failed to load policy_postprocessor.json from %s. Error: %s",
            post_dir,
            err,
        )
        return None


def _load_policy_processor_or_raise(
    *,
    policy_dir: Path,
    kind: str,
    filename: str,
    overrides: dict[str, object] | None = None,
) -> PolicyProcessorPipeline:
    """Load a processor pipeline and raise a useful error if it cannot be loaded.

    DP evaluation is intentionally strict: silent fallbacks can produce wrong-scale actions.
    """
    if kind == "pre":
        proc_dir = _find_policy_preprocessor_dir(policy_dir)
    elif kind == "post":
        proc_dir = _find_policy_postprocessor_dir(policy_dir)
    else:
        raise ValueError(f"Unknown processor kind: {kind}")

    expected_a = policy_dir / filename
    expected_b = policy_dir.parent / f"{policy_dir.name}_migrated" / filename

    if proc_dir is None:
        raise FileNotFoundError(
            f"Missing {filename}. Expected one of:\n"
            f"- {expected_a}\n"
            f"- {expected_b}\n\n"
            "Generate the processor artifacts (json + referenced *.safetensors) and re-run evaluation."
        )

    try:
        return PolicyProcessorPipeline.from_pretrained(
            proc_dir,
            config_filename=filename,
            overrides=overrides,
        )
    except Exception as err:
        raise RuntimeError(
            f"Found {filename} under {proc_dir}, but failed to load it. "
            "Most commonly, the referenced state *.safetensors file(s) are missing, not copied alongside "
            "the json, or incompatible with the installed lerobot version. "
            f"Original error: {err}"
        ) from err


def _to_numpy(x: object, *, dtype=np.float32) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype, copy=False)
    return np.ascontiguousarray(x)


class OmUnetModule(nn.Module):
    """UNet replacement that calls Ascend OM via AclLiteModel."""

    def __init__(self, om_model_path: Path):
        super().__init__()
        if AclLiteModel is None:  # pragma: no cover
            raise RuntimeError(
                "AclLiteModel is not available. Run on the Ascend machine with ACL runtime installed."
            )
        self._om_model_path = Path(om_model_path)
        self._om_model = AclLiteModel(str(self._om_model_path))

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:  # noqa: ANN001
        if global_cond is None:
            raise ValueError("global_cond is required for Diffusion UNet")

        x_np = _to_numpy(x, dtype=np.float32)
        gc_np = _to_numpy(global_cond, dtype=np.float32)

        # OM models are typically compiled for batch=1. If we get B>1, run per-item and stack.
        batch = int(x_np.shape[0])

        def _mk_t_np(i: int | None) -> np.ndarray:
            if isinstance(timestep, int):
                return np.asarray([int(timestep)], dtype=np.int64)
            if isinstance(timestep, torch.Tensor):
                if timestep.numel() == 1:
                    return np.asarray([int(timestep.item())], dtype=np.int64)
                if i is None:
                    raise ValueError("timestep tensor has more than one element; provide batch index")
                return np.asarray(
                    [int(timestep.detach().cpu().to(torch.int64).reshape(-1)[i].item())],
                    dtype=np.int64,
                )
            # ndarray or list-like
            t_arr = np.asarray(timestep, dtype=np.int64).reshape(-1)
            if t_arr.size == 1:
                return t_arr.copy()
            if i is None or i >= t_arr.size:
                raise ValueError("timestep array has more than one element; provide valid batch index")
            return np.asarray([int(t_arr[i])], dtype=np.int64)

        if batch <= 1:
            t_np = _mk_t_np(None)
            out_list = self._om_model.execute([x_np, t_np, gc_np])
            if not isinstance(out_list, (list, tuple)) or len(out_list) < 1:
                raise RuntimeError(f"Unexpected OM outputs: {type(out_list)}")
            out = np.asarray(out_list[0], dtype=np.float32)
            return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)

        outs = []
        for i in range(batch):
            xi = x_np[i: i + 1]
            gci = gc_np[i: i + 1]
            ti = _mk_t_np(i)
            out_list = self._om_model.execute([xi, ti, gci])
            if not isinstance(out_list, (list, tuple)) or len(out_list) < 1:
                raise RuntimeError(f"Unexpected OM outputs (batched) at i={i}: {type(out_list)}")
            outs.append(np.asarray(out_list[0], dtype=np.float32))
        out = np.concatenate(outs, axis=0)
        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)


@dataclass
class RolloutConfig:
    seeds: list[int] | None = None
    return_observations: bool = False
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None
    policy_preprocessor: PolicyProcessorPipeline | None = None
    policy_postprocessor: PolicyProcessorPipeline | None = None


def rollout(env: gym.vector.VectorEnv, policy: PreTrainedPolicy, cfg: RolloutConfig) -> dict:
    device = get_device_from_parameters(policy)

    if cfg.policy_preprocessor is None or cfg.policy_postprocessor is None:
        raise ValueError("DP Ascend eval requires both policy_preprocessor and policy_postprocessor")

    policy.reset()
    observation, info = env.reset(seed=cfg.seeds)
    if cfg.render_callback is not None:
        cfg.render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),
        leave=False,
    )
    check_env_attributes_and_types(env)

    while not np.all(done):
        observation = preprocess_observation(observation)
        if cfg.return_observations:
            from copy import deepcopy

            all_observations.append(deepcopy(observation))

        observation = {key: observation[key].to(device, non_blocking=False) for key in observation}
        observation = add_envs_task(env, observation)

        observation = cfg.policy_preprocessor.process_observation(observation)

        with torch.inference_mode():
            action_t = policy.select_action(observation)

        action_t = cfg.policy_postprocessor.process_action(action_t)
        if not isinstance(action_t, torch.Tensor):
            raise TypeError(f"Postprocessor returned unexpected type: {type(action_t)}")

        action = np.ascontiguousarray(action_t.detach().cpu().numpy())
        if action.ndim != 2:
            raise ValueError(f"Action dimensions should be (batch, action_dim), got {action.shape}")

        observation, reward, terminated, truncated, info = env.step(action)
        if cfg.render_callback is not None:
            cfg.render_callback(env)

        successes = _extract_successes(
            info=info,
            terminated=terminated,
            truncated=truncated,
            num_envs=env.num_envs,
        )

        done = terminated | truncated | done

        all_actions.append(torch.from_numpy(action))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    if cfg.return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(observation)

    ret = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if cfg.return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret["observation"] = stacked_observations

    return ret


@dataclass
class EvalPolicyConfig:
    n_episodes: int
    max_episodes_rendered: int = 0
    videos_dir: Path | None = None
    return_episode_data: bool = False
    start_seed: int | None = None
    policy_preprocessor: PolicyProcessorPipeline | None = None
    policy_postprocessor: PolicyProcessorPipeline | None = None


def eval_policy(env: gym.vector.VectorEnv, policy: PreTrainedPolicy, cfg: EvalPolicyConfig) -> dict:
    if cfg.policy_preprocessor is None or cfg.policy_postprocessor is None:
        raise ValueError("DP Ascend eval requires both policy_preprocessor and policy_postprocessor")
    if cfg.max_episodes_rendered > 0 and not cfg.videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    start = time.time()

    n_batches = cfg.n_episodes // env.num_envs + int((cfg.n_episodes % env.num_envs) != 0)

    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []
    n_episodes_rendered = 0

    def render_frame(env: gym.vector.VectorEnv):
        nonlocal n_episodes_rendered
        if n_episodes_rendered >= cfg.max_episodes_rendered:
            return
        n_to_render_now = min(cfg.max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if cfg.max_episodes_rendered > 0:
        video_paths: list[str] = []

    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        if cfg.max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if cfg.start_seed is None:
            seeds = None
        else:
            seeds = range(
                cfg.start_seed + (batch_ix * env.num_envs), cfg.start_seed + ((batch_ix + 1) * env.num_envs)
            )

        rollout_cfg = RolloutConfig(
            seeds=list(seeds) if seeds else None,
            return_observations=cfg.return_episode_data,
            render_callback=render_frame if cfg.max_episodes_rendered > 0 else None,
            policy_preprocessor=cfg.policy_preprocessor,
            policy_postprocessor=cfg.policy_postprocessor,
        )

        rollout_data = rollout(env, policy, rollout_cfg)

        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()

        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        if cfg.max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= cfg.max_episodes_rendered:
                    break

                cfg.videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = cfg.videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[: cfg.n_episodes]).item() * 100:.1f}%"}
        )

    for thread in threads:
        thread.join()

    info = {
        "per_episode": [],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[: cfg.n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[: cfg.n_episodes])),
            "pc_success": float(np.nanmean(all_successes[: cfg.n_episodes]) * 100),
            "eval_time_sec": float(time.time() - start),
        },
    }
    for episode_ix, (sum_reward, max_reward, success, seed) in enumerate(
        zip(
            sum_rewards[: cfg.n_episodes],
            max_rewards[: cfg.n_episodes],
            all_successes[: cfg.n_episodes],
            all_seeds[: cfg.n_episodes],
            strict=True,
        )
    ):
        info["per_episode"].append(
            {
                "episode_ix": episode_ix,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
        )
    if cfg.max_episodes_rendered > 0:
        info["video_paths"] = video_paths
    return info


@dataclass
class EvalPipelineConfigWithOM(EvalPipelineConfig):
    # 在这里显式定义参数，这样命令行才能识别 --om_model_path
    om_model_path: str | None = None


@parser.wrap()
def eval_main(cfg: EvalPipelineConfigWithOM):
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device if cfg.policy is not None else "cpu", log=True)
    set_seed(cfg.seed)

    if cfg.policy is None or cfg.policy.pretrained_path is None:
        raise ValueError("This evaluator requires --policy.path to a pretrained DiffusionPolicy")

    policy_dir = Path(cfg.policy.pretrained_path)

    om_model_path = _resolve_om_model_path(cfg.om_model_path, policy_dir=policy_dir)
    logging.info("Using OM model: %s", om_model_path)

    # Initialize ACL resource once.
    acl_resource = AclLiteResource(device_id=0)
    acl_resource.init()

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    env = _unwrap_single_vec_env(envs)

    logging.info("Making policy.")
    # Pass rename_map to relax strict visual feature validation when env uses different keys.
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.to(device)
    policy.eval()

    # Replace UNet with OM-backed module.
    if not hasattr(policy, "diffusion") or not hasattr(policy.diffusion, "unet"):
        raise TypeError("Policy does not look like a DiffusionPolicy (missing policy.diffusion.unet)")
    policy.diffusion.unet = OmUnetModule(om_model_path)

    # Inject rename_map into the preprocessor's RenameObservations step (if present).
    pre_overrides: dict[str, object] | None = None
    if isinstance(cfg.rename_map, dict) and len(cfg.rename_map) > 0:
        pre_overrides = {"rename_observations_processor": {"rename_map": cfg.rename_map}}

    policy_preprocessor = _load_policy_processor_or_raise(
        policy_dir=policy_dir,
        kind="pre",
        filename="policy_preprocessor.json",
        overrides=pre_overrides,
    )
    logging.info("Loaded policy_preprocessor.json (required)")

    policy_postprocessor = _load_policy_processor_or_raise(
        policy_dir=policy_dir,
        kind="post",
        filename="policy_postprocessor.json",
        overrides=None,
    )
    logging.info("Loaded policy_postprocessor.json (required)")

    autocast_ctx = (
        torch.autocast(device_type=device.type)
        if (cfg.policy.use_amp if cfg.policy else False)
        else nullcontext()
    )
    eval_cfg = EvalPolicyConfig(
        n_episodes=cfg.eval.n_episodes,
        max_episodes_rendered=10,
        videos_dir=Path(cfg.output_dir) / "videos",
        return_episode_data=False,
        start_seed=cfg.seed,
        policy_preprocessor=policy_preprocessor,
        policy_postprocessor=policy_postprocessor,
    )

    with torch.no_grad(), autocast_ctx:
        info = eval_policy(env, policy, eval_cfg)

    logging.info("Aggregated metrics: %s", info["aggregated"])

    with open(Path(cfg.output_dir) / "eval_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    close_envs(envs)

    logging.info("End of eval")


def main():
    init_logging()
    try:
        eval_main()
    except BrokenPipeError as err:
        # Common when piping output to tools like `head` which close stdout early.
        # Treat as a clean exit.
        try:
            sys.stdout.close()
        except Exception as close_err:
            logging.debug("Failed to close stdout cleanly after BrokenPipeError: %s", close_err)
        raise SystemExit(0) from err


if __name__ == "__main__":
    main()
