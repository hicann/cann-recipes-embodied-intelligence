# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
import time
import os
import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, List

import numpy as np
import torch

# Module imports with fallback handling
try:
    import algorithms as alg
except ImportError:
    alg = None

try:
    from lltk.task_registry import registry
    from lltk.utils.sprint import sprint, ss
    from lltk.utils.stream import DataStream
    from lltk.utils.velocity_tracking.arguments import make_play_argparser
    from lltk.utils.velocity_tracking.cfg_helper import overwrite_cfg, process_curriculum, sync_cfg, CfgOverwriteOptions
    from lltk.utils.velocity_tracking.commander import make_commander
    from lltk.utils.velocity_tracking.run_loader import RunLoader
    from lltk.utils.velocity_tracking.symmetry import symmetry_wrapper
except ImportError:
    # Mock implementations for import fallback
    def _mock_get_device(d=None):
        return torch.device('cpu')

    def _mock_general_actor_make(*a, **k):
        return torch.nn.Linear(10, 6)

    def _mock_adapt_ddp_state_dict(s, *a, **k):
        return s

    def _mock_inference_module(m):
        return m

    def _mock_make_nn(*a, **k):
        return torch.nn.Linear(10, 1)

    def _mock_registry_get(*a, **k):
        return None

    def _mock_sprint_bc(t):
        return t

    def _mock_datastream_publish(*a):
        pass

    def _mock_datastream_set_fixed_interval_mode(*a):
        pass

    def _mock_datastream(*a, **k):
        return type('DS', (), {
            'publish': _mock_datastream_publish,
            'set_fixed_interval_mode': _mock_datastream_set_fixed_interval_mode
        })()

    def _mock_play_argparser():
        return argparse.ArgumentParser()

    def _mock_cfg_helper(*a, **k):
        pass

    def _mock_commander(*a, **k):
        return None

    def _mock_run_loader(r, **k):
        return type('RL', (), {
            'task': type('T', (), {'robot': 'g1'}),
            'iteration': 0,
            'cfg': {'ppo': {}},
            'scaling_dict': {},
            'weight_path': ''
        })()

    def _mock_symmetry_wrapper(p, *a, **k):
        return p

    if alg is None:
        alg = type('alg', (), {
            'get_device': _mock_get_device,
            'GeneralActor': type('GeneralActor', (), {'make': staticmethod(_mock_general_actor_make)}),
            'utils': lambda: type('utils', (), {'adapt_ddp_state_dict': staticmethod(_mock_adapt_ddp_state_dict)})(),
            'InferenceModule': _mock_inference_module,
            'get_pseudo_symmetric_state': None,
            'make_nn': _mock_make_nn,
        })()
    registry = type('registry', (), {'get': _mock_registry_get})()
    sprint = type('sprint', (), {'sprint': print, 'table': print, 'bC': _mock_sprint_bc})()
    ss = type('ss', (), {'u_': '', 'r_': ''})()
    DataStream = _mock_datastream
    make_play_argparser = _mock_play_argparser
    overwrite_cfg = _mock_cfg_helper
    process_curriculum = _mock_cfg_helper
    sync_cfg = _mock_cfg_helper
    make_commander = _mock_commander
    RunLoader = _mock_run_loader
    symmetry_wrapper = _mock_symmetry_wrapper

__all__ = ['Player', 'PlayerUnified']


# Robot model path mapping
ROBOT_MODEL_PATHS = {
    'g1': 'resources/g1/g1_12dof.xml',
    'g1_12dof': 'resources/g1/g1_12dof.xml',
    'g1_15dof': 'resources/g1/g1_15dof.xml',
    'g1_23dof': 'resources/g1/g1_23dof.xml',
    'go2': 'resources/go2/go2.xml',
}

# Camera settings per robot
ROBOT_CAMERA_CONFIGS = {
    'g1': {'lookat_z': 0.8, 'cam_distance': 5.0},
    'g1_12dof': {'lookat_z': 0.8, 'cam_distance': 5.0},
    'g1_15dof': {'lookat_z': 0.8, 'cam_distance': 5.0},
    'g1_23dof': {'lookat_z': 0.8, 'cam_distance': 5.0},
    'go2': {'lookat_z': 0.5, 'cam_distance': 4.0},
}
DEFAULT_CAMERA_CONFIG = {'lookat_z': 0.8, 'cam_distance': 5.0}


@dataclass
class VideoConfig:
    """Video recording configuration."""
    enabled: bool = False
    fps: int = 50
    width: int = 640
    height: int = 480
    camera_id: int = 0
    save_interval: int = 1
    format: str = 'mp4'
    save_depth: bool = False
    # MuJoCo render settings
    render_backend: str = 'egl'
    render_quality: str = 'fast'
    cam_azimuth: float = 30.0
    cam_elevation: float = -20.0
    num_workers: int = 8
    # Frame range for offline mode
    frame_start: int = 0
    frame_end: int = -1


@dataclass(slots=True)
class Arguments:
    """Player arguments."""
    # Task config
    run: str = None
    robot: str = None
    tag: str = None
    algorithm: str = 'ppo'
    algorithm_cfg_path: str = None
    # Runtime config
    quiet: bool = False
    seed: int = None
    headless: bool = False
    extra_cfg_files: tuple = ()
    overwrite: tuple = ()
    overwrite_env: tuple = ()
    sync_cfg: bool = True
    # Model config
    device: str = None
    fp16: bool = False
    symmetry: str = None
    # Playback config
    seconds: int = 60
    endless: bool = False
    command: str = 'js'
    speed: float = 1.
    dump: str = None
    timeout: float = None
    foxglove: bool = False
    cache: bool = False
    stochastic: bool = False
    num_steps: int = None  # Custom step count
    # Video config
    video: VideoConfig = field(default_factory=VideoConfig)
    # Render mode: 'realtime' or 'offline'
    render_mode: str = 'realtime'


def _render_frame_parallel(frame_data: Dict) -> Optional[np.ndarray]:
    """Multi-process render worker for offline mode."""
    try:
        import mujoco
        import mujoco.egl
        
        cfg = frame_data['cfg']
        
        gl_context = mujoco.egl.GLContext(cfg['width'], cfg['height'])
        gl_context.make_current()
        
        os.environ['MUJOCO_GL'] = cfg['render_backend']
        os.environ['LIBGL_ALWAYS_INDIRECT'] = '0'
        
        mj_model = mujoco.MjModel.from_xml_path(cfg['model_xml'])
        mj_data = mujoco.MjData(mj_model)
        
        mj_data.qpos[:] = np.array(frame_data['qpos'])
        mj_data.qvel[:] = np.array(frame_data['qvel'])
        mj_data.time = frame_data['time']
        mujoco.mj_fwdPosition(mj_model, mj_data)
        
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance = cfg['cam_distance']
        cam.azimuth = cfg['cam_azimuth']
        cam.elevation = cfg['cam_elevation']
        cam.lookat = np.array([mj_data.qpos[0], mj_data.qpos[1], cfg['lookat_z']])
        
        scene = mujoco.MjvScene(mj_model, maxgeom=10000)
        viewport = mujoco.MjrRect(0, 0, cfg['width'], cfg['height'])
        render_option = mujoco.MjvOption()
        if cfg['quality'] == 'fast':
            render_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        
        context = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_200.value)
        
        mujoco.mjv_updateScene(mj_model, mj_data, render_option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        
        img = np.empty((cfg['height'], cfg['width'], 3), dtype=np.uint8)
        depth = np.empty((cfg['height'], cfg['width']), dtype=np.float32)
        mujoco.mjr_readPixels(img, depth, viewport, context)
        
        gl_context.free()
        return np.flipud(img)
        
    except Exception as e:
        logging.error(f"[Render Error] Frame {frame_data.get('step', '?')}: {e}")
        return None


class VideoWriter:
    """Video writer with frame collection."""
    
    def __init__(self, config: VideoConfig, output_dir: str, run_name: str):
        self.config = config
        self.frames: List[np.ndarray] = []
        self.depth_frames: List[np.ndarray] = []
        self.start_time = time.time()
        
        os.makedirs(output_dir, exist_ok=True)
        time_str = datetime.now(timezone.utc).strftime('%H-%M-%S')
        self.video_path = os.path.join(output_dir, f"{run_name}_{time_str}.{config.format}")
        self.depth_path = os.path.join(
            output_dir, f"{run_name}_{time_str}_depth.{config.format}"
        ) if config.save_depth else None
    
    def add_frame(self, frame: np.ndarray, depth: Optional[np.ndarray] = None):
        self.frames.append(frame.copy())
        if depth is not None and self.config.save_depth:
            d_min, d_max = depth.min(), depth.max()
            depth_norm = (
                ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                if d_max > d_min else np.zeros_like(depth, dtype=np.uint8)
            )
            self.depth_frames.append(depth_norm)
    
    def save(self):
        if not self.frames:
            logging.warning("[Warning] No frames to save")
            return
        
        try:
            import cv2
        except ImportError:
            logging.error("[Error] opencv-python not installed")
            return
        
        fourcc_map = {'mp4': 'mp4v', 'avi': 'XVID'}
        fourcc = cv2.VideoWriter_fourcc(*fourcc_map.get(self.config.format, 'mp4v'))
        h, w = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(self.video_path, fourcc, self.config.fps, (w, h))
        
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(self.video_path, fourcc, self.config.fps, (w, h))
        
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        
        elapsed = time.time() - self.start_time
        file_size = os.path.getsize(self.video_path) / 1024 / 1024
        logging.info(f"[Video Saved] {self.video_path}")
        logging.info(f"  Frames: {len(self.frames)}, "
                     f"Duration: {len(self.frames)/self.config.fps:.2f}s, "
                     f"Size: {file_size:.2f}MB, Time: {elapsed:.2f}s")
        
        if self.depth_frames:
            d_writer = cv2.VideoWriter(self.depth_path, fourcc, self.config.fps, (w, h))
            for frame in self.depth_frames:
                d_writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            d_writer.release()
            logging.info(f"  Depth: {self.depth_path}")


class PlayerUnified:
    """
    Unified player with two render modes:
    
    1. Real-time mode (default): Render during inference using env.get_rgbd_image()
       - Contains full info (RGB + depth + all sensor data)
       - Use when you need complete perception data
    
    2. Offline mode: Record states during inference, render after
       - Motion info only (qpos, qvel, time)
       - Faster inference, parallel rendering
       - Use --offline-render flag
    """
    
    def __init__(self, args: Arguments = None):
        # Configure logging to show info messages
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        
        self.args = args or self._parse_args()
        
        # Core components
        self.run = RunLoader(self.args.run, cache=self.args.cache)
        self.task = registry.get(
            self.args.robot or getattr(self.run.task, 'robot', 'g1'),
            tag=self.args.tag or getattr(self.run, 'tag', None)
        )
        self.cfg = self._load_cfg()
        self.env_cfg = self.cfg['environment']
        self.cmder = make_commander(self.args.command, self.env_cfg, self.verbose)
        self.env = self._make_env()
        self.device = alg.get_device(self.args.device)
        self.policy, self.helpers = self._make_policy()
        self.pub = DataStream('Foxglove' if self.args.foxglove else 'PlotJuggler', self.args.dump)
        
        # Robot config
        self.robot_config = self._get_robot_config()
        
        # Video components
        self.video_writer: Optional[VideoWriter] = None
        self.frame_data_list: List[Dict] = []  # For offline mode
    
    @property
    def verbose(self) -> bool:
        return not self.args.quiet
    
    @property
    def workspace_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    @property
    def is_offline_mode(self) -> bool:
        return self.args.render_mode == 'offline'
    
    @staticmethod
    def _parse_args() -> Arguments:
        parser = make_play_argparser()
        
        # Additional arguments
        parser.add_argument('--cache', action='store_true', help='cache run at local directory')
        parser.add_argument('--stochastic', action='store_true', help='stochastic policy')
        parser.add_argument('--num-steps', type=int, default=None,
                            help='number of simulation steps (default: auto from --seconds)')
        
        # Render mode selection
        parser.add_argument('--offline-render', action='store_true',
                            help='offline render mode (render after inference)')
        
        # Video arguments
        parser.add_argument('--record-video', action='store_true', help='enable video recording')
        parser.add_argument('--video-fps', type=int, default=50)
        parser.add_argument('--video-width', type=int, default=640)
        parser.add_argument('--video-height', type=int, default=480)
        parser.add_argument('--video-camera-id', type=int, default=0)
        parser.add_argument('--render-backend', type=str, default='egl', choices=['egl', 'osmesa'])
        parser.add_argument('--save-depth', action='store_true')
        
        parsed = parser.parse_args()
        
        # Build Arguments
        args = Arguments()
        arg_keys = ['run', 'robot', 'tag', 'algorithm', 'algorithm_cfg_path',
                    'quiet', 'seed', 'headless', 'extra_cfg_files', 'overwrite',
                    'overwrite_env', 'sync_cfg', 'device', 'fp16', 'symmetry',
                    'seconds', 'endless', 'command', 'speed', 'dump', 'timeout',
                    'foxglove', 'cache', 'stochastic', 'num_steps']
        for key in arg_keys:
            if hasattr(parsed, key):
                setattr(args, key, getattr(parsed, key))
        
        # Render mode
        args.render_mode = 'offline' if getattr(parsed, 'offline_render', False) else 'realtime'
        
        # Video config
        args.video = VideoConfig(
            enabled=getattr(parsed, 'record_video', False),
            fps=getattr(parsed, 'video_fps', 50),
            width=getattr(parsed, 'video_width', 640),
            height=getattr(parsed, 'video_height', 480),
            camera_id=getattr(parsed, 'video_camera_id', 0),
            render_backend=getattr(parsed, 'render_backend', 'egl'),
            save_depth=getattr(parsed, 'save_depth', False),
        )
        
        return args
    
    def _get_task_base_name(self) -> str:
        """Extract base robot name from task name.
        """
        task_name = getattr(self.task, 'name', 'g1')
        return task_name.split('.')[0]
    
    def _get_model_xml_path(self) -> str:
        """Get MuJoCo model XML path based on robot name."""
        task_base = self._get_task_base_name()
        if task_base in ROBOT_MODEL_PATHS:
            return os.path.join(self.workspace_root, ROBOT_MODEL_PATHS[task_base])
        logging.warning(f"[Warning] Unknown robot '{task_base}', using default")
        return os.path.join(self.workspace_root, 'resources/g1/g1_15dof.xml')
    
    def _get_robot_config(self) -> Dict:
        """Get robot camera configuration."""
        task_base = self._get_task_base_name()
        return ROBOT_CAMERA_CONFIGS.get(task_base, DEFAULT_CAMERA_CONFIG)
    
    def _load_cfg(self) -> Dict:
        try:
            cfg = self.task.load_cfg(self.args.algorithm_cfg_path or self.args.algorithm)
        except FileNotFoundError:
            cfg = getattr(self.run, 'cfg', {'environment': {}, 'architecture': {'actor': ''}})
            self.args.sync_cfg = False
        
        process_curriculum(cfg['environment'], getattr(self.run, 'iteration', 0))
        if self.args.sync_cfg:
            sync_cfg(cfg, getattr(self.run, 'cfg', {}), not self.args.quiet)
        cfg = self.task.update_cfg(cfg, files=self.args.extra_cfg_files)
        overwrite_cfg(cfg, CfgOverwriteOptions(
            overwrite=self.args.overwrite,
            overwrite_env=self.args.overwrite_env
        ))
        
        if self.args.timeout is not None and self.args.timeout > 0:
            cfg['environment']['max_time'] = self.args.timeout
        return cfg
    
    def _make_env(self):
        # Offline mode: render happens after simulation
        # Realtime mode: need render unless headless
        if self.is_offline_mode:
            need_render = False
        else:
            need_render = not self.args.headless
        env = self.task.make_env(self.env_cfg, need_render, self.verbose, seed=self.args.seed)
        env.load_scaling(getattr(self.run, 'scaling_dict', {}))
        return env
    
    def _make_policy(self):
        ob_dim = getattr(self.env, 'ob_dim', 10)
        action_dim = getattr(self.env, 'action_dim', 6)
        
        policy = alg.GeneralActor.make(self.cfg['architecture']['actor'], ob_dim, action_dim).to(self.device)
        
        weight_path = getattr(self.run, 'weight_path', '')
        if weight_path and os.path.exists(weight_path):
            try:
                state_dict = torch.load(weight_path, map_location=self.device)
                policy.load_state_dict(state_dict['actor'])
            except Exception as e:
                logging.warning(f"[Warning] Failed to load weights: {e}")
        
        if self.args.symmetry:
            policy = symmetry_wrapper(policy, self.args.symmetry, self.env, self.verbose)
        if self.args.fp16:
            policy.half()
        
        policy = policy.inference()
        
        # Load helpers
        helpers = {}
        if weight_path and os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location=self.device)
            if 'discriminator' in state_dict:
                discriminator = alg.make_nn(
                    self.cfg['architecture'].get('discriminator', ''),
                    getattr(self.env, 'extended_ob_dim', ob_dim), 1
                ).to(self.device)
                discriminator.load_state_dict(
                    alg.utils.adapt_ddp_state_dict(state_dict['discriminator'], False, depth=0)
                )
                discriminator.eval()
                helpers['discriminator'] = alg.InferenceModule(discriminator)
        
        return policy, helpers
    
    def _init_video(self):
        if not self.args.video.enabled:
            return
        
        run_name = os.path.basename(self.args.run) if self.args.run else 'play'
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        output_dir = os.path.join(os.getcwd(), 'results', date_str)
        self.video_writer = VideoWriter(self.args.video, output_dir, run_name)
        
        mode_str = "offline" if self.is_offline_mode else "realtime"
        logging.info(
            f"[Video] {self.args.video.width}x{self.args.video.height} @ "
            f"{self.args.video.fps}fps, mode={mode_str}"
        )
    
    def _capture_frame_realtime(self, step: int):
        """Real-time capture using env.get_rgbd_image (full info)."""
        if not self.args.video.enabled or step % self.args.video.save_interval != 0:
            return
        
        import cv2
        rgb, depth = self.env.get_rgbd_image(
            width=self.args.video.width, height=self.args.video.height, camera_id=self.args.video.camera_id
        )
        self.video_writer.add_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth)
    
    def _capture_frame_offline(self, step: int):
        """Offline capture: record state only (motion info)."""
        if not self.args.video.enabled or step % self.args.video.save_interval != 0:
            return
        
        frame_end = float('inf') if self.args.video.frame_end == -1 else self.args.video.frame_end
        if not (self.args.video.frame_start <= step <= frame_end):
            return
        
        # Record minimal state data
        self.frame_data_list.append({
            'qpos': self.env.getGeneralizedCoordinate().tolist(),
            'qvel': self.env.getGeneralizedVelocity().tolist(),
            'time': self.env.getSimulationTime(),
            'step': step,
        })
    
    def _render_offline(self):
        """Render frames in parallel after inference."""
        if not self.frame_data_list:
            return
        
        logging.info(f"\n[Video] Starting composition...")
        
        # Get model XML path
        model_xml_path = self._get_model_xml_path()
        logging.info(f"[Offline Render] Processing {len(self.frame_data_list)} frames...")
        logging.info(f"[Offline Render] Model: {model_xml_path}")
        start_time = time.time()
        
        render_cfg = {
            'width': self.args.video.width,
            'height': self.args.video.height,
            'quality': self.args.video.render_quality,
            'cam_distance': self.robot_config['cam_distance'],
            'cam_azimuth': self.args.video.cam_azimuth,
            'cam_elevation': self.args.video.cam_elevation,
            'render_backend': self.args.video.render_backend,
            'lookat_z': self.robot_config['lookat_z'],
            'model_xml': model_xml_path,
        }
        
        for fd in self.frame_data_list:
            fd['cfg'] = render_cfg
        
        num_workers = min(self.args.video.num_workers, max(1, mp.cpu_count() // 2))
        with mp.Pool(processes=num_workers) as pool:
            frames = pool.map(_render_frame_parallel, self.frame_data_list)
        
        for frame in (f for f in frames if f is not None):
            self.video_writer.add_frame(frame)
        
        logging.info(f"[Offline Render] Done in {time.time() - start_time:.2f}s")
    
    def _update_info(self, info: Dict, action: np.ndarray, obs: np.ndarray):
        info.update({
            'Action': action.squeeze(),
            'ActionMean': getattr(self.policy, 'action_mean', np.zeros_like(action)).squeeze(),
            'ActionStd': getattr(self.policy, 'action_std', np.zeros_like(action)).squeeze(),
        })
        
        embedding = getattr(self.policy, 'get_last_embedding', lambda: None)()
        if embedding is not None:
            info['Embedding'] = embedding.squeeze()
        
        discriminator = self.helpers.get('discriminator')
        if discriminator is not None:
            indices = self.run.cfg['ppo'].get('symmetry_obs_slice', slice(None))
            if isinstance(indices, str):
                indices = eval(indices)
            extended_obs = getattr(self.env, 'get_extended_observation', lambda: obs)()
            info['SymmetryLogit'] = discriminator(extended_obs).item()
            
            if hasattr(alg, 'get_pseudo_symmetric_state') and alg.get_pseudo_symmetric_state:
                ps_state = alg.get_pseudo_symmetric_state(
                    getattr(discriminator, 'unwrapped', discriminator),
                    torch.as_tensor(extended_obs, device=self.device),
                    indices=indices,
                ).cpu().numpy().squeeze()
                info['SymmetricObservation'] = ps_state
                info['SymmetryUpdatedLogit'] = discriminator(ps_state).item()
                info['SymmetryDistance'] = np.abs(ps_state - extended_obs).mean()
    
    def _report(self, is_terminated: bool):
        if self.env.num_steps == 0:
            return
        avg_reward = self.env.episode_reward / self.env.num_steps
        status = 'End' if is_terminated else 'Timeout'
        logging.info(f"{status} at {self.env.getSimulationTime():.2f}s, "
                     f"avg reward: {avg_reward:.6f}")
    
    def start_playing_loop(self, num_steps: int = None):
        if self.verbose:
            logging.info(f'\nPlaying {self.run}')
            if self.is_offline_mode:
                logging.info(f"Mode: offline (simulation only, render after)")
            else:
                logging.info(f"Mode: realtime (render during simulation)")
        
        # Determine number of steps
        if num_steps is None:
            if self.args.num_steps is not None:
                num_steps = self.args.num_steps
            else:
                num_steps = int(1e10 if self.args.endless else self.args.seconds / self.env.getControlTimeStep())
        
        if self.verbose:
            logging.info(f"Steps: {num_steps}")
            logging.info(f"\n[Inference] Starting simulation...")
        
        self._init_video()
        
        try:
            self._playing_loop(num_steps)
        except KeyboardInterrupt:
            pass
        
        # Post-processing
        if self.args.video.enabled:
            if self.is_offline_mode:
                self._render_offline()
            if self.video_writer:
                self.video_writer.save()
                logging.info(f"\n[Video] Output path: {self.video_writer.video_path}")
        
        self._report(True)
        sprint.table(self.env.get_reward_dict(), num_cols=2, header='Reward Details')
    
    def _playing_loop(self, num_steps: int):
        des_step_time = self.env.getControlTimeStep() / self.args.speed
        
        # Skip visualizer wait in offline mode
        if not self.args.headless and not self.is_offline_mode:
            self.env.wait_until_visualizer_connected()
        
        obs = self.env.reset()
        self.pub.set_fixed_interval_mode(time.time(), self.env.getSimulationTimeStep())
        start_time = time.time()
        
        # Select capture method based on mode
        capture_frame = self._capture_frame_offline if self.is_offline_mode else self._capture_frame_realtime
        
        for step in range(num_steps):
            try:
                action = self.policy(obs, stochastic=self.args.stochastic)
            except Exception as e:
                logging.error(f"[Policy Error] {e}")
                action = np.zeros(getattr(self.env, 'action_dim', 6))
            
            for substep_info in self.env.substeps(action, skip_last_info=True):
                self.pub.publish(substep_info)
            
            capture_frame(step)
            
            obs, rew, done, timeout, info = self.env.poststep()
            self._update_info(info, action, obs)
            self.pub.publish(info)
            
            if self.cmder:
                cmd_vel = self.cmder.get_cmd_vel()
                if cmd_vel is not None:
                    self.env.setCmdVel(*cmd_vel)
                cmd_h = self.cmder.get_cmd_height(self.env.getCmdHeight())
                if cmd_h is not None:
                    self.env.setCmdHeight(cmd_h)
                cmd_p = self.cmder.get_cmd_pitch(self.env.getCmdPitch())
                if cmd_p is not None:
                    self.env.setCmdPitch(cmd_p)
            
            if done or (self.args.timeout and timeout):
                self._report(done)
                reset_start = time.time()
                obs = self.env.reset()
                getattr(self.policy, 'reset', lambda: None)()
                start_time += time.time() - reset_start
            
            dt = start_time + des_step_time * step - time.time()
            if dt > 0:
                time.sleep(dt)


# Alias for backward compatibility
Player = PlayerUnified


if __name__ == "__main__":
    Player().start_playing_loop()
