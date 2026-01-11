# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
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

"""
在Ascend 310P上进行Spirit-v1.5推理，并与mozRobot进行对接的示例代码
"""

import os
import sys
import time

import cv2
import numpy as np
from scipy.interpolate import interp1d
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
torch.npu.set_compile_mode(jit_compile=False)

# 把项目根目录加入 Python 的搜索路径
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)
from mozrobot import MOZ1Robot, MOZ1RobotConfig
from modeling_spirit_vla import SpiritVLAPolicy

MODEL_PATH = "/home/huawei/spiritVLA/model/"        # 根据实际模型（.config&.safetensors)路径修改
REALSENSE_SERIALS = "318122301415, 230322273678, 230322271083"  # 根据机器人实际连接realsense相机序列号修改

# ----------------------频率转换函数（30Hz→120Hz，线性插值） ----------------------
def resample_action_data(raw_data, raw_freq, target_freq, interp_kind="linear"):
    """
    机器人动作数据频率插值（线性插值保证动作平滑）
    """
    # 计算目标动作点数（120Hz是30Hz的4倍，点数≈原始×4）
    raw_points = raw_data.shape[0]
    total_time = raw_points / raw_freq  # 总时长不变
    target_points = int(total_time * target_freq)

    # 构建时间轴（确保插值后时间对齐，无动作加速/减速）
    raw_time = np.linspace(0, total_time, raw_points, endpoint=False)
    target_time = np.linspace(0, total_time, target_points, endpoint=False)

    # 对每个维度单独插值（避免维度干扰）
    resampled_data = np.zeros((target_points, raw_data.shape[1]))
    for dim in range(raw_data.shape[1]):
        interp_func = interp1d(
            x=raw_time,
            y=raw_data[:, dim],
            kind=interp_kind,
            bounds_error=False,
            fill_value="extrapolate"
        )
        resampled_data[:, dim] = interp_func(target_time)

    return resampled_data, target_time

def main():
    dtype = torch.float16
    device = torch.device("npu")

    # 1. 加载模型Spirit VLA 1.5
    policy = SpiritVLAPolicy.from_pretrained(MODEL_PATH, local_files_only=True).to(device="npu", dtype=dtype)

    # 2. 创建机器人配置
    config = MOZ1RobotConfig(
        realsense_serials=REALSENSE_SERIALS,
        structure="wholebody",
        robot_control_hz=120
    )

    # 3. 创建机器人实例
    robot = MOZ1Robot(config)

    # 发送控制命令频率
    dt = 1.0 / robot.control_hz

    # 插帧比例
    raw_freq = 30
    target_freq = robot.control_hz

    try:
        # 4. 连接机器人
        robot.connect()

        # 5. 启用外部控制模式（必需）
        robot.enable_external_following_mode()

        while True:
            # 6. 获取机器人当前状态
            obs = robot.capture_observation()
            temp_state = np.concatenate([obs["leftarm_state_cart_pose"], obs["leftarm_gripper_state_pos"], obs["rightarm_state_cart_pos"], obs["rightarm_gripper_state_pos"]], axis=0)
            robot_state = torch.from_numpy(temp_state).to(dtype=dtype, device=device).unsqueeze(0)

            # 数据处理(gbr2rgb, normalize)
            obs["cam_high"] = cv2.cvtColor(obs["cam_high"], cv2.COLOR_BGR2RGB)
            obs["cam_high"] = obs["cam_high"].astype(np.float32) / 255.0

            obs["cam_left_wrist"] = cv2.cvtColor(obs["cam_left_wrist"], cv2.COLOR_BGR2RGB)
            obs["cam_left_wrist"] = obs["cam_left_wrist"].astype(np.float32) / 255.0

            obs["cam_right_wrist"] = cv2.cvtColor(obs["cam_right_wrist"], cv2.COLOR_BGR2RGB)
            obs["cam_right_wrist"] = obs["cam_right_wrist"].astype(np.float32) / 255.0

            image_top = torch.from_numpy(obs["cam_high"]).permute(2, 0, 1).to(dtype=dtype, device=device).unsqueeze(0)
            image_left = torch.from_numpy(obs["cam_left_wrist"]).permute(2, 0, 1).to(dtype=dtype, device=device).unsqueeze(0)
            image_right = torch.from_numpy(obs["cam_right_wrist"]).permute(2, 0, 1).to(dtype=dtype, device=device).unsqueeze(0)

            # 7. 根据输入进行推理
            with torch.inference_mode():
                policy.reset()
                observation = {
                    "observation.images.cam_high": image_top,
                    "observation.images.cam_left_wrist": image_left,
                    "observation.images.cam_right_wrist": image_right,
                    "observation.state":robot_state,
                    "task": ["Insert Flowers."],
                    "robot_type": ["moz1"],
                }
                t1 = time.perf_counter()
                action_pi= policy.select_action(observation)
                t2 = time.perf_counter()
                print(f"Infer time: {(t2 - t1)*1000} ms")
                print("send action to the robot server")


            action_list = action_pi.cpu().numpy()
            action_list = action_list[0]
            # 8.将推理结果插值后发送给moz机器人
            resampled_data, _ = resample_action_data(
                raw_data=action_list[0:30],
                raw_freq=raw_freq,
                target_freq=target_freq,
                interp_kind="linear"  # 线性插值适合机器人动作，无异常极值
            )
            i = 0
            for action in resampled_data:
                # 手臂末端笛卡尔坐标 [x, y, z, rx, ry, rz]
                act = {
                        "leftarm_cmd_cart_pos": np.asarray(action[0:6], dtype=np.float32).tolist(),
                        "leftarm_gripper_cmd_pos": np.asarray(action[6:7], dtype=np.float32).tolist(),
                        "rightarm_cmd_joint_cart": np.asarray(action[7:13], dtype=np.float32).tolist(),
                        "rightarm_cmd_cart_pos": np.asarray(action[13:14], dtype=np.float32).tolist(),
                    }
                robot.send_action(act)
                i += 1
                time.sleep(dt)

    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 10. 断开连接
        robot.disconnect()

if __name__ == "__main__":
    main()
