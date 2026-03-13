import logging

import torch

DEFAULT_IMAGE_SHAPE = (3, 224, 224)
DEFAULT_STATE_DIM = 32
_WARNED_SYNC_DEVICE_TYPES = set()


def get_device(device_name: str, logger: logging.Logger) -> torch.device:
    """Select device from user input, supports indexed format like npu:1/cuda:1."""
    requested_device = device_name.strip().lower()

    if requested_device.startswith("npu"):
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            logger.warning("NPU requested (%s) but is unavailable, fallback to CPU.", device_name)
            return torch.device("cpu")
        if ":" in requested_device:
            try:
                index = int(requested_device.split(":", 1)[1])
                return torch.device(f"npu:{index}")
            except ValueError:
                logger.warning("Invalid NPU device format: %s, fallback to npu:0.", device_name)
        return torch.device("npu:0")

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested (%s) but is unavailable, fallback to CPU.", device_name)
            return torch.device("cpu")
        if ":" in requested_device:
            try:
                index = int(requested_device.split(":", 1)[1])
                return torch.device(f"cuda:{index}")
            except ValueError:
                logger.warning("Invalid CUDA device format: %s, fallback to cuda:0.", device_name)
        return torch.device("cuda")

    if requested_device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS requested but is unavailable, fallback to CPU.")
        return torch.device("cpu")

    if requested_device == "cpu":
        return torch.device("cpu")

    logger.warning("Unknown device '%s', fallback to CPU.", device_name)
    return torch.device("cpu")


def move_to_device(data, device: torch.device):
    """Recursively move tensors in nested dict/list structures to target device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [move_to_device(value, device) for value in data]
    return data


def move_to_device_and_dtype(data, device: torch.device, dtype: torch.dtype):
    """Recursively move tensors and cast float32 tensors to target dtype."""
    if isinstance(data, torch.Tensor):
        data = data.to(device)
        if data.dtype == torch.float32:
            data = data.to(dtype)
        return data
    if isinstance(data, dict):
        return {
            key: move_to_device_and_dtype(value, device, dtype)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [move_to_device_and_dtype(value, device, dtype) for value in data]
    return data


def make_dummy_observation(
    batch_size: int = 1,
    image_shape=DEFAULT_IMAGE_SHAPE,
    state_dim: int = DEFAULT_STATE_DIM,
    task="Pick up the object\n",
):
    """Generate dummy observation data for PI0.5 model."""
    image_tensor_shape = (batch_size, *image_shape)
    dummy_base_image = torch.randint(0, 256, image_tensor_shape, dtype=torch.uint8)
    dummy_left_wrist_image = torch.randint(0, 256, image_tensor_shape, dtype=torch.uint8)
    dummy_right_wrist_image = torch.randint(0, 256, image_tensor_shape, dtype=torch.uint8)
    dummy_state = torch.randn(batch_size, state_dim, dtype=torch.float32)

    return {
        "observation.images.base_0_rgb": dummy_base_image,
        "observation.images.left_wrist_0_rgb": dummy_left_wrist_image,
        "observation.images.right_wrist_0_rgb": dummy_right_wrist_image,
        "observation.state": dummy_state,
        "task": task,
    }


def synchronize(device: torch.device, logger: logging.Logger):
    """Synchronize target device for reliable timing."""
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cpu":
        return
    elif device.type not in _WARNED_SYNC_DEVICE_TYPES:
        logger.warning("No synchronize handler for device type '%s'.", device.type)
        _WARNED_SYNC_DEVICE_TYPES.add(device.type)
