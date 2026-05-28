"""util/Cameras.py: Utilities for camera definitions and configurations."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Generic

import numpy as np
import torch

from Cameras.Base import BaseCamera
from Cameras.Equirectangular import EquirectangularCamera
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import BaseDistortion, RadialTangentialDistortion

_T = TypeVar('_T')

@dataclass(frozen=True)
class ConfigurableProperty(Generic[_T]):
    key: str
    name: str
    property_type: type[_T]
    default: _T

    # Numerical properties
    min_value: _T | None = None
    max_value: _T | None = None
    change_speed: _T | None = None

    # Format specifier for number inputs
    format: _T | None = None


@dataclass(frozen=True)
class CameraDefinition:
    name: str
    camera_class: type[BaseCamera]
    configurable_properties: list[ConfigurableProperty]


@dataclass(frozen=True)
class DistortionDefinition:
    name: str
    distortion_class: type[BaseDistortion]
    configurable_properties: list[ConfigurableProperty]


_NEAR_PLANE = ConfigurableProperty(
    key='near_plane',
    name='Near Plane',
    property_type=float,
    default=0.1,
)
_FAR_PLANE = ConfigurableProperty(
    key='far_plane',
    name='Far Plane',
    property_type=float,
    default=100.0,
)
_BACKGROUND_COLOR = ConfigurableProperty(
    key='background_color',
    name='Background Color',
    property_type=torch.Tensor,
    default=torch.zeros(3),
)

_CUBE_SCALE = ConfigurableProperty(
    key='cube_scale',
    name='Cube Scale',
    property_type=float,
    default=1.0,
    min_value=0.0,
    max_value=10.0,
    change_speed=0.005,
)

VIRTUAL_CAMERA_SETTINGS: list[ConfigurableProperty] = [
    _NEAR_PLANE,
    _FAR_PLANE,
    _BACKGROUND_COLOR,
]
CAMERA_TYPES: list[CameraDefinition] = [
    CameraDefinition('Perspective', PerspectiveCamera, []),
    CameraDefinition('Equirectangular', EquirectangularCamera, []),
]

DISTORTION_TYPES: list[DistortionDefinition] = [
    DistortionDefinition('Radial-Tangential Distortion', RadialTangentialDistortion, [
        ConfigurableProperty(key='k1', name='K1', property_type=float, default=0.0,
            min_value=-1.0, max_value=1.0, change_speed=0.005,
        ),
        ConfigurableProperty(key='k2', name='K2', property_type=float, default=0.0,
            min_value=-1.0, max_value=1.0, change_speed=0.005,
        ),
        ConfigurableProperty(key='k3', name='K3', property_type=float, default=0.0,
            min_value=-1.0, max_value=1.0, change_speed=0.005,
        ),
        ConfigurableProperty(key='p1', name='P1', property_type=float, default=0.0,
            min_value=-1.0, max_value=1.0, change_speed=0.005,
        ),
        ConfigurableProperty(key='p2', name='P2', property_type=float, default=0.0,
            min_value=-1.0, max_value=1.0, change_speed=0.005,
        ),
        ConfigurableProperty(key='undistortion_iterations', name='Iterations', property_type=int, default=10,
            min_value=1, max_value=100, change_speed=1,
        ),
        ConfigurableProperty(key='undistortion_eps', name='Undistortion ε', property_type=float, default=1e-9,
            min_value=1e-11, max_value=1.0, change_speed=1e-10, format='%.10f',
        ),
    ]),
]


def get_camera_idx(camera: BaseCamera) -> int:
    """Get the index of the camera type in the CAMERA_TYPES list."""
    for idx, cam_def in enumerate(CAMERA_TYPES):
        if isinstance(camera, cam_def.camera_class):
            return idx
    raise ValueError(f'Camera type {type(camera)} is not yet supported by the GUI. '
                     f'Please add a definition to the source file at: {Path(__file__).absolute()}')


def get_distortion_idx(distortion: BaseDistortion | None) -> int:
    """Get the index of the distortion type in the DISTORTION_TYPES list."""
    if distortion is None:
        return -1
    for idx, dist_def in enumerate(DISTORTION_TYPES):
        if isinstance(distortion, dist_def.distortion_class):
            return idx
    raise ValueError(f'Distortion type {type(distortion)} is not yet supported by the GUI. '
                     f'Please add a definition to the source file at: {Path(__file__).absolute()}')


def calculate_similarity(a: np.typing.NDArray[np.float32], b: np.typing.NDArray[np.float32], /):
    """Calculates the similarity between two view matrices using the Frobenius norm.

    Args:
      a: A numpy array representing the first view matrix.
      b: A numpy array representing the second view matrix.

    Returns:
      A float representing the similarity between the two view matrices, where higher values indicate greater similarity.
    """
    difference = a - b
    frobenius_norm = np.linalg.norm(difference, 'fro')
    similarity_score = 1.0 - frobenius_norm / np.linalg.norm(a, 'fro')
    return similarity_score


def argmax_similarity(target: np.typing.NDArray[np.float32], choices: list[np.typing.NDArray[np.float32]], /) -> int:
    """Finds the index of the element in choices that is most similar to the target.

    Args:
      target: A numpy array representing the target view matrix.
      choices: A list of numpy arrays representing the candidate view matrices.

    Returns:
      The index of the most similar view matrix in choices.
    """
    similarities = [calculate_similarity(target, choice) for choice in choices]
    return np.argmax(similarities).item()


def argmax_temporal_similarity(target_pose: np.typing.NDArray[np.float32], target_t: float,
                               choices: list[tuple[np.typing.NDArray[np.float32], float]],
                               prefer_time: bool = True) -> int:
    """Finds the index of the element in choices that is most similar to the target.

    Args:
      target_pose: A numpy array representing the target view matrix.
      target_t: A float representing the timestamp of the target view matrix.
      choices: A list of numpy arrays representing the candidate view matrices.
      prefer_time: If True, prioritize temporal proximity over pose similarity. If False, prioritize pose.

    Returns:
      The index of the most similar view matrix / timestamp in choices, prioritized according to prefer_time.
    """
    if prefer_time:
        time_differences = np.array([abs(target_t - choice_t) for _, choice_t in choices])
        time_argmin = np.argwhere(time_differences == np.min(time_differences)).flatten()
        choice_poses = np.array([choice_pose for choice_pose, _ in choices])[time_argmin]
        pose_similarities = [calculate_similarity(target_pose, choice) for choice in choice_poses]
        pose_index = np.argmax(pose_similarities).item()
        return time_argmin[pose_index]
    else:
        pose_similarities = np.array([calculate_similarity(target_pose, choice_pose) for choice_pose, _ in choices])
        pose_argmax = np.argwhere(pose_similarities == np.max(pose_similarities)).flatten()
        time_differences = np.array([abs(target_t - choice_t) for _, choice_t in choices])[pose_argmax]
        time_index = np.argmin(time_differences).item()
        return pose_argmax[time_index]


def parse_mat4(tensor_str: str) -> np.ndarray:
    """Parses a string representation of a 4x4 array into a numpy array. Entries are
    expected to be floats separated by whitespace and/or commas, with optional square brackets.
    """
    tensor_str = tensor_str.replace('[', ' ').replace(']', ' ').replace(',', ' ').strip()
    values = [float(x) for x in tensor_str.split() if x]
    if len(values) != 16:
        raise ValueError('Expected 16 values for a 4x4 matrix.')
    return np.array(values, dtype=np.float32).reshape((4, 4))
