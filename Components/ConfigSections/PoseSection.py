"""Components/ConfigSections/PoseSection.py: Camera pose configuration section for the config window."""

import json
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

from imgui_bundle import imgui, icons_fontawesome_6 as fa

from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from ICGui.Components.HelpIndicator import help_indicator
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.Components.TensorInput import input_mat4
from ICGui.Controls import InputCallback
from ICGui.State.Volatile import GlobalState, CameraState, TimeState
from ICGui.util.Cameras import argmax_temporal_similarity
from ICGui.util.Enums import Action
from Logging import Logger
from .Section import Section


@dataclass
class PoseSection(Section):
    name: str = f'{fa.ICON_FA_LOCATION_DOT} Pose'
    always_open: bool = False
    default_open: bool = False

    _show_advanced = False
    _matrices_as_text = False
    _last_pose_idx: int = -1
    _recording: bool = False
    _record_fps: int = 30
    _record_output_path: str = field(default_factory=lambda: f'trajectories/gui_recorded_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json')
    _record_keyframes: list[dict] = field(default_factory=list)
    _record_start_time: float = 0.0
    _last_record_time: float = 0.0
    _last_record_c2w: list[list[float]] | None = None

    def __post_init__(self):
        super().__post_init__()

        input_manager = GlobalState().input_manager
        input_manager.register_callback(
            'TOGGLE_GROUND_TRUTH',
            InputCallback(self.toggle_gt, continuous=False, interrupt_animation=True),
            'Toggle Ground Truth',
        )
        input_manager.register_callback(
            'JUMP_TO_CLOSEST_POSE',
            InputCallback(self.jump_to_closest, continuous=False, interrupt_animation=False),
            'Jump to closest dataset pose',
        )
        input_manager.register_callback(
            Action.NEXT_POSE,
            InputCallback(lambda: self.cycle_pose(1), continuous=False, interrupt_animation=False),
            'Next dataset pose',
        )
        input_manager.register_callback(
            Action.PREVIOUS_POSE,
            InputCallback(lambda: self.cycle_pose(-1), continuous=False, interrupt_animation=False),
            'Previous dataset pose',
        )
        input_manager.register_callback(
            Action.NEXT_DATASET_SPLIT,
            InputCallback(lambda: self.cycle_dataset_split(1), continuous=False, interrupt_animation=False),
            'Next dataset split',
        )
        input_manager.register_callback(
            Action.PREVIOUS_DATASET_SPLIT,
            InputCallback(lambda: self.cycle_dataset_split(-1), continuous=False, interrupt_animation=False),
            'Previous dataset split',
        )

    def _render(self):
        self._render_dataset_pose_selector()
        self._render_trajectory_recorder()

        if imgui.button('Jump to closest dataset pose'):
            self.jump_to_closest()

        _, GlobalState().skip_animations = styled_toggle('Skip Animations', GlobalState().skip_animations)
        help_indicator('Skip camera animations when jumping to a new position.')

        _, CameraState().prefer_temporal_similarity = styled_toggle('Snap to closest time first', CameraState().prefer_temporal_similarity)
        help_indicator('Whether to consider timestamp or pose first for jumping to nearest camera (and showing GT).')

        changed, _ = styled_toggle('Show ground truth image', CameraState().render_gt)
        if changed:
            self.toggle_gt()
        help_indicator('Show the ground-truth image if available for the current pose, '
                       'scaled according to focal length / projection center.')

        imgui.begin_disabled(CameraState().render_gt)
        # TODO: timestamp should update according to the snapped-to GT view
        changed, snap_state = styled_toggle('Snap to ground truth', CameraState().render_gt or CameraState().snap_to_gt)
        if changed:
            CameraState().snap_to_gt = snap_state
        imgui.end_disabled()
        help_indicator('Lock the camera to the nearest ground truth pose. You can '
                       'still move the camera, but the view will only update, once '
                       'you\'re closer to another ground truth pose.')

        # Advanced settings (Direct Matrix Input)
        _, self._show_advanced = styled_toggle(
            'Show Advanced Options',
            self._show_advanced,
        )
        if self._show_advanced:
            self._render_advanced()

        if self._recording:
            self._record_frame_if_due()

    def _render_trajectory_recorder(self):
        imgui.separator_text('Trajectory Recording')
        imgui.text(f'Keyframes: {len(self._record_keyframes)}')

        changed, fps = imgui.input_int('Record FPS', self._record_fps)
        if changed:
            self._record_fps = max(1, fps)

        changed, output_path = imgui.input_text('Output Path', self._record_output_path)
        if changed:
            self._record_output_path = output_path

        if not self._recording:
            if imgui.button(f'{fa.ICON_FA_CIRCLE} Start Recording'):
                self._start_recording()
        else:
            if imgui.button(f'{fa.ICON_FA_STOP} Stop & Save'):
                self._stop_and_save_recording()
            imgui.same_line()
            if imgui.button('Add Keyframe Now'):
                self._append_record_keyframe(force=True)

        imgui.same_line()
        if imgui.button('Clear'):
            self._clear_recording()

        help_indicator(
            'Records the current GUI camera pose and intrinsics to JSON. '
            'Replay it with scripts/inference.py --recorded-trajectory <path>.'
        )

    def _start_recording(self):
        self._record_keyframes = []
        self._record_start_time = monotonic()
        self._last_record_time = 0.0
        self._last_record_c2w = None
        self._recording = True
        self._append_record_keyframe(force=True)

    def _stop_and_save_recording(self):
        self._append_record_keyframe(force=True)
        self._recording = False
        self._save_recording()

    def _clear_recording(self):
        self._record_keyframes = []
        self._recording = False
        self._last_record_c2w = None

    def _record_frame_if_due(self):
        now = monotonic()
        min_interval = 1.0 / float(max(1, self._record_fps))
        if now - self._last_record_time >= min_interval:
            self._append_record_keyframe()

    def _append_record_keyframe(self, force: bool = False):
        c2w = GlobalState().input_manager.control_scheme.c2w.astype(float)
        c2w_list = c2w.tolist()
        if not force and self._last_record_c2w == c2w_list:
            return

        camera = CameraState().current_camera
        camera_payload = {
            'width': int(camera.width),
            'height': int(camera.height),
            'near_plane': float(camera.near_plane),
            'far_plane': float(camera.far_plane),
            'background_color': camera.background_color.detach().cpu().tolist(),
        }
        if isinstance(camera, PerspectiveCamera):
            camera_payload.update({
                'focal_x': float(camera.focal_x),
                'focal_y': float(camera.focal_y),
                'center_x': float(camera.center_x),
                'center_y': float(camera.center_y),
            })

        now = monotonic()
        self._record_keyframes.append({
            'time': now - self._record_start_time,
            'timestamp': float(TimeState().timestamp),
            'c2w': c2w_list,
            'camera': camera_payload,
        })
        self._last_record_time = now
        self._last_record_c2w = c2w_list

    def _save_recording(self):
        if not self._record_keyframes:
            Logger.log_warning('No trajectory keyframes recorded.')
            return
        output_path = Path(self._record_output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'format': 'nerficg_gui_trajectory',
            'version': 1,
            'fps': int(self._record_fps),
            'frames': self._record_keyframes,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        Logger.log_info(f'Saved GUI trajectory with {len(self._record_keyframes)} keyframes to {output_path}')

    def _render_dataset_pose_selector(self):
        global_state = GlobalState()
        input_manager = global_state.input_manager
        camera_state = CameraState()
        current_split = camera_state.dataset_split
        splits = global_state.shared.splits
        poses = camera_state.dataset_poses

        # Render Split Selector
        if imgui.begin_combo('Dataset Split', current_split):
            for split in splits:
                changed, val = imgui.selectable(split, split == current_split)
                if changed:
                    pose: View = poses[split][0]
                    camera_state.dataset_split = split
                    camera_state.current_view.frame_idx = pose.frame_idx
                    camera_state.current_view.global_frame_idx = pose.global_frame_idx
                    input_manager.control_scheme.ease_to(to_c2w=pose.c2w_numpy, to_timestamp=pose.timestamp)
                    self._last_pose_idx = 0
            imgui.end_combo()

        # Render Camera Pose Selector
        pose_idx = -1 if input_manager.control_scheme.has_moved else self._last_pose_idx
        if (imgui.begin_combo('##DatasetPoseSelector',
                              'Choose pose from dataset' if pose_idx == -1
                              else f'Pose {self._last_pose_idx}')):
            for i, pose in enumerate(poses[camera_state.dataset_split]):
                if imgui.selectable(f'Pose {i}', False)[0]:
                    input_manager.control_scheme.ease_to(to_c2w=pose.c2w_numpy, to_timestamp=pose.timestamp)
                    camera_state.current_view.frame_idx = pose.frame_idx
                    camera_state.current_view.global_frame_idx = pose.global_frame_idx
                    self._last_pose_idx = i
            imgui.end_combo()

    def _render_advanced(self):
        input_manager = GlobalState().input_manager

        _, self._matrices_as_text = styled_toggle('Show Matrices as Text', self._matrices_as_text)
        help_indicator('Show matrices as a multi-line text input, to support copy & paste. Changes are applied '
                       'on enter/focus loss of the text field. Valid formats include any set of 16 numbers, '
                       'separated by whitespace, commas or square brackets.')

        imgui.spacing()
        changed, c2w = input_mat4(f'{fa.ICON_FA_TABLE_CELLS} Camera to World Transformation', input_manager.control_scheme.c2w,
                                  as_separate=not self._matrices_as_text)
        if changed:
            input_manager.control_scheme.c2w = c2w

        changed, w2c = input_mat4(f'{fa.ICON_FA_TABLE_CELLS} World to Camera Transformation', input_manager.control_scheme.w2c,
                                  as_separate=not self._matrices_as_text)
        if changed:
            input_manager.control_scheme.w2c = w2c

    @staticmethod
    def toggle_gt():
        """Toggle the visibility of the ground truth."""
        gt_idx = CameraState().gt_idx
        CameraState().render_gt = not CameraState().render_gt

        # When toggling off, reset the current camera pose to the ground truth pose
        if gt_idx >= 0:
            pose = CameraState().dataset_poses[CameraState().dataset_split][gt_idx]
            GlobalState().input_manager.control_scheme.c2w = pose.c2w_numpy
            TimeState().time = pose.timestamp

    def jump_to_closest(self):
        """Jump to the closest pose in the dataset."""
        camera = GlobalState().input_manager.control_scheme
        camera_state = CameraState()
        cam_idx = argmax_temporal_similarity(
            camera.c2w,
            TimeState().timestamp,
            [(pose.c2w_numpy, pose.timestamp) for pose in camera_state.dataset_poses[camera_state.dataset_split]],
            prefer_time=camera_state.prefer_temporal_similarity,
        )

        pose: View = camera_state.dataset_poses[camera_state.dataset_split][cam_idx]
        camera.ease_to(to_c2w=pose.c2w_numpy, to_timestamp=pose.timestamp)
        camera_state.current_view.frame_idx = pose.frame_idx
        camera_state.current_view.global_frame_idx = pose.global_frame_idx
        self._last_pose_idx = cam_idx

    def cycle_pose(self, increment: int = 1):
        """Jump to the next pose in the dataset."""
        camera_state = CameraState()
        poses = camera_state.dataset_poses[camera_state.dataset_split]
        if not poses:
            return
        self._last_pose_idx = (self._last_pose_idx + increment) % len(poses)
        pose = poses[self._last_pose_idx]
        camera_state.current_view.frame_idx = pose.frame_idx
        camera_state.current_view.global_frame_idx = pose.global_frame_idx
        GlobalState().input_manager.control_scheme.ease_to(to_c2w=pose.c2w_numpy, to_timestamp=pose.timestamp)

    def cycle_dataset_split(self, increment: int = 1):
        """Switch to the next dataset split."""
        camera_state = CameraState()
        global_state = GlobalState()

        splits = global_state.shared.splits
        if not splits:
            return
        current_idx = splits.index(camera_state.dataset_split)
        new_idx = (current_idx + increment) % len(splits)

        camera_state.dataset_split = splits[new_idx]
        self._last_pose_idx = 0
        if camera_state.dataset_poses[camera_state.dataset_split]:
            pose = camera_state.dataset_poses[camera_state.dataset_split][0]
            global_state.input_manager.control_scheme.ease_to(to_c2w=pose.c2w_numpy, to_timestamp=pose.timestamp)
            camera_state.current_view.frame_idx = pose.frame_idx
            camera_state.current_view.global_frame_idx = pose.global_frame_idx
