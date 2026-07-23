"""ModelViewer/Overlays/EnvScopeSphere.py: Wireframe sphere for DR env-scope placement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import OpenGL.GL as gl
import yaml
from imgui_bundle import imgui
from OpenGL.arrays.vbo import VBO
from OpenGL.GL.shaders import ShaderProgram

import Framework
from .Base import BaseOverlay
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from ICGui.ModelViewer.Shaders import get_line_shader
from ICGui.ModelViewer.Shaders.ContextManagers import (
    bind_shader_program,
    bind_textures,
    bind_vertex_attributes,
    bind_vertex_buffer,
)
from ICGui.ModelViewer.Shaders.Geometry import get_sphere_wireframe
from ICGui.ModelViewer.Shaders.utils import fill_uniforms
from ICGui.State.Volatile import GlobalState
from Logging import Logger


class EnvScopeSphere(BaseOverlay):
    """Wireframe sphere matching TRAINING.DEFERRED_REFLECTION_SCHEDULE env scope."""

    def __init__(self) -> None:
        super().__init__()
        self._wireframe_shader_program: ShaderProgram = get_line_shader()
        self._sphere_vao_location: int = gl.glGenVertexArrays(1)

        self._sphere_vertices_vbo: VBO | None = None
        self._sphere_indices_vbo: VBO | None = None
        self._index_count: int = 0

        self._center = np.zeros(3, dtype=np.float32)
        self._radius = 1.0
        self._meridians = 24
        self._parallels = 12
        self._line_width = 0.004
        self._color = np.array([0.1, 0.85, 0.95, 0.95], dtype=np.float32)
        self._geometry_key: tuple[float, ...] | None = None
        self._auto_sync_from_config = True
        self._config_mtime: float | None = None
        self._config_path: Path | None = None
        self._config_use_env_scope = False
        self._config_has_scope_fields = False

        self._uniform_locations: dict[str, tuple[int, str, tuple]] = {
            'view': (gl.glGetUniformLocation(self._wireframe_shader_program, 'viewMatrix'), 'mat4', (True,)),
            'projection': (
                gl.glGetUniformLocation(self._wireframe_shader_program, 'projectionMatrix'),
                'mat4',
                (True,),
            ),
            'viewportSize': (gl.glGetUniformLocation(self._wireframe_shader_program, 'viewportSize'), 'vec2', tuple()),
            'near': (gl.glGetUniformLocation(self._wireframe_shader_program, 'near'), 'float', tuple()),
            'far': (gl.glGetUniformLocation(self._wireframe_shader_program, 'far'), 'float', tuple()),
            'lineWidth': (gl.glGetUniformLocation(self._wireframe_shader_program, 'lineWidth'), 'float', tuple()),
            'color': (gl.glGetUniformLocation(self._wireframe_shader_program, 'color'), 'vec4', tuple()),
        }

        self._load_from_training_config()

    @property
    def name(self) -> str:
        return 'Env Scope Sphere'

    @property
    def help(self) -> str | None:
        return (
            'Wireframe sphere for deferred-reflection env scope. '
            'Reads ENV_SCOPE_CENTER / ENV_SCOPE_RADIUS from the training yaml '
            '(independent of USE_ENV_SCOPE). Auto-syncs when the config file changes.'
        )

    @staticmethod
    def _training_config_path() -> Path | None:
        launch_path = GlobalState().launch_config.training_config_path
        if launch_path is not None and launch_path.is_file():
            return launch_path
        framework_path = getattr(Framework.config, 'path', None)
        if framework_path is not None and Path(framework_path).is_file():
            return Path(framework_path)
        return None

    @staticmethod
    def _parse_env_scope_schedule(schedule: dict[str, Any] | None) -> dict[str, Any] | None:
        if not schedule:
            return None
        has_center = 'ENV_SCOPE_CENTER' in schedule
        has_radius = 'ENV_SCOPE_RADIUS' in schedule
        if not has_center and not has_radius:
            return None
        center_raw = schedule.get('ENV_SCOPE_CENTER', [0.0, 0.0, 0.0])
        return {
            'center': np.array([float(c) for c in center_raw], dtype=np.float32),
            'radius': float(schedule.get('ENV_SCOPE_RADIUS', 0.0)),
            'use_env_scope': bool(schedule.get('USE_ENV_SCOPE', False)),
            'has_radius': has_radius,
        }

    @classmethod
    def _read_env_scope_from_yaml(cls, config_path: Path) -> dict[str, Any] | None:
        try:
            with config_path.open('r', encoding='utf-8') as handle:
                yaml_dict = yaml.safe_load(handle) or {}
        except OSError as exc:
            Logger.log_warning(f'Env scope overlay could not read config file {config_path}: {exc}')
            return None

        training = yaml_dict.get('TRAINING', {})
        if not isinstance(training, dict):
            return None
        schedule = training.get('DEFERRED_REFLECTION_SCHEDULE', {})
        if not isinstance(schedule, dict):
            return None
        return cls._parse_env_scope_schedule(schedule)

    @classmethod
    def _read_env_scope_from_framework(cls) -> dict[str, Any] | None:
        try:
            schedule = Framework.config.TRAINING.DEFERRED_REFLECTION_SCHEDULE
        except AttributeError:
            return None
        if hasattr(schedule, 'toDict'):
            schedule_dict = schedule.toDict()
        elif isinstance(schedule, dict):
            schedule_dict = schedule
        else:
            schedule_dict = {
                key: getattr(schedule, key)
                for key in ('USE_ENV_SCOPE', 'ENV_SCOPE_CENTER', 'ENV_SCOPE_RADIUS')
                if hasattr(schedule, key)
            }
        return cls._parse_env_scope_schedule(schedule_dict)

    @classmethod
    def _read_training_env_scope(cls) -> dict[str, Any] | None:
        config_path = cls._training_config_path()
        if config_path is not None:
            scope = cls._read_env_scope_from_yaml(config_path)
            if scope is not None:
                return scope
        return cls._read_env_scope_from_framework()

    def _apply_scope(self, scope: dict[str, Any]) -> None:
        self._center = scope['center'].astype(np.float32, copy=True)
        self._radius = max(1e-6, float(scope['radius']))
        self._config_use_env_scope = bool(scope['use_env_scope'])
        self._config_has_scope_fields = True
        self._geometry_key = None

    def _load_from_training_config(self) -> bool:
        scope = self._read_training_env_scope()
        if scope is None:
            self._config_has_scope_fields = False
            self._config_use_env_scope = False
            return False
        if float(scope['radius']) <= 0.0:
            self._config_has_scope_fields = True
            self._config_use_env_scope = bool(scope['use_env_scope'])
            self._center = scope['center'].astype(np.float32, copy=True)
            return False
        self._apply_scope(scope)
        config_path = self._training_config_path()
        if config_path is not None:
            self._config_path = config_path
            try:
                self._config_mtime = config_path.stat().st_mtime
            except OSError:
                self._config_mtime = None
        return True

    def _maybe_auto_sync_from_config(self) -> None:
        if not self._auto_sync_from_config:
            return
        config_path = self._training_config_path()
        if config_path is None:
            return
        try:
            mtime = config_path.stat().st_mtime
        except OSError:
            return
        if self._config_path == config_path and self._config_mtime == mtime:
            return
        self._config_path = config_path
        self._config_mtime = mtime
        self._load_from_training_config()

    def _geometry_signature(self) -> tuple[float, ...]:
        return (
            float(self._radius),
            float(self._meridians),
            float(self._parallels),
            float(self._center[0]),
            float(self._center[1]),
            float(self._center[2]),
        )

    def _rebuild_geometry(self) -> None:
        signature = self._geometry_signature()
        if signature == self._geometry_key:
            return
        self._sphere_vertices_vbo, self._sphere_indices_vbo, self._index_count = get_sphere_wireframe(
            center=self._center,
            radius=self._radius,
            meridians=self._meridians,
            parallels=self._parallels,
        )
        self._geometry_key = signature

    def render_options(self) -> None:
        self._maybe_auto_sync_from_config()

        config_path = self._training_config_path()
        if config_path is not None:
            imgui.text_disabled(f'Config: {config_path}')
        else:
            imgui.text_disabled('Config: no training yaml path available')

        scope = self._read_training_env_scope()
        if scope is not None:
            cfg_center = scope['center']
            cfg_radius = float(scope['radius'])
            imgui.text_disabled(
                f'File values: center=({cfg_center[0]:.3f}, {cfg_center[1]:.3f}, {cfg_center[2]:.3f}), '
                f'radius={cfg_radius:.3f}'
            )
            scope_state = 'enabled' if scope['use_env_scope'] else 'disabled'
            imgui.text_disabled(f'Training USE_ENV_SCOPE: {scope_state}')
            if cfg_radius <= 0.0:
                imgui.text_disabled('File radius is 0 — training scope is off; adjust radius below or in yaml.')
        else:
            imgui.text_disabled(
                'File has no ENV_SCOPE_CENTER / ENV_SCOPE_RADIUS under TRAINING.DEFERRED_REFLECTION_SCHEDULE. '
                'Display defaults to center=(0, 0, 0), radius=1.'
            )

        imgui.text_disabled(
            f'Displaying: center=({self._center[0]:.3f}, {self._center[1]:.3f}, {self._center[2]:.3f}), '
            f'radius={self._radius:.3f}'
        )

        _, self._auto_sync_from_config = imgui.checkbox(
            f'Auto-sync from config file##{self.name}',
            self._auto_sync_from_config,
        )
        if imgui.button(f'Reload from config##{self.name}'):
            self._load_from_training_config()

        changed, center = imgui.drag_float3(f'Center##{self.name}', self._center.tolist(), 0.01)
        if changed:
            self._center = np.array(center, dtype=np.float32)
            self._geometry_key = None

        min_radius, max_radius = 1e-6, 1e4
        changed, radius = imgui.drag_float(
            f'Radius##{self.name}',
            self._radius,
            0.01,
            min_radius,
            max_radius,
            '%.4f',
        )
        if changed:
            self._radius = max(min_radius, min(radius, max_radius))
            self._geometry_key = None

        changed, meridians = imgui.slider_int(f'Meridians##{self.name}', self._meridians, 4, 64)
        if changed:
            self._meridians = meridians
            self._geometry_key = None

        changed, parallels = imgui.slider_int(f'Parallels##{self.name}', self._parallels, 2, 32)
        if changed:
            self._parallels = parallels
            self._geometry_key = None

        min_line_width, max_line_width = 1e-9, 0.1
        changed, line_width = imgui.drag_float(
            f'Line Width##{self.name}',
            self._line_width,
            0.000025,
            min_line_width,
            max_line_width,
            '%.5f',
        )
        if changed:
            self._line_width = max(min_line_width, min(line_width, max_line_width))

        changed, color = imgui.color_edit4(f'Color##{self.name}', self._color.tolist())
        if changed:
            self._color = np.array(color, dtype=np.float32)

    def render(self, view: View, extra_params: dict[str, Any]) -> None:
        camera = view.camera
        if not isinstance(camera, PerspectiveCamera):
            Logger.log_warning(f'Env scope sphere requires a PerspectiveCamera, got {type(camera)}')
            return

        self._rebuild_geometry()
        if self._sphere_vertices_vbo is None or self._sphere_indices_vbo is None or self._index_count <= 0:
            return

        model_transform = GlobalState().input_manager.control_scheme.model_transform
        uniform_values = {
            'view': (model_transform @ view.w2c_numpy @ model_transform.T).astype(np.float32).copy(order='C'),
            'projection': camera.get_projection_matrix(invert_z=True).cpu().numpy().astype(np.float32).copy(order='C'),
            'viewportSize': np.array(extra_params['viewportSize'], dtype=np.float32),
            'near': camera.near_plane,
            'far': camera.far_plane,
            'lineWidth': self._line_width,
            'color': self._color,
        }

        with (
            bind_shader_program(self._wireframe_shader_program),
            bind_textures(extra_params['modelTextureLocations']['depth'], extra_params['modelTextureLocations']['alpha']),
            bind_vertex_buffer(self._sphere_vao_location, self._sphere_vertices_vbo, self._sphere_indices_vbo),
            bind_vertex_attributes(3, gl.GL_FLOAT, 0),
        ):
            fill_uniforms(self._uniform_locations, uniform_values)
            gl.glDrawElements(gl.GL_LINES, self._index_count, gl.GL_UNSIGNED_INT, None)
