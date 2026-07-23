from __future__ import annotations

import numpy as np
from OpenGL import GL as gl
from OpenGL.arrays import vbo


def get_sphere_wireframe(
    center: np.ndarray | tuple[float, float, float] | None = None,
    radius: float = 1.0,
    meridians: int = 24,
    parallels: int = 12,
) -> tuple[vbo.VBO, vbo.VBO, int]:
    """Returns VBOs for a wireframe sphere (meridians + latitude rings)."""
    if center is None:
        center = np.zeros(3, dtype=np.float32)
    else:
        center = np.asarray(center, dtype=np.float32).reshape(3)

    meridians = max(4, int(meridians))
    parallels = max(2, int(parallels))
    segments = max(8, meridians * 2)

    vertices: list[np.ndarray] = []
    indices: list[int] = []

    def _add_polyline(points: np.ndarray) -> None:
        base = len(vertices)
        vertices.extend(points)
        for i in range(len(points) - 1):
            indices.extend((base + i, base + i + 1))

    # Meridians (longitude lines).
    theta_samples = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=np.float32)
    for meridian_idx in range(meridians):
        theta = 2.0 * np.pi * meridian_idx / meridians
        phi_samples = np.linspace(0.0, np.pi, segments, dtype=np.float32)
        points = np.stack(
            (
                np.sin(phi_samples) * np.cos(theta),
                np.sin(phi_samples) * np.sin(theta),
                np.cos(phi_samples),
            ),
            axis=1,
            dtype=np.float32,
        )
        _add_polyline(points)

    # Latitude rings (parallels), skip poles to avoid duplicate vertices.
    phi_samples = np.linspace(0.0, np.pi, parallels + 2, dtype=np.float32)[1:-1]
    for phi in phi_samples:
        ring_radius = np.sin(phi)
        z = np.cos(phi)
        points = np.stack(
            (
                ring_radius * np.cos(theta_samples),
                ring_radius * np.sin(theta_samples),
                np.full_like(theta_samples, z),
            ),
            axis=1,
            dtype=np.float32,
        )
        _add_polyline(np.vstack((points, points[:1])))

    vertex_array = radius * np.asarray(vertices, dtype=np.float32) + center
    index_array = np.asarray(indices, dtype=np.uint32)
    return (
        vbo.VBO(vertex_array),
        vbo.VBO(index_array, target=gl.GL_ELEMENT_ARRAY_BUFFER),
        len(index_array),
    )


def get_cube_wireframe():
    """Returns the vertices for a wireframe unit cube."""
    vertices = 0.5 * np.array([
        [-1.0, -1.0, -1.0],  # 0
        [-1.0, -1.0,  1.0],  # 1
        [-1.0,  1.0, -1.0],  # 2
        [-1.0,  1.0,  1.0],  # 3
        [ 1.0, -1.0, -1.0],  # 4
        [ 1.0, -1.0,  1.0],  # 5
        [ 1.0,  1.0, -1.0],  # 6
        [ 1.0,  1.0,  1.0],  # 7
    ], dtype=np.float32)
    vertex_positions = vbo.VBO(vertices)

    indices = np.array([
        [0, 1], [0, 2], [1, 3], [2, 3],  # bottom
        [4, 5], [4, 6], [5, 7], [6, 7],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # sides
    ], dtype=np.uint32).flatten()
    index_positions = vbo.VBO(indices, target=gl.GL_ELEMENT_ARRAY_BUFFER)

    return vertex_positions, index_positions