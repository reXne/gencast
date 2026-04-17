from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _normalize(points: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(points, axis=-1, keepdims=True)
    return (points / norms).astype(np.float32)


def _base_icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + 5.0**0.5) / 2.0
    vertices = np.asarray(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int64,
    )
    return _normalize(vertices), faces


def create_icosphere(refinement: int) -> Tuple[np.ndarray, np.ndarray]:
    vertices, faces = _base_icosahedron()
    vertices_list = vertices.tolist()

    for _ in range(refinement):
        midpoint_cache: Dict[Tuple[int, int], int] = {}
        new_faces = []

        def midpoint_index(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key in midpoint_cache:
                return midpoint_cache[key]
            midpoint = (
                np.asarray(vertices_list[a], dtype=np.float32)
                + np.asarray(vertices_list[b], dtype=np.float32)
            ) / 2.0
            midpoint = _normalize(midpoint[None, :])[0]
            vertices_list.append(midpoint.tolist())
            index = len(vertices_list) - 1
            midpoint_cache[key] = index
            return index

        for face in faces:
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            ab = midpoint_index(a, b)
            bc = midpoint_index(b, c)
            ca = midpoint_index(c, a)
            new_faces.extend(
                [
                    (a, ab, ca),
                    (b, bc, ab),
                    (c, ca, bc),
                    (ab, bc, ca),
                ]
            )

        vertices = np.asarray(vertices_list, dtype=np.float32)
        faces = np.asarray(new_faces, dtype=np.int64)
        vertices = _normalize(vertices)
        vertices_list = vertices.tolist()

    return vertices.astype(np.float32), faces.astype(np.int64)

