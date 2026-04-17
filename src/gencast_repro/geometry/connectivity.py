from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .spherical import (
    cartesian_to_lat_lon,
    grid_lat_lon_to_cartesian,
    wrap_longitude_delta,
)


@dataclass
class GraphConnectivity:
    grid_latitudes: np.ndarray
    grid_longitudes: np.ndarray
    grid_xyz: np.ndarray
    mesh_xyz: np.ndarray
    mesh_latitudes: np.ndarray
    mesh_longitudes: np.ndarray
    grid_to_mesh_senders: np.ndarray
    grid_to_mesh_receivers: np.ndarray
    grid_to_mesh_features: np.ndarray
    mesh_to_grid_senders: np.ndarray
    mesh_to_grid_receivers: np.ndarray
    mesh_to_grid_features: np.ndarray
    mesh_neighbors: np.ndarray


def _edge_features(
    sender_latitudes: np.ndarray,
    sender_longitudes: np.ndarray,
    receiver_latitudes: np.ndarray,
    receiver_longitudes: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    sender_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
) -> np.ndarray:
    delta_lat = receiver_latitudes[receivers] - sender_latitudes[senders]
    delta_lon = wrap_longitude_delta(
        receiver_longitudes[receivers] - sender_longitudes[senders]
    )
    distance = np.linalg.norm(
        receiver_xyz[receivers] - sender_xyz[senders],
        axis=-1,
    )
    features = np.stack(
        [
            sender_latitudes[senders] / 90.0,
            receiver_latitudes[receivers] / 90.0,
            delta_lat / 180.0,
            delta_lon / 180.0,
            distance,
        ],
        axis=-1,
    )
    return features.astype(np.float32)


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for face in faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((u, v))
            edges.add((v, u))
    return np.asarray(sorted(edges), dtype=np.int64)


def _k_hop_neighbors(num_nodes: int, edges: np.ndarray, k_hop: int) -> np.ndarray:
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for sender, receiver in edges.tolist():
        adjacency[sender].append(receiver)

    neighborhoods: List[List[int]] = []
    max_neighbors = 0
    for root in range(num_nodes):
        visited = {root}
        frontier = {root}
        for _ in range(k_hop):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(adjacency[node])
            frontier = next_frontier.difference(visited)
            visited.update(frontier)
            if not frontier:
                break
        ordered = [root] + sorted(node for node in visited if node != root)
        neighborhoods.append(ordered)
        max_neighbors = max(max_neighbors, len(ordered))

    padded = np.full((num_nodes, max_neighbors), -1, dtype=np.int64)
    for index, neighborhood in enumerate(neighborhoods):
        padded[index, : len(neighborhood)] = neighborhood
    return padded


def build_connectivity(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    k_grid_to_mesh: int = 4,
    k_mesh_to_grid: int = 3,
    attention_k_hop: int = 8,
) -> GraphConnectivity:
    grid_xyz = grid_lat_lon_to_cartesian(latitudes, longitudes)
    grid_latitudes, grid_longitudes = np.meshgrid(
        latitudes, longitudes, indexing="ij"
    )
    grid_latitudes = grid_latitudes.reshape(-1).astype(np.float32)
    grid_longitudes = grid_longitudes.reshape(-1).astype(np.float32)

    mesh_latitudes, mesh_longitudes = cartesian_to_lat_lon(mesh_vertices)
    mesh_tree = cKDTree(mesh_vertices)
    grid_tree = cKDTree(grid_xyz)

    _, mesh_indices = mesh_tree.query(grid_xyz, k=k_grid_to_mesh)
    if mesh_indices.ndim == 1:
        mesh_indices = mesh_indices[:, None]
    grid_to_mesh_senders = np.repeat(np.arange(grid_xyz.shape[0]), mesh_indices.shape[1])
    grid_to_mesh_receivers = mesh_indices.reshape(-1).astype(np.int64)
    grid_to_mesh_features = _edge_features(
        grid_latitudes,
        grid_longitudes,
        mesh_latitudes,
        mesh_longitudes,
        grid_to_mesh_senders,
        grid_to_mesh_receivers,
        grid_xyz,
        mesh_vertices,
    )

    _, mesh_receiver_indices = mesh_tree.query(grid_xyz, k=k_mesh_to_grid)
    if mesh_receiver_indices.ndim == 1:
        mesh_receiver_indices = mesh_receiver_indices[:, None]
    mesh_to_grid_senders = mesh_receiver_indices.reshape(-1).astype(np.int64)
    mesh_to_grid_receivers = np.repeat(
        np.arange(grid_xyz.shape[0]), mesh_receiver_indices.shape[1]
    ).astype(np.int64)
    mesh_to_grid_features = _edge_features(
        mesh_latitudes,
        mesh_longitudes,
        grid_latitudes,
        grid_longitudes,
        mesh_to_grid_senders,
        mesh_to_grid_receivers,
        mesh_vertices,
        grid_xyz,
    )

    mesh_edges = _mesh_edges(mesh_faces)
    neighbors = _k_hop_neighbors(
        num_nodes=mesh_vertices.shape[0],
        edges=mesh_edges,
        k_hop=attention_k_hop,
    )

    return GraphConnectivity(
        grid_latitudes=grid_latitudes,
        grid_longitudes=grid_longitudes,
        grid_xyz=grid_xyz.astype(np.float32),
        mesh_xyz=mesh_vertices.astype(np.float32),
        mesh_latitudes=mesh_latitudes.astype(np.float32),
        mesh_longitudes=mesh_longitudes.astype(np.float32),
        grid_to_mesh_senders=grid_to_mesh_senders.astype(np.int64),
        grid_to_mesh_receivers=grid_to_mesh_receivers.astype(np.int64),
        grid_to_mesh_features=grid_to_mesh_features.astype(np.float32),
        mesh_to_grid_senders=mesh_to_grid_senders.astype(np.int64),
        mesh_to_grid_receivers=mesh_to_grid_receivers.astype(np.int64),
        mesh_to_grid_features=mesh_to_grid_features.astype(np.float32),
        mesh_neighbors=neighbors.astype(np.int64),
    )
