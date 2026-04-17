from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..constants import DEFAULT_VARIABLE_WEIGHTS


@dataclass(frozen=True)
class ChannelMetadata:
    name: str
    variable: str
    level: Optional[int]
    kind: str
    is_precipitation: bool = False


@dataclass(frozen=True)
class ChannelLayout:
    surface_variables: Tuple[str, ...]
    atmospheric_variables: Tuple[str, ...]
    pressure_levels: Tuple[int, ...]
    forcing_variables: Tuple[str, ...]
    static_variables: Tuple[str, ...]
    precipitation_variables: Tuple[str, ...]
    input_steps: int = 2

    @classmethod
    def from_config(cls, config: object, input_steps: int = 2) -> "ChannelLayout":
        return cls(
            surface_variables=tuple(config.surface_variables),
            atmospheric_variables=tuple(config.atmospheric_variables),
            pressure_levels=tuple(config.pressure_levels),
            forcing_variables=tuple(config.forcing_variables),
            static_variables=tuple(config.static_variables),
            precipitation_variables=tuple(config.precipitation_variables),
            input_steps=int(input_steps),
        )

    @property
    def state_channels(self) -> Tuple[ChannelMetadata, ...]:
        channels: List[ChannelMetadata] = []
        precipitation = set(self.precipitation_variables)
        for variable in self.surface_variables:
            channels.append(
                ChannelMetadata(
                    name=variable,
                    variable=variable,
                    level=None,
                    kind="surface",
                    is_precipitation=variable in precipitation,
                )
            )
        for variable in self.atmospheric_variables:
            for level in self.pressure_levels:
                channels.append(
                    ChannelMetadata(
                        name="%s@%s" % (variable, level),
                        variable=variable,
                        level=level,
                        kind="atmospheric",
                        is_precipitation=variable in precipitation,
                    )
                )
        return tuple(channels)

    @property
    def forcing_channels(self) -> Tuple[ChannelMetadata, ...]:
        channels: List[ChannelMetadata] = []
        for variable in self.forcing_variables:
            channels.append(ChannelMetadata(variable, variable, None, "forcing"))
        for variable in self.static_variables:
            channels.append(ChannelMetadata(variable, variable, None, "static"))
        return tuple(channels)

    @property
    def num_state_channels(self) -> int:
        return len(self.state_channels)

    @property
    def num_forcing_channels(self) -> int:
        return len(self.forcing_channels)

    @property
    def state_channel_names(self) -> Tuple[str, ...]:
        return tuple(channel.name for channel in self.state_channels)

    @property
    def forcing_channel_names(self) -> Tuple[str, ...]:
        return tuple(channel.name for channel in self.forcing_channels)

    def precipitation_channel_mask(self) -> np.ndarray:
        return np.asarray(
            [channel.is_precipitation for channel in self.state_channels],
            dtype=bool,
        )

    def variable_weight_vector(self) -> np.ndarray:
        weights = []
        for channel in self.state_channels:
            weights.append(DEFAULT_VARIABLE_WEIGHTS.get(channel.variable, 1.0))
        return np.asarray(weights, dtype=np.float32)
