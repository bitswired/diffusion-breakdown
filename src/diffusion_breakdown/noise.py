from dataclasses import dataclass

@dataclass
class Noise:
    timesteps: int
    variance_schedule: int

    def __post_init__(self):

