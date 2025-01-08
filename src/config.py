from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class Config:
    model_path: Path = Path("models/vosk-model-small-en-us-0.15")
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    noise_threshold: float = 0.1
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))
            
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)