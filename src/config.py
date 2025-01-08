from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class Config:
    model_path: Path = Path("models/vosk-model-small-en-us-0.15")
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)