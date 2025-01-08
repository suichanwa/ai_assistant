import argparse
import logging
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
# Using relative imports
from .mic import MicrophoneHandler 
from .stt import SpeechToText
from .nlp import NLPHandler
from .tts import TextToSpeech