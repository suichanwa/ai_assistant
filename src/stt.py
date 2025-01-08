import json
import os
import wave
import numpy as np
import logging
from vosk import Model, KaldiRecognizer
from pathlib import Path
from typing import Optional, Callable

class SpeechToText:
    def __init__(self, model_path="models/vosk-model-small-en-us-0.15", sample_rate=16000):
        """Initialize STT with Vosk model.
        
        Args:
            model_path (str): Path to Vosk model directory
            sample_rate (int): Audio sample rate in Hz
        """
        try:
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, sample_rate)
            self.sample_rate = sample_rate
            logger.info(f"Initialized STT with model: {model_path}")
        except Exception as e:
            logger.error(f"Error loading Vosk model: {e}")
            raise

    def transcribe_wav(self, wav_file: str) -> str:
        """Transcribe audio from a WAV file.
        
        Args:
            wav_file (str): Path to WAV file
            
        Returns:
            str: Transcribed text
        """
        if not os.path.exists(wav_file):
            raise ValueError(f"File not found: {wav_file}")
            
        try:
            wf = wave.open(wav_file, "rb")
            self._validate_audio(wf)
            
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    if result["text"]:
                        results.append(result["text"])
            
            # Get final bits
            final = json.loads(self.recognizer.FinalResult())
            if final["text"]:
                results.append(final["text"])
                    
            return " ".join(results)
        except wave.Error:
            raise ValueError("Invalid WAV file format")
        finally:
            wf.close()

    def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe raw audio data.
        
        Args:
            audio_data (bytes): Raw audio data to transcribe
            
        Returns:
            str: Transcribed text
        """
        try:
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                return result.get("text", "")
            return ""
        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")
            return ""

    def transcribe_microphone(self, callback: Callable[[str], None]) -> None:
        """Transcribe audio from microphone in real-time.
        
        Args:
            callback (callable): Function to call with transcribed text
        """
        import pyaudio
        
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=self.sample_rate,
                       input=True,
                       frames_per_buffer=8000)
        
        logger.info("Starting microphone transcription")
        stream.start_stream()
        
        try:
            while True:
                data = stream.read(4000)
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result["text"]
                    if text:
                        callback(text)
        except KeyboardInterrupt:
            logger.info("Stopping microphone transcription")
        except Exception as e:
            logger.error(f"Error in microphone transcription: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def _validate_audio(self, wf: wave.Wave_read) -> None:
        """Validate WAV file format.
        
        Args:
            wf (wave.Wave_read): Wave file object
            
        Raises:
            ValueError: If audio format is invalid
        """
        if wf.getsampwidth() != 2:
            raise ValueError("Audio must be 16-bit PCM WAV")
        if wf.getnchannels() != 1:
            raise ValueError("Audio must be mono, not stereo")
        if wf.getframerate() != self.sample_rate:
            raise ValueError(f"Audio must have {self.sample_rate}Hz sample rate")

def test_microphone():
    """Test microphone transcription."""
    def print_callback(text: str) -> None:
        print(f"Recognized: {text}")
        
    stt = SpeechToText()
    print("Listening... Press Ctrl+C to stop")
    stt.transcribe_microphone(print_callback)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription.log'),
        logging.StreamHandler()
    ]
)

def preprocess_audio(self, audio_data: bytes) -> bytes:
    """Apply audio preprocessing.
    
    Args:
        audio_data (bytes): Raw audio data
    Returns:
        bytes: Processed audio data
    """
    # Convert to numpy array
    audio = np.frombuffer(audio_data, dtype=np.int16)
    
    # Normalize
    normalized = audio / np.max(np.abs(audio))
    
    # Noise reduction
    noise_threshold = 0.1
    normalized[np.abs(normalized) < noise_threshold] = 0
    
    return (normalized * 32767).astype(np.int16).tobytes()

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Speech-to-Text Transcription")
    parser.add_argument("--file", type=str, help="WAV file to transcribe")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    args = parser.parse_args()
    
    stt = SpeechToText()
    
    if args.file:
        text = stt.transcribe_wav(args.file)
        print(f"Transcribed text: {text}")
    elif args.mic:
        test_microphone()
    else:
        parser.print_help()