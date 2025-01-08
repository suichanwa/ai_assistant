import pyaudio
import wave
import numpy as np
import threading
import logging
from pathlib import Path

class MicrophoneHandler:
    def __init__(self, rate=16000, chunk_size=1024, channels=1):
        """Initialize microphone handler.
        
        Args:
            rate (int): Sample rate
            chunk_size (int): Recording chunk size
            channels (int): Number of audio channels
        """
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.frames = []
        self.record_thread = None

    def start_recording(self):
        """Start recording audio from microphone."""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        self.frames = []
        self.recording = True
        
        def record():
            while self.recording:
                try:
                    data = self.stream.read(self.chunk_size)
                    self.frames.append(data)
                except Exception as e:
                    logging.error(f"Error recording audio: {e}")
                    break
            
        self.record_thread = threading.Thread(target=record)
        self.record_thread.start()

    def stop_recording(self):
        """Stop recording and cleanup resources."""
        self.recording = False
        if self.record_thread:
            self.record_thread.join()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def save_recording(self, filename="output.wav"):
        """Save recorded audio to WAV file.
        
        Args:
            filename (str): Output WAV file path
        """
        if not self.frames:
            raise ValueError("No audio data recorded")
            
        wf = wave.open(str(filename), 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def get_audio(self) -> bytes:
        """Get the latest audio data.
        
        Returns:
            bytes: Latest recorded audio data or None if no new data
        """
        if not self.recording:
            return None
            
        if len(self.frames) > 0:
            # Get and clear the current frames
            data = b''.join(self.frames)
            self.frames.clear()
            return data
        return None 

    def __del__(self):
        """Cleanup resources on deletion."""
        if self.recording:
            self.stop_recording()
        if self.audio:
            self.audio.terminate()

if __name__ == "__main__":
    import time
    
    # Test recording
    mic = MicrophoneHandler()
    print("Recording for 5 seconds...")
    mic.start_recording()
    time.sleep(5)
    mic.stop_recording()
    mic.save_recording()
    print("Recording saved to output.wav")