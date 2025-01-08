import pyttsx3
import logging
from typing import Optional

class TextToSpeech:
    def __init__(self, voice: Optional[str] = None, rate: int = 150):
        """Initialize TTS engine.
        
        Args:
            voice (str, optional): Voice ID to use. None for default voice.
            rate (int): Speech rate (words per minute). Defaults to 150.
        """
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            
            if voice:
                self.engine.setProperty('voice', voice)
                
            # Get available voices
            voices = self.engine.getProperty('voices')
            if voices:
                logging.info(f"Available voices: {len(voices)}")
                for v in voices:
                    logging.info(f"Voice ID: {v.id}")
                    
        except Exception as e:
            logging.error(f"Error initializing TTS engine: {e}")
            raise

    def speak(self, text: str) -> None:
        """Convert text to speech and play it.
        
        Args:
            text (str): Text to convert to speech
        """
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logging.error(f"Error speaking text: {e}")

    def save_to_file(self, text: str, filename: str) -> None:
        """Save speech to an audio file.
        
        Args:
            text (str): Text to convert to speech
            filename (str): Output audio filename
        """
        try:
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
        except Exception as e:
            logging.error(f"Error saving speech to file: {e}")

    def __del__(self):
        """Clean up TTS engine resources."""
        try:
            self.engine.stop()
        except:
            pass

if __name__ == "__main__":
    # Test TTS
    tts = TextToSpeech()
    
    test_texts = [
        "Hello! Testing text to speech.",
        "This is another test message.",
        "Goodbye!"
    ]
    
    for text in test_texts:
        print(f"Speaking: {text}")
        tts.speak(text)