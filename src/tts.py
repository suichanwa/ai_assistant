import pyttsx3
import logging
from typing import Optional

class TextToSpeech:
    def __init__(self, voice_id: Optional[str] = None, rate: int = 175):
        """Initialize TTS engine.
        
        Args:
            voice_id (str): Voice ID to use (None for default)
            rate (int): Speech rate (default: 175)
        """
        try:
            self.engine = pyttsx3.init()
            
            # List available voices
            voices = self.engine.getProperty('voices')
            logging.info(f"Available voices: {len(voices)}")
            for voice in voices:
                logging.info(f"Voice ID: {voice.id}")
            
            # Set voice if specified
            if voice_id:
                self.engine.setProperty('voice', voice_id)
                
            # Set speech rate
            self.engine.setProperty('rate', rate)
            
        except Exception as e:
            logging.error(f"Error initializing TTS: {e}")
            raise

    def speak(self, text: str) -> None:
        """Convert text to speech and play it.
        
        Args:
            text (str): Text to speak
        """
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logging.error(f"Error speaking text: {e}")

    def __del__(self):
        """Cleanup resources."""
        try:
            self.engine.stop()
        except:
            pass

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test different voices
    tts = TextToSpeech()
    test_texts = [
        "Hello! This is a test of the text-to-speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "How does this voice sound?"
    ]
    
    for text in test_texts:
        print(f"Speaking: {text}")
        tts.speak(text)