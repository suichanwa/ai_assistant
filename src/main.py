import argparse
import logging
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
# Use relative imports
from .mic import MicrophoneHandler 
from .stt import SpeechToText
from .nlp import NLPHandler
from .tts import TextToSpeech

def create_cli():
    """Create command line interface parser."""
    parser = argparse.ArgumentParser(description="AI Assistant with speech recognition")
    parser.add_argument('--mode', choices=['interactive', 'file'], default='interactive',
                       help='Run in interactive or file transcription mode')
    parser.add_argument('--input', type=Path, help='Input WAV file to transcribe')
    parser.add_argument('--output', type=Path, help='Output text file for transcription')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--model', default='models/vosk-model-small-en-us-0.15',
                       help='Path to speech recognition model')
    return parser

class AIAssistant:
    def __init__(self):
        """Initialize all components of the AI assistant."""
        try:
            self.mic = MicrophoneHandler()
            self.stt = SpeechToText()
            self.nlp = NLPHandler()
            self.tts = TextToSpeech()
            self.console = Console()
            logging.info("AI Assistant initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing AI Assistant: {e}")
            raise

    def transcribe_file(self, input_path: Path, output_path: Path):
        """Transcribe an audio file with progress bar."""
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Transcribing...", total=None)
            try:
                text = self.stt.transcribe_wav(str(input_path))
                progress.update(task, completed=100)
                
                if output_path:
                    output_path.write_text(text)
                    self.console.print(f"[green]Transcription saved to {output_path}[/]")
                else:
                    self.console.print(f"[yellow]Transcription:[/] {text}")
            except Exception as e:
                self.console.print(f"[red]Error transcribing file: {e}[/]")
                raise

    def run(self):
        """Run interactive mode."""
        self.console.print("[blue]Starting interactive mode... Press Ctrl+C to exit[/]")
        try:
            self.mic.start_recording()
            while True:
                # Process microphone input in real-time
                audio_data = self.mic.get_audio()
                if audio_data:
                    text = self.stt.transcribe_audio(audio_data)
                    if text:
                        self.console.print(f"[green]You:[/] {text}")
                        response = self.nlp.process_input(text)
                        self.console.print(f"[blue]AI:[/] {response}")
                        self.tts.speak(response)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Stopping recording...[/]")
        finally:
            self.mic.stop_recording()

def main():
    """Main entry point."""
    parser = create_cli()
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('assistant.log'),
            logging.StreamHandler()
        ]
    )

    try:
        assistant = AIAssistant()
        if args.mode == 'file':
            if not args.input:
                raise ValueError("Input file required for file mode")
            assistant.transcribe_file(args.input, args.output)
        else:
            assistant.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())