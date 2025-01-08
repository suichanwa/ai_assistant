import pytest
from pathlib import Path
from src.main import AIAssistant
from src.stt import SpeechToText
from src.mic import MicrophoneHandler

@pytest.fixture
def assistant():
    return AIAssistant()

@pytest.fixture
def test_wav():
    return Path("tests/data/test.wav")

def test_assistant_initialization(assistant):
    assert assistant.mic is not None
    assert assistant.stt is not None
    assert assistant.nlp is not None
    assert assistant.tts is not None

def test_transcribe_file(assistant, test_wav, tmp_path):
    output_file = tmp_path / "transcription.txt"
    assistant.transcribe_file(test_wav, output_file)
    assert output_file.exists()
    assert output_file.read_text().strip() != ""

def test_invalid_audio_format():
    with pytest.raises(ValueError):
        stt = SpeechToText()
        stt.transcribe_wav("tests/data/invalid.mp3")

def test_microphone_recording(tmp_path):
    mic = MicrophoneHandler()
    output_file = tmp_path / "recording.wav"
    
    try:
        mic.start_recording()
        time.sleep(1)  # Record for 1 second
        mic.stop_recording()
        mic.save_recording(str(output_file))
        
        assert output_file.exists()
        assert output_file.stat().st_size > 0
    finally:
        mic.stop_recording()  # Ensure cleanup