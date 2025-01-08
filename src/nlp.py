from transformers import pipeline
import logging

class NLPHandler:
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        """Initialize NLP handler with a conversational AI model.
        
        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        try:
            self.conversational = pipeline(
                "conversational",
                model=model_name,
                device=-1  # Use CPU
            )
            self.conversation_history = []
        except Exception as e:
            logging.error(f"Error loading NLP model: {e}")
            raise

    def process_input(self, text: str) -> str:
        """Process text input and generate a response.
        
        Args:
            text (str): Input text to process
            
        Returns:
            str: Generated response
        """
        try:
            # Add user input to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": text
            })
            
            # Generate response using the model
            response = self.conversational(text)
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant", 
                "content": generated_text
            })
            
            return generated_text

        except Exception as e:
            logging.error(f"Error processing input: {e}")
            return "I'm sorry, I had trouble processing that."

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

if __name__ == "__main__":
    # Test NLP processing
    nlp = NLPHandler()
    
    test_inputs = [
        "Hello, how are you?",
        "What's the weather like?",
        "Tell me a joke"
    ]
    
    for text in test_inputs:
        print(f"\nUser: {text}")
        response = nlp.process_input(text)
        print(f"Assistant: {response}")