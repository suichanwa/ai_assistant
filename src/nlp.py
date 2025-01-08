from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

class NLPHandler:
    def __init__(self, model_name="decapoda-research/llama-7b-hf"):
        """Initialize with a publicly available LLaMA model."""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.conversation_history = []
            logging.info(f"AI Assistant initialized with model: {model_name}")

            # Generation parameters
            self.max_length = 256
            self.temperature = 0.7
            self.top_p = 0.9
            self.repetition_penalty = 1.2
            self.num_beams = 4

        except Exception as e:
            logging.error(f"Error initializing NLPHandler: {e}")
            raise

    def process_input(self, text: str) -> str:
        """Generate a response using the model."""
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Format conversation for model
            prompt = self._format_prompt()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate with improved parameters
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                repetition_penalty=self.repetition_penalty,
                do_sample=True
            )
            
            # Extract only the new response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            logging.error(f"Error processing input: {e}")
            return "I'm sorry, I had trouble processing that."

    def _format_prompt(self) -> str:
        """Format conversation history for better context."""
        prompt = []
        # Keep last 5 turns for context
        for msg in self.conversation_history[-5:]:
            role = "Human:" if msg["role"] == "user" else "Assistant:"
            prompt.append(f"{role} {msg['content']}")
        return "\n".join(prompt) + "\nAssistant:"

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []