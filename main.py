import pyttsx3
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class TeachingAssistant:
    def __init__(self):
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        
        # Initialize GPT-2 model
        print("Loading the knowledge model... This may take a moment...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        print("Model loaded successfully!")
        
    def speak(self, text):
        """Convert text to speech"""
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
    
    def get_ai_response(self, user_input):
        """Get response from GPT-2 model"""
        try:
            # Format the input to encourage detailed, knowledgeable responses
            prompt = f"Question: {user_input}\nDetailed Answer:"
            
            # Encode the input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=1000, truncation=True)
            
            # Generate response with parameters optimized for detailed answers
            outputs = self.model.generate(
                inputs,
                max_length=500,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
            
            # Decode and clean up the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response
            response = response.replace(prompt, "").strip()
            
            # If response is empty or too short, provide a fallback
            if not response or len(response) < 10:
                return "Let me explain that in detail. " + self.get_fallback_response(user_input)
                
            return response
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_fallback_response(self, query):
        """Provide a fallback response when the main model fails"""
        query = query.lower()
        
        # Basic math operations
        if "addition" in query or "add" in query:
            return "Addition is a fundamental mathematical operation where we combine two or more numbers to get their sum. For example, 2 + 3 = 5. To add numbers, you line them up by place value and add each column, carrying over when the sum exceeds 9."
        elif "subtraction" in query or "subtract" in query:
            return "Subtraction is the process of taking away one number from another. For example, 5 - 3 = 2. To subtract, you line up the numbers by place value and subtract each column, borrowing when necessary."
        elif "multiplication" in query or "multiply" in query:
            return "Multiplication is repeated addition. For example, 3 × 4 means adding 3 four times: 3 + 3 + 3 + 3 = 12. You can use the multiplication table or the long multiplication method for larger numbers."
        elif "division" in query or "divide" in query:
            return "Division is the process of sharing or grouping numbers. For example, 12 ÷ 3 = 4 means that 12 can be divided into 3 equal groups of 4. You can use long division for larger numbers."
            
        # General knowledge topics
        elif "history" in query:
            return "History is the study of past events, particularly in human affairs. It helps us understand how societies, cultures, and civilizations have evolved over time."
        elif "science" in query:
            return "Science is the systematic study of the structure and behavior of the physical and natural world through observation and experiment. It includes fields like physics, chemistry, biology, and astronomy."
        elif "geography" in query:
            return "Geography is the study of places and the relationships between people and their environments. It includes physical geography (landforms, climate) and human geography (population, culture)."
            
        # If no specific topic is matched
        return "I can help you learn about that. Could you please ask a more specific question?"

    def run(self):
        """Main loop for the teaching assistant"""
        self.speak("Hello! I'm your AI teaching assistant. I can help explain any topic in detail. Just ask me anything, and I'll provide a comprehensive explanation. Type 'exit' to quit.")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit", "bye"]:
                    self.speak("Goodbye! I hope you learned something new today!")
                    break
                
                response = self.get_ai_response(user_input)
                self.speak(response)
                
            except KeyboardInterrupt:
                self.speak("Goodbye! I hope you learned something new today!")
                break
            except Exception as e:
                self.speak(f"Sorry, I encountered an error: {str(e)}")

if __name__ == "__main__":
    assistant = TeachingAssistant()
    assistant.run()
