from src.config import Config, PROMPT_MAPPING
from src.llm import get_model_response_stream

class PromptAgent:
    def __init__(self, agent_type, model_name=None):
        self.prompt = PROMPT_MAPPING.get(agent_type)
        if not self.prompt:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.model_name = model_name or Config.DEFAULT_MODEL
        self.messages = [{"role": "system", "content": self.prompt}]
    
    async def generate_content(self):
        response_stream = await get_model_response_stream(
            model_name=self.model_name, 
            messages=self.messages
        )
        async for chunk in response_stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    yield content
    
    async def chat(self):
        while True:
            user_input = input("user: ")
            if user_input.lower() in Config.EXIT_COMMANDS:
                break
                
            self.messages.append({"role": "user", "content": user_input})

            reply_msg = ""
            print("assistant: ", end="", flush=True)
            async for content in self.generate_content():
                reply_msg += content
                print(content, end="", flush=True)
            print("\n")
            
            self.messages.append({"role": "assistant", "content": reply_msg}) 
