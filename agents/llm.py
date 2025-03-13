
from ollama import ChatResponse
from ollama import chat
class LLM:
    def __init__(self, server, model):
        self.server = server
        self.model = model
        self.call_count = 0
        if server == "ollama":
            from ollama import ChatResponse
            from ollama import chat

    def call(self, prompt):
        if self.server == "ollama":
            response: ChatResponse = chat(model='deepseek-r1:1.5b', messages=[
            {
                'role': 'user',
                'content': 'Why is the sky blue?',
            },
            ])            
            return response.message.content
        else:
            raise Exception("Server not supported")



