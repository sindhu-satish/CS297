from enum import Enum

class ChatmlSpecialTokens(str, Enum):
    
    user = "user"
    assistant = "assistant"
    eos_token = tokenizer.eos_token
    bos_token = tokenizer.bos_token
    pad_token = tokenizer.pad_token

    @classmethod
    def list(cls):
        return [c.value for c in cls]