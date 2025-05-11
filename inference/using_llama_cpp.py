from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import hf_hub_download
import os

def format_prompt(instruction: str, input_text: str) -> str:
    """Format the prompt to match training structure with conversation context"""
    # Split input into messages if it contains multiple turns
    messages = input_text.split('\n')
    
    conversation_context = ""
    if len(messages) > 1:
        conversation_context = "Context:\n" + "\n".join(f"Message {i+1}: {msg.strip()}" 
                                                      for i, msg in enumerate(messages))
    else:
        conversation_context = f"Context:\nMessage: {input_text}"

    return f"""Below is an instruction that describes a task, and an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{conversation_context}

### Response:
"""

def download_model_from_hf():
    """Download the model from HuggingFace if not already present"""
    model_path = hf_hub_download(
        repo_id="Gauri-tr/llama-3.1-8b-sarcasm-Q4_0-GGUF",
        filename="llama-3.1-8b-sarcasm-q4_0.gguf",
        cache_dir="models"
    )
    return model_path

def get_model_path():
    """Get or download the model path"""
    local_path = os.path.join("models", "llama-3.1-8b-sarcasm-q4_0.gguf")
    if not os.path.exists(local_path):
        return download_model_from_hf()
    return local_path

def generate_with_llamacpp(instruction: str, input_text: str) -> str:
    """Direct llama.cpp approach"""
    model_path = get_model_path()
    
    # Initialize model
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_batch=256,
        n_threads=4,
    )
    
    prompt = format_prompt(instruction, input_text)
    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        stop=["</s>", "### Instruction:", "### Input:"],
        echo=False
    )
    
    return output["choices"][0]["text"].strip()

def generate_with_langchain(instruction: str, input_text: str) -> str:
    """LangChain with llama.cpp approach"""
    model_path = get_model_path()
    
    # Setup callback manager for streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Initialize LangChain LLM
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.8,
        max_tokens=64,
        n_ctx=512,
        callback_manager=callback_manager,
        verbose=True
    )
    
    prompt = format_prompt(instruction, input_text)
    response = llm(prompt)
    return response.strip()

if __name__ == "__main__":
    print("Ensuring model is downloaded...")
    get_model_path()
    
    # Example usage with new prompt structure
    instruction = "Write a sarcastic response about the weather"
    input_text = "It's pouring rain outside and someone says the weather is perfect"
    
    print("Using direct llama.cpp:")
    response1 = generate_with_llamacpp(instruction, input_text)
    print(response1)
    
    print("\nUsing LangChain:")
    response2 = generate_with_langchain(instruction, input_text)
    print(response2)
