import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model():
    model_name = "Gauri-tr/llama-3.1-8b-sarcasm"
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

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

def generate_response(instruction: str, input_text: str, model, tokenizer, max_length=512):
    prompt = format_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    return response.split("### Response:")[-1].strip()

if __name__ == "__main__":
    model, tokenizer = load_model()
    
    # Example usage with new prompt structure
    instruction = "Write a sarcastic response about the weather"
    input_text = "It's pouring rain outside and someone says the weather is perfect"
    
    response = generate_response(instruction, input_text, model, tokenizer)
    print("Response:", response)
