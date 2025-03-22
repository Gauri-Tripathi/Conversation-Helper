# Llama-3.1-8B-Sarcasm: Human-Like Conversational AI

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Gauri--tr%2Fllama--3.1--8b--sarcasm-blue)](https://huggingface.co/Gauri-tr/llama-3.1-8b-sarcasm)
[![GGUF Model](https://img.shields.io/badge/🤗%20GGUF-Gauri--tr%2Fllama--3.1--8b--sarcasm--Q4__0--GGUF-green)](https://huggingface.co/Gauri-tr/llama-3.1-8b-sarcasm-Q4_0-GGUF)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fine-tuned LLM based on Meta's Llama-3.1-8B-Instruct that generates natural, human-like responses in conversations - whether sarcastic, funny, or casual. This model helps you craft the perfect reply for any conversation context.

## 🌟 Overview

This project fine-tunes the Llama-3.1-8B-Instruct model to generate responses that mimic human conversational styles, with a special focus on sarcasm and humor. The model is designed to understand conversation context and generate appropriate, engaging responses that feel authentic and natural.

## ✨ Features

- **Human-like responses**: Generates text that mimics genuine human conversation styles
- **Contextual understanding**: Processes both single-turn exchanges and multi-turn conversations (up to 7 turns)
- **Versatile tone**: Can respond with sarcasm, humor, or normal conversation as appropriate to the context
- **Optimized for deployment**: Available in both full model and quantized GGUF formats

## 📊 Model Performance

The model demonstrates impressive capabilities in generating human-like responses across various conversational contexts. Below are examples showcasing the model's output quality:

### Response Examples

<table>
  <tr>
    <th>Input</th>
    <th>Generated Response</th>
  </tr>
  <tr>
    <td>"I just spent 8 hours debugging a single line of code."</td>
    <td>"Oh, only 8 hours? Amateur. The real pros spend at least two days, three energy drinks, and half their sanity on a semicolon. But hey, at least you've discovered a new level of patience most people don't know exists!"</td>
  </tr>
  <tr>
    <td>"My friend keeps canceling our plans at the last minute."</td>
    <td>"Sounds like they've mastered the art of the disappearing act! Maybe start telling them the meeting time is three hours earlier than it actually is. Either they'll show up super early and learn their lesson, or you'll finally have proof they're secretly a time traveler."</td>
  </tr>
</table>

*Note: Screenshots demonstrating the model in action can be found in the [Examples](#examples) section below.*

## 🚀 Getting Started

### Using the Model with Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "Gauri-tr/llama-3.1-8b-sarcasm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Format input
def format_prompt(conversation_input, instruction=None):
    if instruction is None:
        instruction = "Respond to this message as if you were in a conversation. Determine the tone and style of the conversation and reply accordingly. Be funny, sarcastic and smart as well."
    
    prompt = f"""Below is an instruction that describes a task, and an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{conversation_input}

### Response:
"""
    return prompt

# Generate response
def generate_response(conversation_input, instruction=None):
    prompt = format_prompt(conversation_input, instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.15
    )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("### Response:")[1].strip()
    return response

# Example usage
response = generate_response("I just spent 8 hours debugging a single line of code.")
print(response)
```

### Using the Quantized Model with llama.cpp Python

```python
from llama_cpp import Llama

# Load the model
model_path = "path/to/llama-3.1-8b-sarcasm-Q4_0.gguf"  # Download from Hugging Face
llm = Llama(
    model_path=model_path,
    n_ctx=2048,  # Context window
    n_threads=4  # Adjust based on your CPU
)

# Format prompt
def format_prompt(conversation_input, instruction=None):
    if instruction is None:
        instruction = "Respond to this message as if you were in a conversation. Determine the tone and style of the conversation and reply accordingly. Be funny, sarcastic and smart as well."
    
    prompt = f"""Below is an instruction that describes a task, and an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{conversation_input}

### Response:
"""
    return prompt

# Generate response
def generate_response(conversation_input, instruction=None):
    prompt = format_prompt(conversation_input, instruction)
    
    output = llm(
        prompt,
        max_tokens=128,
        temperature=0.8,
        top_p=0.9,
        repeat_penalty=1.15,
        echo=True
    )
    
    response_text = output["choices"][0]["text"]
    response = response_text.split("### Response:")[1].strip()
    return response

# Example usage
response = generate_response("I just spent 8 hours debugging a single line of code.")
print(response)
```

### Integration with LangChain

```python
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the LlamaCpp model
model_path = "path/to/llama-3.1-8b-sarcasm-Q4_0.gguf"  # Download from Hugging Face
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.8,
    top_p=0.9,
    n_ctx=2048,
    repeat_penalty=1.15,
    max_tokens=128,
    n_threads=4  # Adjust based on your CPU
)

# Create a prompt template
template = """Below is an instruction that describes a task, and an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Respond to this message as if you were in a conversation. Determine the tone and style of the conversation and reply accordingly. Be funny, sarcastic and smart as well.

### Input:
{conversation_input}

### Response:"""

prompt = PromptTemplate(template=template, input_variables=["conversation_input"])

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Example usage
response = chain.run("I just spent 8 hours debugging a single line of code.")
print(response)
```

## 📝 Training Methodology

### Dataset

The model was trained on a diverse dataset that includes:
- Multi-turn conversations (up to 7 previous turns of context)
- Single-turn exchanges from social media
- Various conversational styles (casual, humorous, sarcastic)

This mixed approach ensures the model can handle both short, one-off interactions and extended conversation threads with proper context understanding.

### Fine-tuning Approach

The model was fine-tuned using AdaLoRA (Adaptive Low-Rank Adaptation), which provides several advantages:

- **Parameter Efficiency**: Reduces memory requirements while maintaining performance
- **Adaptive Rank Allocation**: Dynamically adjusts ranks across different layers
- **Better Generalization**: Preserves more of the pre-trained model's knowledge
- **Training Stability**: Overcomes the limitations of other PEFT methods which failed to adapt properly

### Technical Implementation

The fine-tuning process used:
- 4-bit quantization (QLoRA) to reduce memory usage
- AdaLoRA configuration with dynamic rank adjustment
- Flash Attention 2 for improved computational efficiency
- Gradient checkpointing to optimize memory usage
- Cosine learning rate schedule with warmup

## 📸 Examples

*Note: Include screenshots of the model in action here to demonstrate its capabilities*

## 🔧 Model Versions

- **Full Model**: [`Gauri-tr/llama-3.1-8b-sarcasm`](https://huggingface.co/Gauri-tr/llama-3.1-8b-sarcasm)
- **Quantized GGUF**: [`Gauri-tr/llama-3.1-8b-sarcasm-Q4_0-GGUF`](https://huggingface.co/Gauri-tr/llama-3.1-8b-sarcasm-Q4_0-GGUF)

## 🛠️ Development

The model was fine-tuned using PyTorch and the Hugging Face Transformers library. The code used for training is available in this repository.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Meta for the base Llama-3.1-8B-Instruct model
- Hugging Face for their Transformers library and model hosting
- The creators of AdaLoRA for their innovative parameter-efficient fine-tuning method

## ⚠️ Disclaimer

This model is designed for generating human-like conversational responses. While it can generate sarcastic or humorous content, it is not intended to produce harmful, offensive, or inappropriate content. Users are responsible for ensuring appropriate use of the model's outputs.