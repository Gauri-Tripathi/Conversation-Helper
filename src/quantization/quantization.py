import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, create_repo, upload_folder
import subprocess
import tempfile
import shutil

def load_and_merge_adalora(base_model_id, adalora_weights_path, output_dir):
    """
    Load a base model and merge AdaLoRA fine-tuned weights
    
    Args:
        base_model_id (str): HuggingFace model ID for the base model
        adalora_weights_path (str): Path to AdaLoRA weights repository
        output_dir (str): Directory to save the merged model
    
    Returns:
        tuple: (merged model, tokenizer)
    """
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    print(f"Loading AdaLoRA adapter from: {adalora_weights_path}")
    model = PeftModel.from_pretrained(base_model, adalora_weights_path)
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

def convert_to_gguf(merged_model_dir, gguf_output_dir, quantization="Q4_0"):
    """
    Convert merged model to GGUF 4.0 format
    
    Args:
        merged_model_dir (str): Directory containing the merged model
        gguf_output_dir (str): Directory to save the GGUF model
        quantization (str): Quantization format
    
    Returns:
        str: Path to the GGUF model file
    """
    # Create output directory if it doesn't exist
    os.makedirs(gguf_output_dir, exist_ok=True)
    
    # Get model name from directory
    model_name = os.path.basename(os.path.normpath(merged_model_dir))
    gguf_model_path = os.path.join(gguf_output_dir, f"{model_name}-{quantization}.gguf")
    
    print(f"Converting model to GGUF format: {quantization}")
    
    # Use llama.cpp convert script - assuming it's installed
    convert_cmd = [
        "python", "-m", "llama_cpp.convert", 
        "--outtype", "f16", 
        "--outfile", gguf_model_path,
        merged_model_dir
    ]
    
    try:
        subprocess.run(convert_cmd, check=True)
        print(f"Successfully converted to GGUF: {gguf_model_path}")
        return gguf_model_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting to GGUF: {e}")
        raise

def push_to_huggingface(local_dir, repo_id, token=None):
    """
    Push model to Hugging Face model repository
    
    Args:
        local_dir (str): Local directory containing the model files
        repo_id (str): Hugging Face repository ID (username/repo-name)
        token (str): Hugging Face API token
    
    Returns:
        str: URL of the Hugging Face repository
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable or token parameter must be provided")
    
    print(f"Creating/accessing repository: {repo_id}")
    api = HfApi(token=token)
    
    try:
        create_repo(repo_id=repo_id, token=token, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")
    
    print(f"Uploading model files to {repo_id}")
    result = upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        token=token,
        commit_message="Upload merged model with AdaLoRA weights and GGUF conversion"
    )
    
    print(f"Successfully pushed to Hugging Face: {result}")
    return result

def main(base_model_id, adalora_weights_path, output_dir, gguf_output_dir, hf_repo_id, hf_token=None):
    """
    Main function to execute the entire workflow
    
    Args:
        base_model_id (str): HuggingFace model ID for the base model
        adalora_weights_path (str): Path to AdaLoRA weights repository
        output_dir (str): Directory to save the merged model
        gguf_output_dir (str): Directory to save the GGUF model
        hf_repo_id (str): Hugging Face repository ID (username/repo-name)
        hf_token (str): Hugging Face API token
    """
    try:
        
        _, _ = load_and_merge_adalora(base_model_id, adalora_weights_path, output_dir)
        
  
        gguf_path = convert_to_gguf(output_dir, gguf_output_dir)
 
        with tempfile.TemporaryDirectory() as temp_dir:
 
            merged_model_temp = os.path.join(temp_dir, "merged_model")
            shutil.copytree(output_dir, merged_model_temp)
            
   
            gguf_temp = os.path.join(temp_dir, "gguf")
            os.makedirs(gguf_temp, exist_ok=True)
            shutil.copy2(gguf_path, gguf_temp)
     
            push_to_huggingface(temp_dir, hf_repo_id, hf_token)
    
    except Exception as e:
        print(f"Error in workflow: {e}")
        raise

