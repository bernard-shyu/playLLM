#!/usr/bin/env python3
#-------------------------------------------------------------------------------------------
import os
import sys
import json
import time
import signal
import logging
import numpy as np

from termcolor import cprint

from .utils.args import ArgParser
from .utils.prompts import load_prompts
from .utils.table import print_table 
from .utils.templates import *

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

"""_Usage_
python3 -m playLLM.chat --help
python3 -m playLLM.chat --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --do-sample 

"""

#-------------------------------------------------------------------------------------------
def load_model(model_id: str):
    """Load the model and tokenizer."""
    print(f"Loading model '{model_id}'... (this may take a few minutes)")

    # Load tokenizer and model (this will download the model if not already present locally)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load model with appropriate configuration for your hardware
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for faster loading (need GPU supports), otherwise, torch.float16 (half precision)  or torch.float32 
        #quant=args.quant, 
        #api=args.api
        device_map="auto",           # Automatically puts model on available GPUs (or CPU)
        trust_remote_code=True
    )
    return model, tokenizer

def prompt_template(model, tokenizer, user_prompt: str, system_prompt: str):
    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant."
    
    # Format the prompt according to Llama 3.2 chat template
    messages = []
    
    # Add system message if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add user message
    messages.append({"role": "user", "content": user_prompt})
    
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 3. Tokenize input
    #inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = tokenizer(prompt, return_tensors="pt")                 # Tokenize the prompt
    inputs = {k: v.to(model.device) for k, v in inputs.items()}     # Move input tensors to the same device as the model

    return inputs
        

def generate_response(model, tokenizer, user_prompt: str, args):
    """Generate text based on a prompt."""

    # apply prompt template
    inputs = prompt_template(model, tokenizer, user_prompt, args.system_prompt)

    # 4. Generate bot reply
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,           # How many tokens to generate (reference: 200)
            min_new_tokens=args.min_new_tokens,
            do_sample=args.do_sample,                     # Random sampling (creative)  (reference: True)
            temperature=args.temperature,                 # Lower = more deterministic  (reference: 0.7)
            top_p=args.top_p,                             # Top-p (nucleus sampling)    (reference: 0.9)
            repetition_penalty=args.repetition_penalty,   # Penalize repeated phrases (reference: 1.1)
            eos_token_id=tokenizer.eos_token_id,          # End generation at EOS
        )

    # Decode and return only the assistant's response
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    assistant_response = full_response.split("<｜Assistant｜>")[-1].strip()
    
    return assistant_response

#-------------------------------------------------------------------------------------------
def main(args):
    # 1. Load the model and tokenizer
    model, tokenizer = load_model(args.model)   # model = "twinkle-ai/Llama-3.2-3B-F1-Instruct"

    # 2. Load --prompt parameters
    prompts = load_prompts(args.prompt)

    #---------------------------------------------------------------------------------------
    print("="*120 + "\n")
    while True: 
        # get the next prompt from the list, or from the user interactivey
        if isinstance(prompts, list):
            if len(prompts) > 0:
                user_prompt = prompts.pop(0)
                print(user_prompt, end='', flush=True)
            else:
                break
        else:
            cprint('>> PROMPT: ', 'blue', end='', flush=True)
            user_prompt = sys.stdin.readline().strip()
            if user_prompt.lower() in ['exit', 'quit', 'bye']:
                print("Exiting chat. Goodbye!")
                break
        
        # Model Generate output
        generated_text = generate_response(model, tokenizer, user_prompt, args)
        print("\n--- Generated Text ---", generated_text, "\n")


#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgParser()
    #parser.add_argument("--no-streaming", action="store_true", help="wait to output entire reply instead of token by token")
    args = parser.parse_args()

    main(args)

#-------------------------------------------------------------------------------------------
