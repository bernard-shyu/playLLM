#!/usr/bin/env python3
#-------------------------------------------------------------------------------------------
import os
import sys
import time
import signal
import logging
import numpy as np

from termcolor import cprint

from .utils.args import ArgParser
from .utils.prompts import load_prompts
from .utils.table import print_table 

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

"""_Usage_
python3 -m playLLM.chat --help
python3 -m playLLM.chat --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --do-sample 

"""

#-------------------------------------------------------------------------------------------
# see utils/args.py for options
parser = ArgParser()
parser.add_argument("--no-streaming", action="store_true", help="wait to output entire reply instead of token by token")
args = parser.parse_args()

# 1. Load the model and tokenizer
#args.model = "twinkle-ai/Llama-3.2-3B-F1-Instruct"

print(f"Loading model '{args.model}'... (this may take a few minutes)")
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for faster loading (if your GPU supports it)
    device_map="auto",           # Automatically puts model on available GPUs (or CPU)
    #quant=args.quant, 
    #api=args.api
    trust_remote_code=True
)

# 2. Load --prompt parameters
prompts = load_prompts(args.prompt)

#-------------------------------------------------------------------------------------------
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
    
    # 3. Tokenize input
    inputs = tokenizer(user_prompt, return_tensors="pt").to(model.device)

    # 4. Generate text
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,           # How many tokens to generate (reference: 200)
            do_sample=args.do_sample,                     # Random sampling (creative)  (reference: True)
            temperature=args.temperature,                 # Lower = more deterministic  (reference: 0.7)
            top_p=args.top_p,                             # Top-p (nucleus sampling)    (reference: 0.9)
            repetition_penalty=args.repetition_penalty,   # Penalize repeated phrases (reference: 1.1)
            eos_token_id=tokenizer.eos_token_id,          # End generation at EOS
            min_new_tokens=args.min_new_tokens,
        )

    # 5. Decode and print output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n--- Generated Response ---")
    print(generated_text)

    """
    # generate bot reply
    reply = model.generate(
        user_prompt, 
        streaming=not args.no_streaming, 
        stop_tokens=StopTokens,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
    )
        
    if args.no_streaming:
        print(reply)
    else:
        for token in reply:
            print(token, end='', flush=True)
            if interrupt:
                reply.stop()
                interrupt.reset()
                break
            
    print('\n')
    print_table(model.stats)
    print('')
    """

