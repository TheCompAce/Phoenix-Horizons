import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

import transformers
import argparse

import requests

def generate_openai_response(messages, api_key, model="gpt-3.5-turbo"):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    data = {
        "model": model,
        "messages": messages
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        chat_output = response.json()
        assistant_message = chat_output["choices"][0]["message"]["content"]
        return assistant_message.strip()
    else:
        return f"Error: {response.status_code}, {response.text}"

def create_openai_message_object(text):
    # Initialize message object list
    messages = []
    
    # Search for the system, human, and response sections in the text
    system_start = text.find("### SYSTEM:")
    human_start = text.find("### HUMAN:") if text.find("### HUMAN:") != -1 else text.find("### USER:")
    response_start = text.find("### RESPONSE:")
    
    # If both system and human sections are found, extract the content
    if system_start != -1 and human_start != -1:
        system_content = text[system_start + 11: human_start].strip()
        human_content = text[human_start + 11: response_start if response_start != -1 else None].strip()
        
        messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": human_content})
    else:
        # If the system section is not found, just use the whole text as the user content
        messages.append({"role": "user", "content": text.strip()})
        
    return messages


def generate_deci_response(prompt, settings, max_new_tokens=4096, do_sample=True, top_p=0.95, early_stopping=True, num_beams=5, device = "cuda"):
    # for GPU usage or "cpu" for CPU usage

    settings["tokenizer"].add_special_tokens({'pad_token': '[PAD]'})  

    # Tokenize the prompt and send it to the device
    inputs = settings["tokenizer"](prompt, return_tensors="pt", padding=True, truncation=True)
    
    if torch.cuda.is_available():
        # inputs = {key: val.to('cuda') for key, val in inputs.items()}
        inputs = {key: val.to(settings["device"]) for key, val in inputs.items()}

    # Extract the input_ids and attention_mask
    input_ids = inputs['input_ids']  # Already on device
    attention_mask = inputs['attention_mask']  # Already on device

    # Generate text using the model
    outputs = settings["model"].generate(input_ids,  do_sample=do_sample, top_p=top_p, attention_mask=attention_mask, early_stopping=early_stopping, num_beams=num_beams)
    
    # Decode the output tensor to text
    return settings["tokenizer"].decode(outputs[0], skip_special_tokens=True)


def generate_stabilityai_response(prompt, settings, max_new_tokens=4096, do_sample=True, top_p=0.95, top_k=0):
    # Prepare inputs for the model
    inputs = settings["tokenizer"](prompt, return_tensors="pt").to(settings["device"])
    
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {key: val.to(settings["device"]) for key, val in inputs.items()}
    
    # Remove the attention_mask from inputs if it exists
    if 'attention_mask' in inputs:
        del inputs['attention_mask']
    
    # Generate output using the model
    output = settings["model"].generate(**inputs, do_sample=do_sample, top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens)
    
    # Decode and return the output
    return settings["tokenizer"].decode(output[0], skip_special_tokens=True)


def generate_marx_response(prompt, settings, max_new_tokens=2048, do_sample=True, top_p=0.95, top_k=0): 
    input_ids = settings["tokenizer"](prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(settings["device"])

    # Print device of model parameters
    # for param in settings["model"].parameters():
    #    print(param.device)

    inputs = settings["tokenizer"](prompt, return_tensors="pt", legacy=False).to(settings["device"])
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        # inputs = {key: val.to('cuda') for key, val in inputs.items()}
        inputs = {key: val.to(settings["device"]) for key, val in inputs.items()}

    output = settings["model"].generate(**inputs, do_sample=do_sample, top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens)

    return settings["tokenizer"].decode(output[0], skip_special_tokens=True)

    
def generate_falcon_1b_response(prompt, settings, max_new_tokens=2048, do_sample=True, top_p=0.95, top_k=0, device = "cuda", num_return_sequences=1):
    sequences = settings["pipeline"](
        prompt,
        max_length=max_new_tokens,
        do_sample=do_sample,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        eos_token_id=settings["tokenizer"].eos_token_id,
    )
    responses = []
    for seq in sequences:
        responses.append(seq['generated_text'])
    
    return responses[0]

def generate_chatglm_6b_response(prompt, settings, history =[], max_new_tokens=2048, do_sample=True, top_p=0.95, top_k=0, device = "cuda", num_return_sequences=1):
    response, history = settings["model"].chat(settings["tokenizer"], prompt, history=history)
    
    response_out = {
        "response" : response,
        "history" : history
    }

    return json.dumps(response_out)

def generate_gpt2_response(prompt, settings, max_new_tokens=512, do_sample=True, top_p=0.95, top_k=0, device="cuda", num_return_sequences=1):
    settings["tokenizer"].add_special_tokens({'pad_token': '[PAD]'})    
    
    # Tokenize the input and get attention mask
    encoded_input = settings["tokenizer"](prompt, return_tensors='pt', padding=True, truncation=True)
    
    # Extract the input_ids and attention_mask
    input_ids = encoded_input['input_ids'].to(device)  # Move to device
    attention_mask = encoded_input['attention_mask'].to(device)  # Move to device

    # Generate model output
    output_data = settings["model"].generate(input_ids, top_p=top_p, do_sample=True, attention_mask=attention_mask, max_length=max_new_tokens)

    output = settings["tokenizer"].decode(output_data[0], skip_special_tokens=True)

    return output



def generate_gptj_response(prompt, settings, history =[], max_new_tokens=2048, do_sample=True, top_p=0.95, top_k=0, device = "cuda", num_return_sequences=1):
    settings["tokenizer"].add_special_tokens({'pad_token': '[PAD]'})    
    
    # Tokenize the input and get attention mask
    encoded_input = settings["tokenizer"](prompt, return_tensors='pt', padding=True, truncation=True)
    
    # Extract the input ids and attention mask
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    # Generate model output
    output_data = settings["model"].generate(input_ids, attention_mask=attention_mask, max_length=max_new_tokens)

    output = settings["tokenizer"].decode(output_data[0], skip_special_tokens=True)

    return output

def generate_mistral_response(prompt, settings, history =[], max_new_tokens=2048, do_sample=True, top_p=0.95, top_k=0, device = "cuda", num_return_sequences=1):
    messages = []
    # Search for the system, human, and response sections in the text
    system_start = prompt.find("### SYSTEM:")
    human_start = prompt.find("### HUMAN:") if prompt.find("### HUMAN:") != -1 else prompt.find("### USER:")
    response_start = prompt.find("### RESPONSE:")
    
    # If both system and human sections are found, extract the content
    if system_start != -1 and human_start != -1:
        system_content = prompt[system_start + 11: human_start].strip()
        human_content = prompt[human_start + 11: response_start if response_start != -1 else None].strip()
        
        messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": human_content})
    else:
        # If the system section is not found, just use the whole text as the user content
        messages.append({"role": "user", "content": prompt.strip()})


    encodeds = settings["tokenizer"].apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    settings["model"].to(device)

    generated_ids = settings["model"].generate(model_inputs, max_new_tokens=1000, do_sample=True)
    output = settings["tokenizer"].batch_decode(generated_ids)
    
    return output

def unload_gpu(settings):
    del settings["model"]
    del settings["tokenizer"]
    if "pipeline" in settings:
        del settings["pipeline"]
    torch.cuda.empty_cache()

def setup_model(model_base = "Deci"):
    checkpoint = ""
    device = ""
    pipeline = None
    tokenizer = None
    model = None

    if model_base == "stabilityai":
        checkpoint = "stabilityai/StableBeluga-7B"
        device = "cuda"  # for GPU usage or "cpu" for CPU usage

        tokenizer = AutoTokenizer.from_pretrained("stabilityai/StableBeluga-7B", use_fast=False)
        model = AutoModelForCausalLM.from_pretrained("stabilityai/StableBeluga-7B", torch_dtype=torch.float16, low_cpu_mem_usage=True)

        # Explicitly moving model to GPU if available
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 7:
                model = model.half()
            model = model.to(device)
    elif model_base == "Marx":
        checkpoint = "acrastt/Marx-3B-V2"
        device = "cuda"  # for GPU usage or "cpu" for CPU usage

        tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
        model = LlamaForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device
        )

        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 7:
                model = model.half()
            model = model.to(device)

    elif model_base == "falcon-1b":
        checkpoint = "euclaise/falcon_1b_stage2"
        device = "cuda"  # for GPU usage or "cpu" for CPU usage

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        pipeline = transformers.pipeline(
                "text-generation",
                model=checkpoint,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
    elif model_base == "ChatGLM":
        checkpoint = "THUDM/chatglm2-6b"
        device = "cuda"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).half().cuda()
        
        if torch.cuda.is_available():
            model = model.to(device)

        model = model.eval()
    elif model_base == "gpt2":
        checkpoint = "gpt2"
        device = "cuda"

        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        model = GPT2LMHeadModel.from_pretrained(checkpoint)
        # Explicitly moving model to GPU if available
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 7:
                model = model.half()
            model = model.to(device)

    elif model_base == "gpt3" or model_base == "gpt4":
        checkpoint = model_base
    elif model_base == "GPT-J":
        checkpoint = "EleutherAI/gpt-j-6B"
        device = "cuda"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        # Explicitly moving model to GPU if available
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 7:
                model = model.half()
            model = model.to(device)
            

        model = model.eval()
    elif model_base == "Mistral":
        checkpoint = "mistralai/Mistral-7B-v0.1"
        device = "cuda"

        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        # Explicitly moving model to GPU if available
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 7:
                model = model.half()
    else:
        checkpoint = "Deci/DeciLM-6b"
        device = "cuda"  # for GPU usage or "cpu" for CPU usage
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        # Explicitly moving model to GPU if available
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 7:
                model = model.half()
            model = model.to(device)

    settings = {
        "checkpoint" : checkpoint,
        "device" : device,
        "tokenizer" : tokenizer,
        "model" : model,
        "pipeline" : pipeline
    }

    return settings

def generate_response(prompt, model_base = "Deci", max_new_tokens=1024, responses=1, api_key=None, settings=None):
    # Check if the prompt is a file path
    if os.path.isfile(prompt):
        with open(prompt, 'r') as file:
            prompt = file.read()

    start_time = time.time()

    if settings is None:
        settings = setup_model(model_base)

    ret_resp = []
    for i in range(responses):
        if i > 0:
            start_time = time.time()
        
        resp = ""
        if model_base == "stabilityai":
            resp = generate_stabilityai_response(prompt, settings, max_new_tokens=max_new_tokens)
        elif model_base == "Marx":
            resp = generate_marx_response(prompt, settings, max_new_tokens=max_new_tokens)
        elif model_base == "falcon-1b":
            resp = generate_falcon_1b_response(prompt, settings, max_new_tokens=max_new_tokens)
        elif model_base == "ChatGLM":
            # https://huggingface.co/THUDM/chatglm2-6b
            resp = generate_chatglm_6b_response(prompt, settings, max_new_tokens=max_new_tokens)
        elif model_base == "gpt2":
            # https://huggingface.co/gpt2
            resp = generate_gpt2_response(prompt, settings, max_new_tokens=max_new_tokens)
        elif model_base == "gpt3":
            # https://api.openai.com/v1/chat/completions
            messages = create_openai_message_object(prompt)
            resp = generate_openai_response(messages, model="gpt-3.5-turbo", api_key=api_key)
        elif model_base == "gpt4":
            # https://api.openai.com/v1/chat/completions
            messages = create_openai_message_object(prompt)
            resp = generate_openai_response(messages, model="gpt-4", api_key=api_key)
        elif model_base == "GPT-J":
            # https://huggingface.co/EleutherAI/gpt-j-6b
            resp = generate_gptj_response(prompt, settings, max_new_tokens=max_new_tokens)
        elif model_base == "Mistral":
            # https://huggingface.co/EleutherAI/gpt-j-6b
            resp = generate_mistral_response(prompt, settings, max_new_tokens=max_new_tokens)
        else:
            resp = generate_deci_response(prompt, settings, max_new_tokens=max_new_tokens)
        time_length = time.time() - start_time
        data = {
            "prompt" : prompt,
            "response" : resp,
            "model" : model_base,
            "time" : time_length
        }
        ret_resp.append(data)

    unload_gpu(settings)

    return ret_resp

def TestModels(prompt, max_new_tokens=1024, api_key=None):
    print("Testing models...")
    results =[]
    for model_base in ["Mistral","gpt2",  "stabilityai", "falcon-1b", "Deci", "Marx", "ChatGLM"]: #  "GPT-J", "gpt3", "gpt4"]:
        print(f"Testing {model_base}...")
        start_time = time.time()
        responses = generate_response(prompt, model_base, max_new_tokens=max_new_tokens, responses=2, api_key=api_key)
        time_length = time.time() - start_time
        results.append({
            "prompt" : prompt,
            "responses" : responses,
            "model" : model_base,
            "time" : time_length
        })
        print(json.dumps(results[len(results)-1], indent=4))
    
    out_file = "results.json"

    with open(out_file, 'w') as file:
        file.write(json.dumps(results))

def loop_generate_response(initial_prompt, model_name="Deci", max_new_tokens=1024, api_key=None):

    # Initialize chat history list
    chat_history = []
    
    # Initialize prompt with the starter prompt
    prompt = initial_prompt

    settings = setup_model(model_name)
    
    while True:
        # Record start time
        start_time = time.time()
        
        # Generate response using existing `generate_response` function
        response = generate_response(prompt, model_base=model_name, api_key=api_key, settings=settings)

        response = response[0]["response"]

        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        # Record end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Append response and elapsed time to chat history
        chat_entry = {
            "prompt": prompt,
            "response": response,
            "elapsed_time": elapsed_time
        }
        chat_history.append(chat_entry)
        
        # Save chat history to chat.json
        with open("chat.json", "w") as json_file:
            json.dump(chat_history, json_file)
        
        # Print the response
        print(f"Response: {response}")
        
        # Update the prompt for the next loop
        prompt = response


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate response using AI model')
    parser.add_argument('prompt', type=str, help='Prompt or file path containing the prompt')
    parser.add_argument('--model', type=str, default="Deci", help='Prompt or file path containing the prompt')
    parser.add_argument('--size', type=int, default=1024, help='Maximum token length for the generated response')
    parser.add_argument('--responses', type=int, default=1, help='Number of responses to generate')
    parser.add_argument('--openai_key', type=str, default=None, help='OpenAI api key.')

    args = parser.parse_args()

    time_length = 0
    
    TestModels(args.prompt, args.size, args.openai_key)
    # loop_generate_response(args.prompt, model_name=args.model, api_key=args.openai_key, max_new_tokens=args.size)

"""     start_time = time.time()
        responses = generate_response(args.prompt, args.model, max_new_tokens=args.size, responses=args.responses)
        time_length = time.time() - start_time
        response = {
            "prompt" : args.prompt,
            "responses" : responses,
            "model" : time_length,
            "time" : time_length
        }
        print(json.dumps(response)) """
