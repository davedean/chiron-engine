from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


import os

def download_model(model_name = "Qwen/Qwen2.5-0.5B",local_path = "./qwen2.5-0.5B"):

    model_path="./models/"+local_path

    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"Model and tokenizer already exist at {model_path}.")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        print(f"Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Downloading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)

        print("Download complete!")

        # Optionally, you can save the model and tokenizer locally
        print(f"Saving model and tokenizer locally in {model_path}...")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        print("Model and tokenizer saved locally.")

    return model, tokenizer

def generate_response(model, tokenizer, model_inputs, max_new_tokens=50, sampler='greedy'):
    
    # Select the sampler
    if sampler == 'greedy':
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs,max_new_tokens=max_new_tokens)
            # output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    elif sampler == 'beam':
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs,max_new_tokens=max_new_tokens,num_beams=5, early_stopping=True)
            # output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, num_beams=5, early_stopping=True)
    elif sampler == 'sampling':
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs,max_new_tokens=max_new_tokens,do_sample=True, temperature=0.7)
            # output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    else:
        raise ValueError("Sampler not recognized. Please use 'greedy', 'beam', or 'sampling'.")
    

    generated_ids = model.generate(**model_inputs,max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    #print(f"Model response: {response.strip()}")

    #response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    
    # prompt for base style model
    # prompt = (
    #     "Q: Which number is larger, 3 or 5?\n"
    #     "A: 5\n"
    #     "Q: Which number is larger, 10.5 or 7.4?\n"
    #     "A: 10.5\n"
    #     "Q: Which is larger, 9.8 or 9.11?\n"
    #     "A: "
    # )

    # prompt = ( "Q: Which is larger, 9.8 or 9.11?\n"
    #            "A: To compare the sizes of 9.8 and 9.11, we can look at their decimal places:\n"
    # )

    # model, tokenizer = download_model()
    # response = generate_response(model, tokenizer, prompt, sampler='greedy',max_new_tokens=800)
    # print(f"Model response: {response.strip()}")
    # printf("\n\n")

    # set up model/tokenizer
    model, tokenizer = download_model("Qwen/Qwen2.5-0.5B-Instruct","./qwen2.5-0.5B-Instruct")

    # system prompt
    prompt_system = (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
    )

    while True:
        # Prompt the user for input
        prompt = input("Enter your question (or press Enter to quit): ").strip()
        
        # Check if the user wants to quit
        if prompt == "":
            print("Exiting the interactive session.")
            break

        # Prepare the message and model input        
        messages = [ {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt}  
        ]

        # messages.append({"role": "user", "content": prompt}) # use append if continuing a session

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate and print the response
        response = generate_response(model, tokenizer, model_inputs, sampler='greedy', max_new_tokens=2000)
        print(f"Model response: {response.strip()}")