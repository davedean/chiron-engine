from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys

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

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    #print(f"Model response: {response.strip()}")

    #response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return response


class TokenSampler:
    """Base class for samplers."""
    def sample(self, logits, **kwargs):
        raise NotImplementedError("Sampler must implement the 'sample' method.")
class GreedySampler(TokenSampler):
    def sample(self, logits, **kwargs):
        return torch.argmax(logits, dim=-1).item()

class BeamSampler(TokenSampler):
    def __init__(self, num_beams=5):
        self.num_beams = num_beams
    
    def sample(self, logits, **kwargs):
        # Beam sampling logic can be complex,
        # for simplicity we will handle it differently.
        raise NotImplementedError("Beam search isn't easily token-wise without advanced handling.")

class SamplingSampler(TokenSampler):
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def sample(self, logits, **kwargs):
        logits = logits / self.temperature
        probabilities = torch.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).item()

def create_sampler(sampler_type, **kwargs):
    if sampler_type == 'greedy':
        return GreedySampler()
    elif sampler_type == 'beam':
        return BeamSampler(**kwargs)
    elif sampler_type == 'sampling':
        return SamplingSampler(**kwargs)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

def generate_response_token_by_token(model, tokenizer, model_inputs, max_new_tokens=50, initial_sampler='greedy'):
    sampler = create_sampler(initial_sampler)
    generated_ids = []

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # Focus on the last token's output

        # Sampling a token
            logits = outputs.logits[:, -1, :]
        next_token_id = sampler.sample(logits)

        # Update input_ids for the next iteration
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=-1)

        # Allow adaptations to the sampler based on heuristics
        sampler = apply_heuristic_logic(generated_ids, sampler)

        # Decode token and print for debugging
        print(tokenizer.decode([next_token_id], skip_special_tokens=True),end="")
        sys.stdout.flush()

        # print(f"Generated token: {tokenizer.decode([next_token_id], skip_special_tokens=True)}")
        # Optional stopping condition if needed
        if next_token_id == tokenizer.eos_token_id:
            break

    print("\n")
    #response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    response = True
    return response

def apply_heuristic_logic(generated_ids, current_sampler):
    # Implement heuristic logic that changes the sampler
    # based on generated_ids or other state.
    # return current_sampler  # Return the possibly modified sampler
    return current_sampler

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
    # model, tokenizer = download_model("Qwen/Qwen2.5-0.5B-Instruct","./qwen2.5-0.5B-Instruct")
    model, tokenizer = download_model()

    # system prompt
    prompt_system = (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
    )

    prompt_system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
    while True:
        # Prompt the user for input
        prompt = input("\nEnter your question (or press Enter to quit): ").strip()

        # Check if the user wants to quit
        if prompt == "":
            print("Exiting the interactive session.")
            break

        messages = [ {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt}
        ]
        # messages.append({"role": "user", "content": prompt}) # use append if continuing a session

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        messages = [ {"role": "system", "content": prompt_system}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate and print the response
        # response = generate_response(model, tokenizer, model_inputs, sampler='greedy', max_new_tokens=2000)
        response = generate_response_token_by_token(model, tokenizer, model_inputs, max_new_tokens=2000,initial_sampler="greedy")
        # print(f"Model response: {response.strip()}")
