import torch
import re

def generate_response(model, tokenizer, prompt):
    """
    Generate an interviewer response from the model based on the given prompt.
    
    Args:
        model: The language model to generate responses.
        tokenizer: The tokenizer to encode and decode text.
        prompt (str): The input prompt for the model.
    
    Returns:
        str: The generated response from the model.
    """
    # Determine the device of the model
    device = next(model.parameters()).device
    
    # Encode the prompt and move to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate output
    with torch.no_grad():  # More memory efficient for inference
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,  # Increased from 256 for more context
            do_sample=True, 
            temperature=0.7,
            top_p=0.9,  # Added nucleus sampling
            repetition_penalty=1.2,  # Discourage repetitive text
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response by removing the prompt if it's included
    if prompt in decoded:
        response = decoded.replace(prompt, "").strip()
    else:
        # Get just the model's response, not the original prompt
        response = decoded.split("\n\n")[-1].strip()
    
    # Clean up the response - remove redundant newlines and spaces
    response = re.sub(r'\n{3,}', '\n\n', response)
    response = re.sub(r'\s{3,}', ' ', response)
    
    # Ensure we're returning an actual response
    if not response or len(response) < 10:
        # Fallback response if model output is too short or empty
        response = "Could you please elaborate on your previous answer? I'd like to understand your approach better."
    
    return response