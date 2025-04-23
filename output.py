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
    
    # More robust way to extract the model's response
    if prompt in decoded:
        # Find where the prompt ends and the response begins
        response_start = decoded.find(prompt) + len(prompt)
        response = decoded[response_start:].strip()
    else:
        # Try to find a response separator or take the last part
        parts = decoded.split("\n\n")
        if len(parts) > 1:
            # Take the last meaningful part
            response = parts[-1].strip()
        else:
            # If no clear division, just take everything after the first line
            lines = decoded.split("\n")
            if len(lines) > 1:
                response = "\n".join(lines[1:]).strip()
            else:
                # Just use the entire output if nothing else works
                response = decoded.strip()
    
    # Clean up the response
    response = re.sub(r'\n{3,}', '\n\n', response)
    response = re.sub(r'\s{3,}', ' ', response)
    
    # Ensure we're returning an actual response
    if not response or len(response) < 10:
        # Fallback response if model output is too short or empty
        response = "Could you please elaborate on your previous answer? I'd like to understand your approach better."
    
    return response