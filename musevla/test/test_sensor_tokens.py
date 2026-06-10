"""Test script to verify sensor token encoding/decoding"""
import torch
from transformers import AutoTokenizer, PaliGemmaProcessor

# Load the base tokenizer/processor
print("Loading base tokenizer...")
processor = PaliGemmaProcessor.from_pretrained("google/paligemma2-3b-mix-224")
tokenizer = processor.tokenizer

print(f"Base tokenizer vocab size: {len(tokenizer)}")
print(f"Base tokenizer type: {type(tokenizer)}")

# Add sensor tokens
special_tokens = ["<None>", "<thermal>", "<radar>", "<audio>"]
print(f"\nAdding special tokens: {special_tokens}")

existing_special_tokens = [token for token in special_tokens if token in tokenizer.get_vocab()]
print(f"Existing special tokens: {existing_special_tokens}")

if len(existing_special_tokens) < len(special_tokens):
    new_special_tokens = [token for token in special_tokens if token not in existing_special_tokens]
    print(f"New special tokens to add: {new_special_tokens}")
    
    num_added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": new_special_tokens}
    )
    print(f"Number of tokens added: {num_added_tokens}")

print(f"Final tokenizer vocab size: {len(tokenizer)}")

# Test encoding and decoding
print("\n" + "="*60)
print("Testing sensor token encoding/decoding")
print("="*60)

sensor_token_ids = {}
for token in special_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    sensor_token_ids[token] = token_id
    
    print(f"\nToken: {token}")
    print(f"  Token ID: {token_id}")
    
    # Test encoding
    encoded = tokenizer.encode(token, add_special_tokens=False)
    print(f"  Encoded (add_special_tokens=False): {encoded}")
    
    encoded_with_special = tokenizer.encode(token, add_special_tokens=True)
    print(f"  Encoded (add_special_tokens=True): {encoded_with_special}")
    
    # Test decoding
    decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    print(f"  Decoded (skip_special_tokens=False): '{decoded}'")
    
    decoded_skip = tokenizer.decode([token_id], skip_special_tokens=True)
    print(f"  Decoded (skip_special_tokens=True): '{decoded_skip}'")
    
    # Test in context
    test_text = f"{token} object_name"
    encoded_text = tokenizer.encode(test_text, add_special_tokens=False)
    decoded_text = tokenizer.decode(encoded_text, skip_special_tokens=False)
    print(f"  Test text: '{test_text}'")
    print(f"  Encoded: {encoded_text}")
    print(f"  Decoded: '{decoded_text}'")
    
    # Check if token is preserved
    if token in decoded_text:
        print(f"  ✓ Token preserved in decoded text")
    else:
        print(f"  ✗ Token NOT preserved in decoded text")

# Test generation scenario
print("\n" + "="*60)
print("Testing generation scenario")
print("="*60)

test_sequences = [
    "<None> apple",
    "<thermal> person", 
    "<radar> car",
    "<audio> bell"
]

for seq in test_sequences:
    encoded = tokenizer.encode(seq, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    
    print(f"\nOriginal: '{seq}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    
    # Check if sensor token is in decoded text
    sensor_found = False
    for sensor in special_tokens:
        if sensor in decoded:
            print(f"  ✓ Found sensor token: {sensor}")
            sensor_found = True
            break
    
    if not sensor_found:
        print(f"  ✗ No sensor token found in decoded text")

# Test what happens with outputs from model.generate()
print("\n" + "="*60)
print("Testing model.generate() output decoding")
print("="*60)

# Simulate model output that includes sensor tokens
for token in special_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    
    # Simulate a generated sequence: [BOS, image_token, instruction_tokens, sensor_token, object_tokens, EOS]
    simulated_output = torch.tensor([[
        tokenizer.bos_token_id if tokenizer.bos_token_id else 0,
        token_id,  # sensor token
        12345,  # random token representing "object"
        tokenizer.eos_token_id if tokenizer.eos_token_id else 2,
    ]])
    
    decoded_with_special = tokenizer.decode(simulated_output[0], skip_special_tokens=False)
    decoded_without_special = tokenizer.decode(simulated_output[0], skip_special_tokens=True)
    
    print(f"\nSimulated output with {token}:")
    print(f"  Token IDs: {simulated_output[0].tolist()}")
    print(f"  Decoded (skip_special_tokens=False): '{decoded_with_special}'")
    print(f"  Decoded (skip_special_tokens=True): '{decoded_without_special}'")
    
    if token in decoded_with_special:
        print(f"  ✓ Sensor token present when skip_special_tokens=False")
    else:
        print(f"  ✗ Sensor token MISSING when skip_special_tokens=False")

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Sensor token IDs: {sensor_token_ids}")
print("\nAll tests complete!")
