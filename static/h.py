# Install first: pip install tiktoken

import tiktoken

# 1. Load an encoding
# Use encoding_for_model() to get the right tokenizer for a specific model
encoding = tiktoken.encoding_for_model("gpt-4o")

# Or load an encoding directly by name:
# "o200k_base"  -> GPT-4o family
# "cl100k_base" -> GPT-4, GPT-3.5-turbo, text-embedding-ada-002
# "p50k_base"   -> older GPT-3 models (e.g. text-davinci-003)
# encoding = tiktoken.get_encoding("cl100k_base")

text = "Tokenization is how language models read text."

# 2. Encode text -> list of token IDs
tokens = encoding.encode(text)
print("Token IDs:", tokens)
print("Number of tokens:", len(tokens))

# 3. Decode token IDs -> text
decoded_text = encoding.decode(tokens)
print("Decoded text:", decoded_text)

# 4. Inspect individual tokens (useful for debugging/prompt engineering)
token_strings = [encoding.decode([t]) for t in tokens]
print("Individual tokens:", token_strings)


# --- Common use case: counting tokens before sending to the API ---
def count_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

prompt = "Write a short poem about the ocean."
print(f"This prompt uses {count_tokens(prompt)} tokens.")


# --- Counting tokens for a full chat conversation (with role overhead) ---
def count_chat_tokens(messages, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    tokens_per_message = 3  # rough overhead per message (varies slightly by model)
    total = 0
    for msg in messages:
        total += tokens_per_message
        for value in msg.values():
            total += len(enc.encode(value))
    total += 3  # priming tokens for the reply
    return total

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"},
]
print(f"Chat uses ~{count_chat_tokens(messages)} tokens.")