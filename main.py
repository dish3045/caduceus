import torch
from transformers import AutoTokenizer, AutoModel

def main():
    model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    #temporarily hardcoding input until we understand how the model runs
    dna_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    
    print("Tokenizing DNA sequence...")
    inputs = tokenizer(dna_sequence, return_tensors="pt", padding=True, truncation=True)
    # Mo
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print("Getting model output...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Input sequence: {dna_sequence}")
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Output shape: {outputs.last_hidden_state.shape}")
    print(f"Output (first 5 values): {outputs.last_hidden_state[0, 0, :5]}")
    
    embeddings = outputs.last_hidden_state
    print(f"Got embeddings with shape: {embeddings.shape}")

if __name__ == "__main__":
    main()