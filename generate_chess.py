import torch
import torch.nn as nn
from chess_transformer import ChessTransformer, ChessTokenizer, generate_move

# Model Configuration (must match training config)
D_MODEL = 512      # Model size
N_HEADS = 8        # Heads
N_LAYERS = 6       # Layers
DROPOUT = 0.1      # Dropout
CONTEXT_LENGTH = 1024  # Context length

def load_model(checkpoint_path: str):
    """Load the trained chess model from a checkpoint."""
    tokenizer = ChessTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ChessTransformer(
        vocab_size=len(tokenizer.token_to_id),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model('chess_model_checkpoint.pt')
    
    # Example game sequence
    game = "<START><WHITE>Pe2me4<BLACK>Pe7me5<WHITE>"
    
    # Generate continuation
    continuation = generate_move(model, tokenizer, game, temperature=0.8)
    print(continuation) 