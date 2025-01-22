import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from chess_transformer import ChessTokenizer  # Import from chess_transformer.py

def analyze_token_lengths():
    # Initialize tokenizer
    tokenizer = ChessTokenizer()
    
    # Lists to store statistics
    token_lengths = []
    num_over_512 = 0
    num_over_1024 = 0
    max_length = 0
    
    # Process games
    print("Analyzing token lengths...")
    with open("processed_games_10k.json", "r") as f:
        games = json.load(f)
        for game in tqdm(games):
            tokens = tokenizer.encode_with_special_tokens(game)
            length = len(tokens)
            token_lengths.append(length)
            max_length = max(max_length, length)
            if length > 512:  # Current context length
                num_over_512 += 1
            if length > 1024:  # Doubled context length
                num_over_1024 += 1
    
    # Calculate statistics
    avg_length = np.mean(token_lengths)
    median_length = np.median(token_lengths)
    percentile_95 = np.percentile(token_lengths, 95)
    percentile_99 = np.percentile(token_lengths, 99)
    
    print(f"\nToken Length Statistics:")
    print(f"Average length: {avg_length:.2f}")
    print(f"Median length: {median_length:.2f}")
    print(f"95th percentile: {percentile_95:.2f}")
    print(f"99th percentile: {percentile_99:.2f}")
    print(f"Max length: {max_length}")
    print(f"\nContext Length Analysis:")
    print(f"Games over 512 tokens: {num_over_512} ({(num_over_512/len(games))*100:.2f}%)")
    print(f"Games over 1024 tokens: {num_over_1024} ({(num_over_1024/len(games))*100:.2f}%)")
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(token_lengths, bins=50, alpha=0.75)
    plt.axvline(x=512, color='r', linestyle='--', label='Current Limit (512)')
    plt.axvline(x=1024, color='g', linestyle='--', label='Double Limit (1024)')
    plt.xlabel('Token Length')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Game Token Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('token_length_distribution.png')
    plt.close()

if __name__ == "__main__":
    analyze_token_lengths() 