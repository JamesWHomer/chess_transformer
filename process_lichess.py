import chess.pgn
import io
import json
import multiprocessing as mp
from typing import List, Optional
from tqdm import tqdm
import requests
import zstandard
from chess_transformer import ChessTokenizer  # Import from chess_transformer.py

def download_and_decompress(url: str, chunk_size: int = 8192):
    """Download and decompress a .zst file from Lichess."""
    print(f"Downloading {url}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # First download the compressed data
    compressed_data = io.BytesIO()
    with tqdm(total=total_size, unit='iB', unit_scale=True, desc='Downloading') as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            compressed_data.write(chunk)
            pbar.update(len(chunk))
    
    # Reset buffer position
    compressed_data.seek(0)
    
    # Then decompress it
    print("\nDecompressing data...")
    dctx = zstandard.ZstdDecompressor()
    output_file = url.split('/')[-1][:-4]
    
    with open(output_file, 'wb') as f_out:
        with tqdm(desc='Decompressing', unit='iB', unit_scale=True) as pbar:
            with dctx.stream_reader(compressed_data) as reader:
                while True:
                    chunk = reader.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"\nDecompressed file saved as {output_file}")
    return output_file

def convert_standard_to_long_algebraic(move: chess.Move, board: chess.Board) -> str:
    """
    Convert a chess.Move to our custom long algebraic notation.
    Examples:
    - e4 -> Pe2me4 (pawn moves)
    - Nf3 -> Ng1mf3 (piece moves)
    - exd5 -> Pe4xd5 (pawn captures)
    - O-O -> O-O (kingside castling)
    - O-O-O -> O-O-O (queenside castling)
    - e8=Q -> Pe7me8Q (pawn promotion)
    """
    # Get the UCI format which gives us the from and to squares
    uci = move.uci()
    from_square = uci[:2]
    to_square = uci[2:4]
    
    # Handle castling
    if board.is_castling(move):
        if to_square == 'g1':  # White kingside
            return "O-O"
        elif to_square == 'c1':  # White queenside
            return "O-O-O"
        elif to_square == 'g8':  # Black kingside
            return "O-O"
        elif to_square == 'c8':  # Black queenside
            return "O-O-O"
    
    # Handle en passant captures specifically
    if board.is_en_passant(move):
        piece_symbol = 'P'  # En passant is always a pawn capture
        return f"{piece_symbol}{from_square}x{to_square}"
    
    # Get piece symbol
    piece = board.piece_at(chess.parse_square(from_square))
    if piece is None:  # Defensive check
        return None
    piece_symbol = piece.symbol().upper()
    
    # Handle captures
    is_capture = board.is_capture(move)
    move_symbol = 'x' if is_capture else 'm'
    
    # Handle promotion - use the piece symbol directly
    promotion = ''
    if move.promotion is not None:
        promotion_piece = chess.Piece(move.promotion, chess.WHITE).symbol().upper()
        promotion = f"={promotion_piece}"
    
    # Build the move string
    result = f"{piece_symbol}{from_square}{move_symbol}{to_square}{promotion}"
    
    # Add check/checkmate symbols
    board.push(move)
    if board.is_checkmate():
        result += '#'
    elif board.is_check():
        result += '+'
    board.pop()
    
    return result

def process_game(game: chess.pgn.Game) -> Optional[str]:
    """
    Process a single game and convert it to our custom format.
    Returns None if the game is invalid, incomplete, or exceeds context length.
    """
    try:
        processed_moves = ["<START>"]
        board = game.board()
        
        for move in game.mainline_moves():
            # Add turn marker
            processed_moves.append("<WHITE>" if board.turn else "<BLACK>")
            
            # Convert and add move
            move_str = convert_standard_to_long_algebraic(move, board)
            processed_moves.append(move_str)
            
            # Early length check - each token is at least 1 character, so if we exceed
            # context length in raw characters, we'll definitely exceed in tokens
            if len("".join(processed_moves)) >= 1024:  # CONTEXT_LENGTH
                return None
            
            # Make the move on our board
            board.push(move)
        
        # Add game result
        result = game.headers.get("Result", "*")
        if result == "1-0":
            processed_moves.append("<WHITE_WINS>")
        elif result == "0-1":
            processed_moves.append("<BLACK_WINS>")
        else:
            processed_moves.append("<DRAW>")
        
        game_str = "".join(processed_moves)
        
        # Final length check using actual tokenization
        tokenizer = ChessTokenizer()  # Use shared tokenizer
        tokens = tokenizer.encode_with_special_tokens(game_str)
        if len(tokens) > 1024:  # CONTEXT_LENGTH
            return None
            
        return game_str
    
    except Exception as e:
        print(f"Error processing game: {str(e)}")
        return None

def process_game_from_string(game_str: str) -> Optional[str]:
    """Process a game from its PGN string representation."""
    try:
        game = chess.pgn.read_game(io.StringIO(game_str))
        return process_game(game)
    except Exception as e:
        print(f"Error processing game: {str(e)}")
        return None

def read_games_in_chunks(pgn_path: str, chunk_size: int = 1000):
    """Read games from PGN file in chunks."""
    current_game = []
    chunk = []
    
    with open(pgn_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('[Event ') and current_game:
                # End of a game
                chunk.append('\n'.join(current_game))
                current_game = [line]
                
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            else:
                current_game.append(line)
    
    # Add the last game
    if current_game:
        chunk.append('\n'.join(current_game))
    
    # Yield remaining games
    if chunk:
        yield chunk

def process_pgn_file(pgn_path: str, output_path: str, max_games: Optional[int] = None):
    """
    Process a PGN file and save games in our custom format using multiprocessing.
    Args:
        pgn_path: Path to input PGN file
        output_path: Path to save processed games
        max_games: Maximum number of games to process (None for all games)
    """
    # Count total games first for progress bar
    print("Counting total games...")
    total_games = sum(1 for line in open(pgn_path) if line.startswith('[Event '))
    if max_games:
        total_games = min(total_games, max_games)
    
    print(f"\nProcessing up to {total_games} games from {pgn_path}...")
    
    # Initialize multiprocessing pool
    pool = mp.Pool(processes=mp.cpu_count())
    
    processed_games = []
    num_games = 0
    
    # Process games in chunks
    with tqdm(total=total_games, desc="Processing games", unit="games") as pbar:
        
        for chunk in read_games_in_chunks(pgn_path):
            if max_games and num_games >= max_games:
                break
            
            # Process chunk in parallel
            results = pool.map(process_game_from_string, chunk)
            valid_games = [g for g in results if g is not None]
            
            # Update progress
            num_processed = len(valid_games)
            num_games += num_processed
            pbar.update(num_processed)
            
            # Save batch
            if valid_games:
                processed_games.extend(valid_games)
    
    pool.close()
    pool.join()
    
    print(f"\nSuccessfully processed {num_games} games")
    print(f"Saving games to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(processed_games, f)
    
    return processed_games

if __name__ == "__main__":
    # Example usage
    lichess_url = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-04.pgn.zst"
    pgn_file = download_and_decompress(lichess_url)
    processed_games = process_pgn_file(pgn_file, "processed_games.json", max_games=100_000) 