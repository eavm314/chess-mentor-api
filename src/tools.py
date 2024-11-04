import requests
from llama_index.core.tools import FunctionTool
from functools import cache
import chess

from src.rags import ChessExpertRAG
from src.prompts import chess_guide_qa_tpl, chess_expert_description
from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata

@cache
def get_stockfish_analysis(fen: str) -> dict:
    print(f"Getting Stockfish analysis for FEN: {fen}")
    response = requests.post("https://chess-api.com/v1", {"fen": fen})
    return response.json()


from src.config import get_agent_settings

SETTINGS = get_agent_settings()


chess_expert_tool = QueryEngineTool(
    query_engine=ChessExpertRAG(
        store_path=SETTINGS.store_path,
        data_dir=SETTINGS.docs_path,
        qa_prompt_tpl=chess_guide_qa_tpl,
    ).get_chat_engine(),
    metadata=ToolMetadata(
        name="chess_expert", description=chess_expert_description, return_direct=False
    ),
)

def get_best_move(fen: str) -> dict:
    """
    Get the best move based on the current board state.
    Args:
        - fen (str): FEN notation of the current board state.
    Returns:
        - dict: The best move to make and addtional information.
    """
    print(f"Getting best move for FEN: {fen}")

    piece_types = {
        "p": "pawn",
        "n": "knight",
        "b": "bishop",
        "r": "rook",
        "q": "queen",
        "k": "king"
    }

    move_types = {
        "n": "non-capture",
        "b": "pawn push of two squares",
        "e": "en passant capture",
        "c": "standard capture",
        "p": "promotion",
        "k": "kingside castling",
        "q": "queenside castling"
    }

    analysis = get_stockfish_analysis(fen)

    text = analysis.get("text")
    move = analysis.get("move")

    from_square = analysis.get("from")
    to_square = analysis.get("to")

    piece = analysis.get("piece")
    captured = analysis.get("captured")
    promotion = analysis.get("promotion")

    flags = analysis.get("flags")
    move_type = [move_types[flag] for flag in list(flags)]

    return {
        "text": text,
        "move": move,
        "from_square": from_square,
        "to_square": to_square,
        "piece_type": piece_types[piece],
        "move_type": move_type,
        "captured": piece_types[captured] if captured in piece_types else None,
        "promotion": piece_types[promotion] if promotion in piece_types else None,
    }


def get_piece_info(board: chess.Board, square: chess.Square) -> dict:
    piece = board.piece_at(square)
    return {
        "square": square,
        "position": chess.square_name(square),
        "piece": piece.symbol(),
        "color": "white" if piece.color else "black",
    }


def analize_board(fen: str) -> dict:
    """
    Analize the current board state to provide valuable insights.
    Args:
        - fen (str): FEN notation of the current board state.
    Returns:
        - dict: The analysis of the board state.
    Fields:
        - turn (str): The color of the player to move.
        - centipawn_score (int): The centipawn score of the position.
        - win_chance (float): The win chance of the position (<50 black & >50 white).
        - checkers (list[dict]): The pieces that are delivering check.
        - material (dict): The material count for both sides.
        - pieces_info (list[dict]): The information of each piece on the board.
        - pawn_structure (dict): The pawn structure for both sides.
    """
    print(f"Analizing position for FEN: {fen}")

    analysis = get_stockfish_analysis(fen)
    centipawn_score = analysis["centipawns"]
    win_chance = analysis["winChance"]

    board = chess.Board(fen)
    pieces = board.piece_map()

    piece_values = {
        "p": 1,
        "n": 3,
        "b": 3,
        "r": 5,
        "q": 9,
    }

    material = {
        "white": {"pieces":{}},
        "black": {"pieces":{}}
    }

    for square, piece in pieces.items():
        color = "white" if piece.color else "black"
        symbol = piece.symbol()
        if symbol in ["K", "k"]:
            continue
        if symbol in material[color]["pieces"]:
            material[color]["pieces"][symbol]["count"] += 1
        else:
            piece_info = {
                "value": piece_values[symbol.lower()],
                "count": 1,
            }

            material[color]["pieces"][symbol] = piece_info

    for color in material:
        material[color]["total"] = sum(
            [piece["value"] * piece["count"] for piece in material[color]["pieces"].values()])

    checkers = board.checkers()

    attackers = {square: [] for square in chess.SQUARES}
    attacked = {square: [] for square in chess.SQUARES}

    for square in pieces:
        attacks = board.attackers(not pieces[square].color, square)
        for attacker in attacks:
            attacked[attacker].append(square)
            attackers[square].append(attacker)

    pawn_structure = {
        "white": [chess.square_file(pawn) for pawn in board.pieces(chess.PAWN, chess.WHITE)],
        "black": [chess.square_file(pawn) for pawn in board.pieces(chess.PAWN, chess.BLACK)],
    }

    pieces_info = [get_piece_info(board, square) for square in pieces.keys()]
    for pi in pieces_info:
        pi["attacked_by"] = [get_piece_info(board, square) for square in attackers[pi["square"]]]
        pi["attacking"] = [get_piece_info(board, square) for square in attacked[pi["square"]]]

    return {
        "turn": "white" if board.turn else "black",
        "centipawn_score": centipawn_score,
        "win_chance": win_chance,
        "checkers": [get_piece_info(board, square) for square in checkers],
        "material": material,
        "pieces_info": pieces_info,
        "pawn_structure": pawn_structure,
    }


def analize_move(fen: str, move: str) -> dict:
    """
    Analize the move made in the current board state.
    Args:
        - fen (str): FEN notation of the current board state.
        - move (str): The move made.
    Returns:
        - dict: The analysis of the move made.
    """
    print(f"Analizing move {move} from FEN: {fen}")

    prev_analysis = analize_board(fen)
    board = chess.Board(fen)
    board.push_san(move)
    new_fen = board.fen()
    new_analysis = analize_board(new_fen)

    centipawn_score_diff = new_analysis["centipawn_score"] - prev_analysis["centipawn_score"]
    win_chance_diff = new_analysis["win_chance"] - prev_analysis["win_chance"]

    material_diff = {
        "white": new_analysis["material"]["white"]["total"] - prev_analysis["material"]["white"]["total"],
        "black": new_analysis["material"]["black"]["total"] - prev_analysis["material"]["black"]["total"]
    }

    return {
        "centipawn_score_diff": centipawn_score_diff,
        "win_chance_diff": win_chance_diff,
        "material_diff": material_diff,
        "prev_analysis": prev_analysis,
        "new_analysis": new_analysis,
    }

def analyze_player(player: int, moves: list[str]) -> list[dict]:
    """
    Simulate the game to provide insights on the game state for each movement.
    Args:
        - player (int): The player to analyze (0: black, 1: white).
        - moves (list[str]): The list of moves made in algebraic notation.
    Returns:
        - list[dict]: The winning state for each movement.
    """
    print(f"Analizing player movements: {moves}")
    game = chess.Board()
    analysis_res = []
    for move in moves:
        game.push_san(move)
        if game.turn == player:
            continue
        analysis = get_stockfish_analysis(game.fen())
        centipawn_score = analysis["centipawns"]
        win_chance = analysis["winChance"]
        if player == 0:
            win_chance = 100 - win_chance
        analysis_res.append({
            "move": move,
            "centipawn_score": centipawn_score,
            "win_chance": win_chance,
            "state": "winning" if win_chance > 75 else "lossing" if win_chance < 25 else "even",
        })

    return analysis

best_move_tool = FunctionTool.from_defaults(
    fn=get_best_move, return_direct=False)
analize_board_tool = FunctionTool.from_defaults(
    fn=analize_board, return_direct=False)
analize_move_tool = FunctionTool.from_defaults(
    fn=analize_move, return_direct=False)
analize_player_tool = FunctionTool.from_defaults(
    fn=analyze_player, return_direct=False)