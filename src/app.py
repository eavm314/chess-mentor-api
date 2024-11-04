from fastapi import FastAPI, Depends
from llama_index.core.agent import ReActAgent
from functools import cache

from src.models import ApiResponse, ChatApiRequest
from src.agent import ChessAgent

from src.models import ApiRequest, ApiResponse, Move
from src.agent import ChessAgent

from src.tools import get_best_move
from fastapi.middleware.cors import CORSMiddleware



@cache
def get_agent() -> ReActAgent:
    return ChessAgent().get_agent()



app = FastAPI(title="Chess Mentor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def get_health():
    return "Server is running"


@app.post("/best-move")
def calculate_best_move(req: ApiRequest, agent: ReActAgent = Depends(get_agent)):
    best_move=get_best_move(fen=req.fen)
    prompt = f"""
        Given this position in the chessboard in FEN notation: "{req.fen}".
        Can you provide the next best move I can do, 
        then explain it as a chess master that is teaching me how to improve my games.
        Summarize your answer in one short paragraph in the following language: {req.language}.
    """
    response = agent.query(prompt)
    return ApiResponse(
        message="Best move calculated succesfully",
        agent_response=str(response),
        data=best_move
    )

@app.post("/state")
def calculate_board_state(req: ApiRequest, agent: ReActAgent = Depends(get_agent)):
    prompt = f"""
        Given this position in the chessboard in FEN notation: "{req.fen}".
        Provide an analisys of the current board state.

        Once you have the information you need, please explain it as a chess master that is teaching me how to improve my games.
        I want to understand the board state and how to take advantage of it.
        Please tell me the strategy I should follow to win the game.

        Summarize your answer in a small list containing the main insights of the game, make sure each line is a relevant insight only, 
        not trivial information, and each item should be at most 2 lines, 
        I only need information that is relevant to my chess learning process, so avoid trivial information.
        Don't use markdown or any other special formatting, just plain text.
        Don't use specific values, just general insights about the board state.
        Use this language for your answer: {req.language}.
    """
    response = agent.query(prompt)
    return ApiResponse(
        message="Board state generated succesfully",
        agent_response=str(response),
    )

@app.post("/player")
def analyze_player(req: ApiRequest, agent: ReActAgent = Depends(get_agent)):
    player = "white" if req.player == 1 else "black"
    prompt = f"""
        Given the following moves: {req.history}.
        Analyze the game state for each movement of the player {player} represented by {req.player},
        for this use the san moves: {[move.san for move in req.history]}.
        Provide a short paragraph explaining how the player is playing during the match.
        Don't use markdown or any other special formatting, just plain text.
        Don't use specific values, just general insights about the player's strategy.
        Use this language for your answer: {req.language}.
    """
    response = agent.query(prompt)
    return ApiResponse(
        message="Match analysis generated succesfully",
        agent_response=str(response),
    )

@app.post("/chat")
def chat(req: ChatApiRequest, agent: ReActAgent = Depends(get_agent)):
    response = agent.chat(req.message)
    return ApiResponse(
        message="Chat response generated succesfully",
        data={"response": response}
    )


