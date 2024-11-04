from pydantic import BaseModel, Field


class Move(BaseModel):
    color: str
    flags: str
    from_square: str = Field(alias='from')
    to_square: str = Field(alias='to')
    piece: str
    san: str


class ApiRequest(BaseModel):
    fen: str
    player: int = 1
    language: str = 'en'
    next_move: Move | None = None
    history: list[Move] | None = None


class BoardAnalysis(BaseModel):
    winning: int
    centipawn_score: int
    win_chance: float


class ApiResponse(BaseModel):
    message: str
    data: dict | list | None = None

class ChatApiRequest(BaseModel):
    message: str
