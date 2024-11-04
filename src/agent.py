from llama_index.core import PromptTemplate, Settings
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.prompts import magnus_carlsen_prompt_text
from src.config import get_agent_settings
from src.tools import best_move_tool, analize_move_tool, analize_board_tool, analize_player_tool, chess_expert_tool

SETTINGS = get_agent_settings()

llm = OpenAI(model=SETTINGS.openai_model, api_key=SETTINGS.openai_api_key)
embed_model = HuggingFaceEmbedding(model_name=SETTINGS.hf_embeddings_model)
Settings.embed_model = embed_model
Settings.llm = llm


    
class ChessAgent:
    def __init__(self):
        self.agent = ReActAgent.from_tools(
            [
                best_move_tool,
                analize_move_tool,
                analize_board_tool,
                analize_player_tool,
                chess_expert_tool,
            ],
            verbose=True,
        )

        system_prompt = PromptTemplate(magnus_carlsen_prompt_text)
        self.agent.update_prompts({"agent_worker:system_prompt": system_prompt})

    def get_agent(self) -> ReActAgent:
        return self.agent
