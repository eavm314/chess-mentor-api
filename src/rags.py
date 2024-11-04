import os

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
    PromptTemplate,
)

from llama_index.llms.openai import OpenAI

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.prompts import chess_guide_qa_tpl
from src.config import get_agent_settings

SETTINGS = get_agent_settings()

class ChessExpertRAG:
    def __init__(
        self,
        store_path: str,
        data_dir: str | None = None,
        qa_prompt_tpl: PromptTemplate | None = None,
    ):
        self.store_path = store_path

        if not os.path.exists(store_path) and data_dir is not None:
            self.index = self.ingest_data(store_path, data_dir)
        else:
            self.index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=store_path)
            )

        self.qa_prompt_tpl = qa_prompt_tpl

    def ingest_data(self, store_path: str, data_dir: str) -> VectorStoreIndex:
        documents = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=store_path)
        return index

    def get_chat_engine(self):
        chat_engine = self.index.as_chat_engine()

        return chat_engine



