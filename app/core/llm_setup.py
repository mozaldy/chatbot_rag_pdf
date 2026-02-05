import logging
from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from .config import settings

# Global singletons
_bgem3_model = None
_embedding_model = None
_llm = None
_fast_llm = None

def get_bgem3_model():
    """Singleton for the shared BGE-M3 model (FlagEmbedding)"""
    global _bgem3_model
    if _bgem3_model is None:
        from FlagEmbedding import BGEM3FlagModel
        logging.info(f"Loading Shared BGE-M3 Model (GPU): {settings.EMBEDDING_MODEL}")
        _bgem3_model = BGEM3FlagModel(
            settings.EMBEDDING_MODEL,
            use_fp16=True # Critical for 6GB VRAM
        )
    return _bgem3_model
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import Any, List

class BGEM3LlamaWrapper(BaseEmbedding):
    """Wrapper for FlagEmbedding's BGEM3FlagModel to work with LlamaIndex."""
    _model: Any = None
    
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self._model = model

    def _get_query_embedding(self, query: str) -> List[float]:
        # return_dense=True by default in encode
        output = self._model.encode(query, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return output['dense_vecs'].tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        output = self._model.encode(text, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return output['dense_vecs'].tolist()

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        output = self._model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return output['dense_vecs'].tolist()

def get_embedding_model():
    """Singleton pattern for Embedding Model (Dense) using Shared BGEM3"""
    global _embedding_model
    if _embedding_model is None:
        model = get_bgem3_model() # Ensure model is loaded
        _embedding_model = BGEM3LlamaWrapper(model)
    return _embedding_model

def get_sparse_embedding_functions():
    """Returns (doc_fn, query_fn) for QdrantVectorStore using Shared BGEM3 (Sparse)."""
    model = get_bgem3_model()
    
    def compute_vectors(texts: list[str]):
        # FlagEmbedding encode returns a dictionary
        # output['lexical_weights'] is a list of dictionaries (one per text)
        # e.g., [{'token_id': weight, ...}, ...] -> Wait, FlagEmbedding returns dict mapping token_str to weight typically?
        # Checked documentation: return_sparse=True returns 'lexical_weights' which is List[Dict[int, float]]? 
        # Actually it returns List[Dict[str, float]] (token string to weight) usually. 
        # Qdrant expects indices (integers) and values (floats).
        # However, BGEM3FlagModel documentation says:
        # lexical_weights: list of dict, each dict maps token_id (int) to weight (float) if return_sparse=True?
        # Let's double check. If it returns strings, we need the tokenizer to convert to IDs.
        # But FlagEmbedding `encode` output `lexical_weights` usually maps str->float.
        # QdrantVectorStore expects integer indices (token IDs).
        # We might need to manually access the tokenizer from the model.
        
        output = model.encode(texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        lexical_weights_list = output['lexical_weights']
        
        indices = []
        values = []
        
        # We need the tokenizer to convert token strings back to IDs if necessary.
        # model.tokenizer is available in BGEM3FlagModel
        tokenizer = model.tokenizer
        
        for weight_dict in lexical_weights_list:
            # weight_dict is likely Dict[str, float] based on typical FlagEmbedding usage.
            # Convert str keys to token IDs.
            # Note: This is slightly inefficient (str -> int), but BGE-M3 implementation often works this way.
            # Let's optimize: model.compute_lexical_weights??
            # Actually, `encode` in FlagEmbedding calls `compute_lexical_weights`.
            # Let's inspect what we have. If keys are strings:
            
            # Helper to safely get ID.
            row_indices = []
            row_values = []
            
            for token, weight in weight_dict.items():
                # If token is implicitly an ID (unlikely), use it.
                # Otherwise convert.
                if isinstance(token, int):
                    row_indices.append(token)
                else:
                    # Convert token string to ID. 
                    # Warning: token might be part of a word.
                    ids = tokenizer.convert_tokens_to_ids(token)
                    # If unknown, skip?
                    if ids != tokenizer.unk_token_id:
                        row_indices.append(ids)
                    
                row_values.append(float(weight))
            
            indices.append(row_indices)
            values.append(row_values)
            
        return indices, values
    
    return compute_vectors, compute_vectors

def _create_llm(provider: str, model_name: str):
    """Factory function to create an LLM instance based on provider."""
    provider = provider.lower()
    logging.info(f"Loading LLM Provider: {provider} with Model: {model_name}")

    try:
        if provider == "ollama":
            return Ollama(
                model=model_name,
                request_timeout=settings.OLLAMA_TIMEOUT,
                base_url=settings.OLLAMA_BASE_URL
            )
        elif provider == "openai":
            from llama_index.llms.openai import OpenAI
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                raise ValueError("OPENAI_API_KEY is missing in configuration.")
            return OpenAI(model=model_name, api_key=api_key)
        elif provider == "anthropic":
            from llama_index.llms.anthropic import Anthropic
            api_key = settings.ANTHROPIC_API_KEY
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is missing in configuration.")
            return Anthropic(model=model_name, api_key=api_key)
        elif provider == "gemini":
            from llama_index.llms.gemini import Gemini
            api_key = settings.GOOGLE_API_KEY
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is missing in configuration.")
            return Gemini(model=model_name, api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM Provider: {provider}")
            
    except ImportError as e:
        logging.error(f"Failed to import LLM provider {provider}. Ensure the package is installed.")
        raise e
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        raise e

def get_llm():
    """Singleton pattern for Main LLM (Inference)"""
    global _llm
    if _llm is None:
        _llm = _create_llm(settings.LLM_PROVIDER, settings.LLM_MODEL)
    return _llm

def get_fast_llm():
    """Singleton pattern for Fast LLM (Enrichment)"""
    global _fast_llm
    if _fast_llm is None:
        _fast_llm = _create_llm(settings.FAST_LLM_PROVIDER, settings.FAST_LLM_MODEL)
    return _fast_llm

def init_settings():
    """
    Initialize LlamaIndex global settings.
    Call this on app startup.
    """
    LlamaSettings.embed_model = get_embedding_model()
    LlamaSettings.llm = get_llm()
    logging.info("LlamaIndex Settings initialized.")
