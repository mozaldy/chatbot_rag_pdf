import json
import logging
from typing import Any

from llama_index.core.llms import ChatMessage
from app.core.llm_setup import get_fast_llm
from .base import BaseQueryRouter
from .models import RouteResult, RouteIntention
from .prompts import ROUTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class LLMQueryRouter(BaseQueryRouter):
    def __init__(self):
        self.llm = get_fast_llm()

    async def route(self, query: str, chat_history: list[dict] | None = None) -> RouteResult:
        """
        Routes the query using the configured Fast LLM.
        """
        history_str = ""
        if chat_history:
            lines = []
            for msg in chat_history:
                role = msg.get("role", "").strip().lower()
                content = msg.get("content", "").strip()
                if role in ["user", "assistant"] and content:
                    speaker = "User" if role == "user" else "Assistant"
                    lines.append(f"{speaker}: {content}")
            history_str = "\n".join(lines)

        user_content = f"Pertanyaan Terbaru: {query}"
        if history_str:
            user_content = f"Riwayat Obrolan:\n{history_str}\n\n{user_content}"

        messages = [
            ChatMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
        ]

        try:
            # We enforce JSON mode via prompt engineering and post-processing,
            # but some LLM providers support strictly structured output (e.g. valid JSON).
            # For Ollama/General usage, we'll request JSON text and parse it.
            
            # Using complete() or chat()
            response = await self.llm.achat(messages)
            content = response.message.content
            
            # Basic cleanup to handle markdown fences if the LLM includes them
            cleaned_content = self._clean_json_markdown(content)
            
            # Parse JSON
            try:
                data = json.loads(cleaned_content)
                return RouteResult(**data)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Query Router output JSON: {content}")
                # Fallback to SEARCH if parsing fails, or handle gracefully
                return RouteResult(
                    intention=RouteIntention.SEARCH,
                    rewritten_query=query,
                    reasoning="JSON Parsing Failed. Defaulting to SEARCH.",
                    clarification_question=None
                )
            except Exception as e:
                logger.error(f"Validation error for Query Router output: {e}")
                return RouteResult(
                    intention=RouteIntention.SEARCH,
                    rewritten_query=query,
                    reasoning=f"Validation Error: {str(e)}",
                    clarification_question=None
                )

        except Exception as e:
            logger.error(f"Error during query routing: {e}")
            # Fallback
            return RouteResult(
                intention=RouteIntention.SEARCH,
                rewritten_query=query,
                reasoning=f"Internal Router Error: {str(e)}",
                clarification_question=None
            )

    def _clean_json_markdown(self, text: str) -> str:
        """
        Removes ```json ... ``` code blocks if present.
        """
        text = text.strip()
        if text.startswith("```"):
            # Find the first newline
            first_newline = text.find("\n")
            if first_newline != -1:
                # Remove the first line (```json)
                text = text[first_newline+1:]
            
            # Remove the last ```
            if text.endswith("```"):
                text = text[:-3]
        
        return text.strip()
