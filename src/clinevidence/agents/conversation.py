"""General-purpose clinical conversation agent."""

from __future__ import annotations

import logging

from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI

from clinevidence.settings import Settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are ClinEvidence, a knowledgeable clinical assistant for ICU
teams. Your role is to answer general medical questions, explain
clinical concepts, and provide educational information.

Guidelines:
- Answer clearly and concisely using clinical terminology where
  appropriate, with layperson explanations when helpful.
- Always clarify that you are an AI assistant, not a physician.
- For patient-specific clinical decisions, always recommend
  consulting qualified clinicians and current guidelines.
- Do not diagnose individual patients.
- Include a brief disclaimer when discussing treatments or
  medications.
"""


class ConversationAgent:
    """Handles general clinical questions and explanations."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = AzureChatOpenAI(
            azure_deployment=settings.llm_deployment,
            azure_endpoint=settings.azure_endpoint,
            api_key=settings.openai_api_key,
            api_version=settings.openai_api_version,
            temperature=settings.conversation_temperature,
        )

    def respond(
        self,
        query: str,
        chat_history: list[BaseMessage],
    ) -> str:
        """Generate a conversational clinical response.

        Args:
            query: The user's clinical question.
            chat_history: Prior messages in the session.

        Returns:
            AI-generated response string.
        """
        max_history = self._settings.max_conversation_messages
        trimmed_history = chat_history[-max_history:]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]
        for msg in trimmed_history:
            role = (
                "user"
                if msg.__class__.__name__ == "HumanMessage"
                else "assistant"
            )
            messages.append(
                {
                    "role": role,
                    "content": str(msg.content),
                }
            )
        messages.append({"role": "user", "content": query})

        try:
            result = self._llm.invoke(messages)
            answer = str(result.content).strip()
            logger.debug(
                "Conversation response generated",
                extra={"answer_len": len(answer)},
            )
            return answer
        except Exception:
            logger.error("Conversation agent failed", exc_info=True)
            return (
                "I encountered an error generating a response. "
                "Please try again or consult a qualified "
                "clinician directly."
            )
