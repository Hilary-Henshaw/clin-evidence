"""LangGraph multi-agent workflow orchestrator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from clinevidence.agents.conversation import ConversationAgent
from clinevidence.agents.imaging.modality_detector import (
    ModalityDetector,
)
from clinevidence.agents.imaging.router import ImagingRouter
from clinevidence.agents.rag.pipeline import KnowledgeBase
from clinevidence.agents.safety_filter import SafetyFilter
from clinevidence.agents.search.search_processor import (
    EvidenceSearchProcessor,
)
from clinevidence.agents.state import WorkflowState
from clinevidence.settings import Settings

logger = logging.getLogger(__name__)

_IMAGING_AGENTS = frozenset({"BRAIN_MRI", "CHEST_XRAY", "SKIN_LESION"})
_MODALITY_TO_AGENT: dict[str, str] = {
    "BRAIN_MRI": "BRAIN_MRI",
    "CHEST_XRAY": "CHEST_XRAY",
    "SKIN_LESION": "SKIN_LESION",
}


class WorkflowOrchestrator:
    """LangGraph-based multi-agent orchestrator for ClinEvidence."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._safety = SafetyFilter(settings)
        self._conversation = ConversationAgent(settings)
        self._knowledge_base = KnowledgeBase(settings)
        self._search_processor = EvidenceSearchProcessor(settings)
        self._modality_detector = ModalityDetector(settings)
        self._imaging_router = ImagingRouter(settings)
        self._decision_llm = self._build_decision_llm()
        self._memory = MemorySaver()
        self._graph = self._build_graph()

    def _build_decision_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_deployment=self._settings.llm_deployment,
            azure_endpoint=self._settings.azure_endpoint,
            api_key=self._settings.openai_api_key,
            api_version=self._settings.openai_api_version,
            temperature=self._settings.decision_temperature,
        )

    # ── Node: assess_input ────────────────────────────────────────
    def _assess_input(self, state: WorkflowState) -> dict[str, Any]:
        """Detect images and validate input safety."""
        last_msg = state["messages"][-1]
        query = str(last_msg.content)

        allowed, reason = self._safety.check_input(query)
        if not allowed:
            logger.warning(
                "Input blocked by safety filter",
                extra={"reason": reason},
            )
            return {
                "response": reason,
                "selected_agent": "SAFETY_BLOCK",
                "error": "INPUT_BLOCKED",
            }

        image_path = state.get("image_path")
        if image_path:
            try:
                detection = self._modality_detector.detect(Path(image_path))
                return {
                    "image_type": detection["image_type"],
                    "routing_confidence": detection["confidence"],
                }
            except Exception:
                logger.warning(
                    "Modality detection failed",
                    exc_info=True,
                )
                return {"image_type": "OTHER"}

        return {}

    # ── Node: select_agent ────────────────────────────────────────
    def _select_agent(self, state: WorkflowState) -> dict[str, Any]:
        """Use LLM to route the query to the correct agent."""
        if state.get("error") == "INPUT_BLOCKED":
            return {}

        image_type = state.get("image_type")
        if image_type and image_type in _MODALITY_TO_AGENT:
            agent = _MODALITY_TO_AGENT[image_type]
            logger.info(
                "Agent selected by image type",
                extra={
                    "agent": agent,
                    "image_type": image_type,
                },
            )
            return {"selected_agent": agent}

        last_msg = state["messages"][-1]
        query = str(last_msg.content)

        prompt = (
            "You are a medical AI routing system. "
            "Classify the following clinical query and respond "
            "with a JSON object: "
            '{"agent": "<AGENT>", "confidence": <0.0-1.0>}\n\n'
            "Available agents:\n"
            "- CONVERSATION: general medical questions, "
            "greetings, explanations\n"
            "- KNOWLEDGE_BASE: evidence retrieval from clinical "
            "guidelines, research papers\n"
            "- WEB_EVIDENCE: real-time clinical data, recent "
            "studies, current guidelines\n\n"
            f"Query: {query}"
        )

        try:
            result = self._decision_llm.invoke(prompt)
            parsed = json.loads(str(result.content))
            agent = str(parsed.get("agent", "CONVERSATION"))
            confidence = float(parsed.get("confidence", 0.5))
        except Exception:
            logger.warning(
                "Agent selection failed, defaulting to CONVERSATION",
                exc_info=True,
            )
            agent = "CONVERSATION"
            confidence = 0.5

        logger.info(
            "Agent selected",
            extra={
                "agent": agent,
                "confidence": confidence,
            },
        )
        return {
            "selected_agent": agent,
            "routing_confidence": confidence,
        }

    # ── Node: run_conversation ────────────────────────────────────
    def _run_conversation(self, state: WorkflowState) -> dict[str, Any]:
        query = str(state["messages"][-1].content)
        history = list(state["messages"][:-1])
        answer = self._conversation.respond(query, history)
        return {
            "response": answer,
            "messages": [AIMessage(content=answer)],
        }

    # ── Node: run_knowledge_base ──────────────────────────────────
    def _run_knowledge_base(self, state: WorkflowState) -> dict[str, Any]:
        query = str(state["messages"][-1].content)
        history = list(state["messages"][:-1])
        result = self._knowledge_base.query(query, history)

        if result["confidence"] < self._settings.kb_min_confidence:
            logger.info(
                "KB confidence below threshold, escalating to web search",
                extra={"confidence": result["confidence"]},
            )
            return {"selected_agent": "WEB_EVIDENCE"}

        sources = result.get("sources", [])
        return {
            "response": result["answer"],
            "sources": list(sources),
            "routing_confidence": result["confidence"],
            "messages": [AIMessage(content=result["answer"])],
        }

    # ── Node: run_web_evidence ────────────────────────────────────
    def _run_web_evidence(self, state: WorkflowState) -> dict[str, Any]:
        query = str(state["messages"][-1].content)
        history = list(state["messages"][:-1])
        answer, sources = self._search_processor.process(query, history)
        return {
            "response": answer,
            "sources": sources,
            "messages": [AIMessage(content=answer)],
        }

    # ── Node: run_brain_mri ───────────────────────────────────────
    def _run_brain_mri(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_imaging(state, "BRAIN_MRI")

    # ── Node: run_chest_xray ──────────────────────────────────────
    def _run_chest_xray(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_imaging(state, "CHEST_XRAY")

    # ── Node: run_skin_lesion ─────────────────────────────────────
    def _run_skin_lesion(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_imaging(state, "SKIN_LESION")

    def _run_imaging(
        self,
        state: WorkflowState,
        modality: str,
    ) -> dict[str, Any]:
        """Run the appropriate imaging analyser."""
        image_path = state.get("image_path")
        if not image_path:
            return {
                "error": "NO_IMAGE",
                "response": "No image provided.",
            }
        try:
            result = self._imaging_router.route_and_analyse(
                Path(image_path), modality
            )
            display_name = modality.replace("_", " ")
            summary = (
                f"**{display_name} Analysis**\n\n"
                f"**Diagnosis:** {result['diagnosis']}\n"
                f"**Confidence:** {result['confidence']:.1%}"
                f"\n\n"
                f"{result['explanation']}\n\n"
                f"*Model: {result['model_name']}*\n\n"
                f"> This analysis is AI-generated and must be "
                f"reviewed by a qualified radiologist or clinician"
                f" before clinical use."
            )
            return {
                "response": summary,
                "requires_validation": True,
                "routing_confidence": result["confidence"],
                "messages": [AIMessage(content=summary)],
            }
        except FileNotFoundError as exc:
            msg = (
                f"Model weights not found for {modality}. "
                f"Please download and place the model at the "
                f"configured path. Details: {exc}"
            )
            logger.error(msg)
            return {
                "error": "MODEL_NOT_FOUND",
                "response": msg,
            }
        except Exception:
            logger.error(
                "Imaging analysis failed",
                extra={"modality": modality},
                exc_info=True,
            )
            return {
                "error": "IMAGING_FAILED",
                "response": ("Image analysis encountered an error."),
            }

    # ── Node: await_validation ────────────────────────────────────
    def _await_validation(self, state: WorkflowState) -> dict[str, Any]:
        """Interrupt graph for human validation."""
        approved = interrupt(
            {
                "action": "validate_analysis",
                "session_id": state["session_id"],
                "preview": state.get("response", ""),
            }
        )
        return {"validation_approved": bool(approved)}

    # ── Node: apply_output_safety ─────────────────────────────────
    def _apply_output_safety(self, state: WorkflowState) -> dict[str, Any]:
        """Filter the final response through output guardrails."""
        response = state.get("response", "")
        if not response:
            return {}

        allowed, reason = self._safety.check_output(response)
        if not allowed:
            logger.warning(
                "Output blocked by safety filter",
                extra={"reason": reason},
            )
            return {
                "response": (
                    "I'm unable to provide that response as it "
                    "may not meet clinical safety standards. "
                    "Please consult a qualified clinician."
                )
            }
        return {}

    # ── Routing functions ─────────────────────────────────────────
    def _route_after_assessment(self, state: WorkflowState) -> str:
        if state.get("error") == "INPUT_BLOCKED":
            return "apply_output_safety"
        return "select_agent"

    def _route_after_selection(self, state: WorkflowState) -> str:
        agent = str(state.get("selected_agent", "CONVERSATION"))
        mapping: dict[str, str] = {
            "CONVERSATION": "run_conversation",
            "KNOWLEDGE_BASE": "run_knowledge_base",
            "WEB_EVIDENCE": "run_web_evidence",
            "BRAIN_MRI": "run_brain_mri",
            "CHEST_XRAY": "run_chest_xray",
            "SKIN_LESION": "run_skin_lesion",
        }
        return mapping.get(agent, "run_conversation")

    def _route_after_kb(self, state: WorkflowState) -> str:
        if state.get("selected_agent") == "WEB_EVIDENCE":
            return "run_web_evidence"
        return "check_validation"

    def _route_after_validation_check(self, state: WorkflowState) -> str:
        if state.get("requires_validation"):
            return "await_validation"
        return "apply_output_safety"

    # ── Graph construction ────────────────────────────────────────
    def _build_graph(self) -> Any:
        """Compile the full LangGraph workflow."""
        builder: StateGraph = StateGraph(WorkflowState)

        builder.add_node("assess_input", self._assess_input)
        builder.add_node("select_agent", self._select_agent)
        builder.add_node("run_conversation", self._run_conversation)
        builder.add_node("run_knowledge_base", self._run_knowledge_base)
        builder.add_node("run_web_evidence", self._run_web_evidence)
        builder.add_node("run_brain_mri", self._run_brain_mri)
        builder.add_node("run_chest_xray", self._run_chest_xray)
        builder.add_node("run_skin_lesion", self._run_skin_lesion)
        builder.add_node("check_validation", lambda s: {})
        builder.add_node("await_validation", self._await_validation)
        builder.add_node(
            "apply_output_safety",
            self._apply_output_safety,
        )

        builder.add_edge(START, "assess_input")
        builder.add_conditional_edges(
            "assess_input",
            self._route_after_assessment,
            {
                "select_agent": "select_agent",
                "apply_output_safety": "apply_output_safety",
            },
        )
        builder.add_conditional_edges(
            "select_agent",
            self._route_after_selection,
            {
                "run_conversation": "run_conversation",
                "run_knowledge_base": "run_knowledge_base",
                "run_web_evidence": "run_web_evidence",
                "run_brain_mri": "run_brain_mri",
                "run_chest_xray": "run_chest_xray",
                "run_skin_lesion": "run_skin_lesion",
            },
        )
        builder.add_edge("run_conversation", "apply_output_safety")
        builder.add_conditional_edges(
            "run_knowledge_base",
            self._route_after_kb,
            {
                "run_web_evidence": "run_web_evidence",
                "check_validation": "check_validation",
            },
        )
        builder.add_edge("run_web_evidence", "check_validation")
        builder.add_edge("run_brain_mri", "check_validation")
        builder.add_edge("run_chest_xray", "check_validation")
        builder.add_edge("run_skin_lesion", "check_validation")
        builder.add_conditional_edges(
            "check_validation",
            self._route_after_validation_check,
            {
                "await_validation": "await_validation",
                "apply_output_safety": "apply_output_safety",
            },
        )
        builder.add_edge("await_validation", "apply_output_safety")
        builder.add_edge("apply_output_safety", END)

        return builder.compile(
            checkpointer=self._memory,
            interrupt_before=["await_validation"],
        )

    def process(
        self,
        query: str,
        session_id: str,
        image_path: str | None = None,
    ) -> dict[str, Any]:
        """Process a clinical query through the full workflow.

        Args:
            query: The clinical question or instruction.
            session_id: Unique session identifier.
            image_path: Optional path to an attached image.

        Returns:
            Final workflow state as a dict.
        """
        config = {"configurable": {"thread_id": session_id}}
        initial_state: WorkflowState = {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "image_path": image_path,
            "image_type": None,
            "selected_agent": None,
            "routing_confidence": 0.0,
            "response": None,
            "sources": [],
            "requires_validation": False,
            "validation_approved": None,
            "error": None,
        }
        result = self._graph.invoke(initial_state, config)
        return dict(result)

    def resume_after_validation(
        self,
        session_id: str,
        approved: bool,
    ) -> dict[str, Any]:
        """Resume a paused workflow after human validation.

        Args:
            session_id: Session to resume.
            approved: Whether the clinician approved.

        Returns:
            Final workflow state as a dict.
        """
        config = {"configurable": {"thread_id": session_id}}
        result = self._graph.invoke(Command(resume=approved), config)
        return dict(result)

    @property
    def knowledge_base(self) -> KnowledgeBase:
        """Expose the knowledge base for document ingestion."""
        return self._knowledge_base
