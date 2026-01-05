"""
TLCAgent is a specialist agent that is responsible for filling the TLC spec with user and then recommend develop solvent and ratio.

TODO: Code needs to be cleaned up.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, interrupt

from src.models.operation import OperationInterruptPayload, OperationResponse
from src.models.tlc import (
    Compound,
    TLCAgentGraphState,
    TLCAgentOutput,
    TLCAIOutput,
    TLCCompoundSpecItem,
    TLCRatioPayload,
    TLCRatioResult,
)
from src.utils.logging_config import logger
from src.utils.models import TLC_MODEL
from src.utils.PROMPT import TLC_AGENT_PROMPT
from src.utils.tools import coerce_operation_resume


def get_recommended_ratio(compounds: list[Compound] | None = None) -> list[TLCRatioResult]:
    """Get the recommended ratio of the TLC experiment for each compound."""
    host = "52.83.119.132"
    url = f"http://{host}:8000/api/tlc-request"

    if not compounds:
        raise ValueError("Missing compounds")

    results: list[TLCRatioResult] = []
    for idx, compound in enumerate(compounds):
        compound_name = (compound.compound_name or "").strip()
        smiles = (compound.smiles or "").strip() if compound.smiles else ""
        if not compound_name and not smiles:
            raise ValueError(f"Missing `compound_name` and `smiles` in `compounds[{idx}]`")

        payload: dict[str, str] = {"model_backend": "GAT"}
        if compound_name:
            payload["compound_name"] = "aspirin"  # TODO: substitute with compound_name
        if smiles:
            payload["smiles"] = ""  # TODO: substitute with smiles

        try:
            t0 = time.perf_counter()
            with httpx.Client(timeout=20.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()["result"]
                parsed = TLCRatioPayload.model_validate(data)
        except Exception:
            logger.exception("Failed to get TLC recommended ratio from MCP server")
            raise
        else:
            results.append(parsed.tlc_parameters)
            logger.info(
                "TLC ratio lookup ok. idx={} backend={} elapsed_ms={}",
                idx,
                parsed.tlc_parameters.backend,
                int((time.perf_counter() - t0) * 1000),
            )
            logger.info(f"Recommended ratio: {parsed.tlc_parameters}")

    return results


class TLCAgent:
    """
    TLC agent implemented as a LangGraph subgraph, with a single public entrypoint: `run()`.

    Flow:
    - extract_compound_and_fill_spec
    - user_confirm
      - revise  -> back to extract_compound_and_fill_spec
      - confirm -> done (return `tlc_spec` to parent graph)
    """

    def __init__(self, *, with_checkpointer: bool = False) -> None:
        """
        Build and compile the internal LangGraph subgraph.

        Args:
            with_checkpointer: If True, compiles with MemorySaver for standalone usage with HITL.
                              If False (default), parent graph's checkpointer will be used.

        """
        logger.info("TLCAgent initialized with model={}", TLC_MODEL)

        subgraph = StateGraph(TLCAgentGraphState)
        subgraph.add_node("extract_compound_and_fill_spec", self._extract_compound_and_fill_spec)
        subgraph.add_node("user_confirm", self._user_confirm)
        subgraph.add_node("fill_recommended_ratio", self._fill_recommended_ratio)

        subgraph.add_edge(START, "extract_compound_and_fill_spec")
        subgraph.add_edge("extract_compound_and_fill_spec", "user_confirm")
        subgraph.add_conditional_edges(
            "user_confirm",
            self._route_user_confirm,
            {
                "revise": "extract_compound_and_fill_spec",
                "confirm": "fill_recommended_ratio",
            },
        )
        subgraph.add_edge("fill_recommended_ratio", END)

        checkpointer = MemorySaver() if with_checkpointer else None

        self.compiled = subgraph.compile(checkpointer=checkpointer)
        self._agent = create_agent(
            model=TLC_MODEL,
            system_prompt=TLC_AGENT_PROMPT,
            response_format=ProviderStrategy(TLCAIOutput),
        )

    # NOTE: Run() is not necessary actually

    def run(
        self,
        *,
        tlc_state: TLCAgentGraphState | Command,
        thread_id: str = str(uuid.uuid4()),
    ) -> dict[str, Any]:
        """
        Unified entrypoint for the TLC subgraph in the main graph.

        This method intentionally does NOT simulate UI or loop. If the subgraph interrupts (HITL),
        it should bubble to the outer runtime (server/UI) to resume later.

        Args:
            tlc_state: The current TLC state.
            thread_id: The thread ID for the conversation.

        Returns:
            The complete state of the TLC subgraph execution.

        """
        return self.compiled.invoke(tlc_state, config=RunnableConfig(configurable={"thread_id": thread_id}))

    @staticmethod
    def _build_response(
        final_state: dict | TLCAgentGraphState,
        original_input: list[AnyMessage],
        start_time: datetime,
    ) -> OperationResponse[list[AnyMessage], TLCAgentOutput]:
        """Extract data from final state and build OperationResponse."""
        if isinstance(final_state, dict):
            output_form = final_state.get("tlc_spec")
            messages = final_state.get("messages", original_input)
        else:
            output_form = getattr(final_state, "tlc_spec", None)
            messages = getattr(final_state, "messages", original_input)

        if not isinstance(output_form, TLCAgentOutput):
            raise TypeError(f"TLCAgent did not produce TLCAgentOutput. Got: {type(output_form)}")

        end_time = datetime.now()
        return OperationResponse[list[AnyMessage], TLCAgentOutput](
            operation_id="tlc_agent.run",
            input=list(messages),
            output=output_form,
            start_time=start_time.isoformat(timespec="microseconds"),
            end_time=end_time.isoformat(timespec="microseconds"),
        )

    def _extract_compound_and_fill_spec(self, state: TLCAgentGraphState) -> dict[str, Any]:
        """Extract compound info from messages and build tlc_spec draft."""
        t0 = time.perf_counter()
        result = self._agent.invoke({"messages": state.messages})  # pyright: ignore[reportArgumentType]
        model_resp: TLCAIOutput = result["structured_response"]
        logger.info(
            "TLC extract done. messages={} compounds={} resp_len={} elapsed_ms={}",
            len(state.messages),
            len([c for c in model_resp.compounds if c is not None]),
            len(model_resp.resp_msg or ""),
            int((time.perf_counter() - t0) * 1000),
        )

        updated_spec: TLCAgentOutput = TLCAgentOutput(
            compounds=model_resp.compounds,
            resp_msg=model_resp.resp_msg,
            spec=[],
            confirmed=False,
        )

        return {"tlc_spec": updated_spec}

    @staticmethod
    def _fill_recommended_ratio(state: TLCAgentGraphState) -> dict[str, Any]:
        if not isinstance(state.tlc_spec, TLCAgentOutput) or not state.tlc_spec.compounds:
            raise TypeError("Missing `tlc_spec`/`compounds` before fill_recommended_ratio")

        compounds = [c for c in state.tlc_spec.compounds if c is not None]
        if any((not (c.compound_name or "").strip()) and (not (c.smiles or "").strip()) for c in compounds):
            raise ValueError("Each compound must include at least one of `compound_name` or `smiles` before requesting ratio")
        t0 = time.perf_counter()
        ratios = get_recommended_ratio(compounds=compounds)
        spec = [
            TLCCompoundSpecItem(
                compound_name=c.compound_name,
                smiles=c.smiles,
                solvent_system=r.solvent_system,
                ratio=r.ratio,
                rf_value=r.rf_value,
                description=r.description,
                origin=r.origin,
                backend=r.backend,
            )
            for c, r in zip(compounds, ratios, strict=True)
        ]
        logger.info(
            "TLC fill_ratio done. compounds={} elapsed_ms={}",
            len(compounds),
            int((time.perf_counter() - t0) * 1000),
        )

        return {"tlc_spec": state.tlc_spec.model_copy(update={"spec": spec, "confirmed": True})}

    @staticmethod
    def _user_confirm(state: TLCAgentGraphState) -> dict[str, Any]:
        """Human-in-the-loop confirm/revise for `tlc_spec`."""
        if not isinstance(state.tlc_spec, TLCAgentOutput):
            raise TypeError("Missing `tlc_spec` before user_confirm")

        interrupt_payload = OperationInterruptPayload(
            message="Please confirm the TLC spec. Approve to confirm; reject to revise. (Structure-only POC)",
            args={"tlc_spec": state.tlc_spec.model_dump(mode="json")},
        )
        payload = interrupt(interrupt_payload.model_dump(mode="json"))
        resume = coerce_operation_resume(payload)

        updates: dict[str, Any] = {"user_approved": bool(resume.approval)}

        edited_text = (resume.comment or "").strip()
        if edited_text and not resume.approval:
            updates["messages"] = [*list(state.messages), HumanMessage(content=edited_text)]

        if resume.approval:
            updates["tlc_spec"] = state.tlc_spec.model_copy(update={"confirmed": True})

        updates_from_data = TLCAgent._coerce_spec(resume.data)
        if isinstance(updates_from_data, TLCAgentOutput):
            updates["tlc_spec"] = updates_from_data
            logger.info("TLC user_confirm applied spec from resume.data")

        return updates

    @staticmethod
    def _route_user_confirm(state: TLCAgentGraphState) -> str:
        return "confirm" if state.user_approved else "revise"

    @staticmethod
    def _coerce_spec(value: Any) -> TLCAgentOutput | None:
        if not isinstance(value, dict):
            return None
        spec_dict = value.get("tlc_spec") if isinstance(value.get("tlc_spec"), dict) else value
        try:
            return TLCAgentOutput.model_validate(spec_dict)
        except Exception:
            return None


# tlc_agent_subgraph = TLCAgent()
# Avoid import side effects and duplicate initialization, instantiate it in node_mapper.py

if __name__ == "__main__":
    from src.models.operation import OperationResume
    from src.utils.tools import _pretty, terminal_approval_handler

    agent = TLCAgent(with_checkpointer=True)
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = RunnableConfig(configurable={"thread_id": thread_id})

    text = input("[user]: ").strip() or "我正在进行水杨酸的乙酰化反应制备乙酰水杨酸帮我进行中控监测IPC"
    print(f"user input: {text}")
    next_input: TLCAgentGraphState | Command = TLCAgentGraphState(messages=[HumanMessage(content=text)], tlc_spec=None)
    res: dict[str, Any] = {}

    while res.get("user_approved") is None or not res.get("user_approved"):
        res = agent.run(tlc_state=next_input, thread_id=thread_id)

        print(_pretty(res))  # which is the complete subgraph state

        if "__interrupt__" in res:  # HITL interrupt
            interrupts: list[Interrupt] = res["__interrupt__"]
            print(_pretty(interrupts[0].value))  # payload for UI, not resume

            next_input = terminal_approval_handler(res)
