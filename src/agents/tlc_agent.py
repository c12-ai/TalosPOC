from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputValidationError, ToolStrategy
from langchain.tools import tool
from langchain_core.messages import AnyMessage

from src.classes.operation import OperationResponse
from src.classes.PROMPT import TLC_AGENT_PROMPT
from src.classes.system_state import Compound, TLCAgentOutput, TLCAIOutput
from src.utils.logging_config import logger
from src.utils.models import TLC_MODEL
from src.utils.settings import ChatModelConfig, settings


@tool
def get_tlc_ratio_from_mcp(smiles: str) -> str:
    """
    Get TLC ratio from MCP server for a given compound SMILES.

    Args:
        smiles: SMILES expression of the compound.

    Returns:
        TLC ratio information as a string.

    """
    # TODO: Implement actual MCP server call when resources are available
    # This is a placeholder that can be extended with actual MCP integration
    logger.debug("Getting TLC ratio for SMILES: {}", smiles)
    return f"TLC ratio for {smiles} (MCP integration pending)"


class TLCAgent:
    def __init__(self) -> None:
        """Initialize the TLCAgent."""
        self.config: ChatModelConfig = settings.agents.tlc_agent
        self.tlc_agent = create_agent(
            model=TLC_MODEL,
            response_format=ToolStrategy[TLCAIOutput](TLCAIOutput),
            system_prompt=TLC_AGENT_PROMPT,
            tools=[get_tlc_ratio_from_mcp],
        )

    def run(self, user_input: list[AnyMessage]) -> OperationResponse[list[AnyMessage], TLCAgentOutput]:
        """
        Run the TLCAgent to extract compounds and get TLC ratio.

        Args:
            user_input (list[AnyMessage]): The user input messages ordered chronologically.

        Returns:
            OperationResponse[list[AnyMessage], TLCAgentOutput]: The operation response containing
                the extracted compounds and TLC ratio information.

        """
        start_time = datetime.now()

        logger.info("TLCAgent.run triggered with {} messages", len(user_input))

        # 1. Extract 1...n compounds from the text or if none extracted return a feedback message ask user to provide.
        ai_output = self._extract_compounds(input_msg=user_input)

        # 2. Call MCP Server to get the ratio
        if not ai_output.compounds:
            logger.warning("No compounds extracted from user input")
            # Return feedback message asking user to provide compounds
            end_time = datetime.now()
            return OperationResponse[list[AnyMessage], TLCAgentOutput](
                operation_id="tlc_agent_001",
                input=user_input,
                output=TLCAgentOutput(compounds=[]),
                start_time=start_time.isoformat(timespec="microseconds"),
                end_time=end_time.isoformat(timespec="microseconds"),
            )

        # Get TLC ratio from MCP server for each compound
        compounds_with_ratio = self._get_tlc_ratios(ai_output.compounds)

        # 3. Return the result
        end_time = datetime.now()

        return OperationResponse[list[AnyMessage], TLCAgentOutput](
            operation_id="tlc_agent_001",
            input=user_input,
            output=TLCAgentOutput(compounds=compounds_with_ratio),
            start_time=start_time.isoformat(timespec="microseconds"),
            end_time=end_time.isoformat(timespec="microseconds"),
        )

    def _extract_compounds(self, input_msg: list[AnyMessage]) -> TLCAIOutput:
        """
        Extract compounds from user input text.

        Args:
            input_msg (list[AnyMessage]): The user input messages.

        Returns:
            TLCAIOutput: The extracted compounds.

        Raises:
            StructuredOutputValidationError: If the agent output validation fails.

        """
        try:
            result = self.tlc_agent.invoke(input={"messages": input_msg})  # type: ignore
            ai_output = result["structured_response"]

            if not isinstance(ai_output, TLCAIOutput):
                raise TypeError(
                    f"TLCAgent output is not a TLCAIOutput: {ai_output}. It is {type(ai_output)}",
                )

            logger.info(
                "TLCAgent extracted {} compounds: {}",
                len(ai_output.compounds),
                [c.compound_name for c in ai_output.compounds],
            )
        except StructuredOutputValidationError as se:
            logger.error("TLCAgent structured output validation error. error={}", se)
            raise
        else:
            return ai_output

    def _get_tlc_ratios(self, compounds: list[Compound]) -> list[Compound]:
        """
        Get TLC ratio from MCP server for each compound.

        Args:
            compounds (list[Compound]): List of compounds to get ratios for.

        Returns:
            list[Compound]: List of compounds with ratio information.

        """
        logger.info("Getting TLC ratios for {} compounds via MCP server", len(compounds))

        # Try to fetch TLC ratio from MCP server for each compound
        for compound in compounds:
            try:
                # Use the tool to get TLC ratio from MCP server
                ratio_info = get_tlc_ratio_from_mcp.invoke({"smiles": compound.smiles})
                logger.debug(
                    "Got TLC ratio for compound {} (SMILES: {}): {}",
                    compound.compound_name,
                    compound.smiles,
                    ratio_info,
                )
            except Exception as e:
                logger.warning("Failed to get TLC ratio for compound {}: {}", compound.compound_name, e)

        # For now, return compounds as-is until MCP server is fully configured
        # In the future, we might want to extend Compound model to include ratio information
        return compounds
