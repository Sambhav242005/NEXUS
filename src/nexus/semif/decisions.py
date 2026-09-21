"""Preset SemIf decision option sets (configurable, per SPEC)."""
from .schemas import DecisionOption

TOOL_ROUTING = [
    DecisionOption(id="browser", description="Use browser tool."),
    DecisionOption(id="computer", description="Use computer-use controller."),
    DecisionOption(id="filesystem", description="Use filesystem tool."),
    DecisionOption(id="shell", description="Use shell tool."),
    DecisionOption(id="response", description="Answer directly, no tool."),
    DecisionOption(id="clarification", description="Ask user for clarification."),
]
ACTION_GATING = [
    DecisionOption(id="allow", description="Action is safe to execute."),
    DecisionOption(id="ask_confirmation", description="Need user confirmation first."),
    DecisionOption(id="reject", description="Action must not execute."),
]
TASK_PROGRESS = [
    DecisionOption(id="completed", description="Task completed successfully."),
    DecisionOption(id="failed", description="Task failed, stop."),
    DecisionOption(id="needs_more_observation", description="Need fresh screenshot/observation."),
    DecisionOption(id="retry", description="Retry last action once."),
    DecisionOption(id="replan", description="Replan from current state."),
]
CLARIFICATION = [
    DecisionOption(id="continue", description="Enough context, continue task."),
    DecisionOption(id="ask_user", description="Ask user a clarifying question."),
]
