"""
target.py
Wraps generate_answer() from model_utils.py in PyRIT's
PromptTarget interface so orchestrators can attack it.
Reuses the already-loaded model (no reload).
"""

from pyrit.models import PromptRequestResponse, construct_response_from_request
from pyrit.prompt_target import PromptTarget

from model_utils import generate_answer


class LlamaAppTarget(PromptTarget):
    """PyRIT target that calls our local Llama-3B QA function."""

    def __init__(self):
        super().__init__()

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        request_piece = prompt_request.request_pieces[0]
        prompt_text = request_piece.converted_value

        answer_text = generate_answer(prompt_text)

        response = construct_response_from_request(
            request=request_piece,
            response_text_pieces=[answer_text],
        )
        return response

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        # Only text prompts supported for this simple target.
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt piece per request.")

    def is_json_response_supported(self) -> bool:
        return False
