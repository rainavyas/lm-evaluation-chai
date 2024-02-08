from typing import Any, List, Tuple
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

eval_logger = utils.eval_logger

def chai_predict(input_text, submission_id, developer_key):
    '''
        Chai Model response to input_text
    '''
    payload = {
            "memory": '',
            "prompt": '',
            "chat_history": [{'message':input_text, 'sender':''}],
            "bot_name": "",
            "user_name": ""
        }
    endpoint = '/models/{submission_id}/chat'
    http_client = SubmitterClient(developer_key)
    response = http_client.post(endpoint=endpoint, submission_id=submission_id, timeout=20, data=payload)
    return response


@register_model("chai")
class ChaiLM(LM):

    def __init__(
        self,
        submission_id: str = "mistralai-mistral-7b-instruct_v2",
        developer_key: str = "",
        **kwargs, # not used yet
    ) -> None:
        """Chai API wrapper.

        :submission_id: str
            Chai model  submission ID e.g. 'mistralai-mistral-7b-instruct_v2'
        :developer_key: str
            Chai developer key
        :param kwargs: Any
            Additional model_args to pass to the API client - NOT USED YET
        """
        super().__init__()

        try:
            from chaiverse.http_client import SubmitterClient
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'chai' LM type, but package `chaiverse` is not installed. \
please install chaiverse via `pip install chaiverse`",
            )

        self.submission_id = submission_id
        self.developer_key = developer_key
        self.kwargs = kwargs

    @property
    def eot_token_id(self):
        raise NotImplementedError("No tokenization.")

    @property
    def max_length(self) -> int:
        return 2048

    @property
    def max_gen_toks(self) -> int:
        raise NotImplementedError("No support.")

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for logits.")

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError("No support for device as we use API.")

    def tok_encode(self, string: str) -> List[int]:
        raise NotImplementedError("No support for tokenizer.")

    def tok_decode(self, tokens: List[int]) -> str:
        raise NotImplementedError("No support for tokenizer.")

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("No support for logits.")

    def generate_until(self, requests) -> List[str]:

        if not requests:
            return []

        _requests: List[Tuple[str, dict]] = [req.args for req in requests]

        res = []
        for request in tqdm(_requests):
            inp = request[0]
            response = chai_predict(
                inp,
                self.submission_id,
                self.developer_key
            )
            res.append(response)
            self.cache_hook.add_partial("generate_until", request, response)
        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.")
