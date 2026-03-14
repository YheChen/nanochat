"""
SmolTalk2 by HuggingFace. Larger conversational/instruction dataset.
https://huggingface.co/datasets/HuggingFaceTB/smoltalk2
"""

from datasets import load_dataset
from tasks.common import Task


class SmolTalk2(Task):
    """
    SmolTalk2 dataset. Large multi-turn conversational corpus.
    Use max_rows to cap size for budgeted runs.
    """

    def __init__(self, split, config="SFT", max_rows=None, **kwargs):
        super().__init__(**kwargs)
        ds = load_dataset("HuggingFaceTB/smoltalk2", config, split=split)
        if max_rows is not None:
            max_rows = min(max_rows, len(ds))
            ds = ds.select(range(max_rows))
        self.ds = ds.shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        # ---------------------------------------------------------------------
        # sanity checking asserts here
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, "SmolTalk2 messages must have at least 2 messages"
        for i, message in enumerate(rest_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, (
                f"Message {i} has role {message['role']} but should be {expected_role}"
            )
            assert isinstance(message["content"], str), "Content must be a string"
        # ---------------------------------------------------------------------
        conversation = {
            "messages": messages,
        }
        return conversation
