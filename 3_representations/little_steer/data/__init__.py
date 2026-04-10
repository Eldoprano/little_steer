"""little_steer.data — Dataset schema, conversion, and tokenization utilities."""

from thesis_schema import AnnotatedSpan, ConversationEntry
from .converter import convert_file, load_dataset, save_dataset, iter_dataset
from .tokenizer_utils import TokenPositionMapper, TokenSpan

__all__ = [
    "AnnotatedSpan",
    "ConversationEntry",
    "convert_file",
    "load_dataset",
    "save_dataset",
    "iter_dataset",
    "TokenPositionMapper",
    "TokenSpan",
]
