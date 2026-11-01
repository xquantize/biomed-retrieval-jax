"""A tiny randomly initialised BERT with a WordPiece tokenizer, saved as a checkpoint."""

from pathlib import Path

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = (
    "the a of and in for with to fever pain joint blood pressure vitamin bone density drugs "
    "older adults kidney disease treatment diabetes chronic short title abstract ##s ##ing"
).split()


def save_tiny_checkpoint(path: Path) -> Path:
    """Save a 2-layer BERT (random weights) and its tokenizer to `path`, and return it."""
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertConfig, BertModel

    vocab = {token: i for i, token in enumerate(SPECIAL + WORDS)}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer.save(str(path / "tokenizer.json"))

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=37,
        max_position_embeddings=64,
        type_vocab_size=2,
        pad_token_id=vocab["[PAD]"],
    )
    BertModel(config).save_pretrained(path)
    return path
