"""Portable single-token labels, checked in their actual answer context."""
import itertools
import string


def label_vocabulary(tokenizer, answer_prefix, capacity=255):
    prefix_ids = tokenizer.encode(answer_prefix, add_special_tokens=False)
    candidates = itertools.chain(string.ascii_uppercase + string.ascii_lowercase + string.digits,
                                 (a + b for a in string.ascii_uppercase for b in string.ascii_uppercase))
    labels, ids = [], []
    for label in candidates:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) != 1 or tokens[0] in ids:
            continue
        if tokenizer.encode(answer_prefix + label, add_special_tokens=False) != prefix_ids + tokens:
            continue
        labels.append(label)
        ids.append(tokens[0])
        if len(labels) == capacity:
            break
    if len(labels) < 2:
        raise ValueError("Tokenizer has too few distinct labels at the answer position")
    return labels, ids


def reject_control_tokens(tokenizer, *texts):
    if any(token in text for token in tokenizer.all_special_tokens for text in texts):
        raise ValueError("Input contains reserved model control tokens")
