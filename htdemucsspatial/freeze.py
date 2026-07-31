"""
freeze.py — choose which HTDemucs layers to train during the spatial fine-tune.

HTDemucs is a U-Net with a large cross-domain transformer at the bottleneck
(~75% of the weights) plus 4-level frequency and time branches:

    model.encoder / model.decoder     frequency (spectrogram) branch  ("z")
    model.tencoder / model.tdecoder   time (waveform) branch          ("t")

Each is an ``nn.ModuleList`` of blocks indexed from the input side, so index 0 is
nearest the input/bottleneck and index -1 is the block closest to the output.
The binaural cues we supervise are OUTPUT-side phenomena, so the useful recipes
train the last decoder block(s); this module lets you say exactly which blocks of
which branch to train, and compose them freely.

Strategy grammar
----------------
A strategy is either the special token ``"all"`` (full fine-tune, nothing
frozen) or one or more ``+``-joined *selectors*; every block not covered by a
selector is frozen::

    strategy := "all" | selector ("+" selector)*
    selector := module ["_" range]
    module   := enc | zenc | dec | zdec | tenc | tdec
    range    := all | first<k> | last<k> | <i>            (default: all)

``enc``/``dec`` address the frequency branch (``zenc``/``zdec`` are aliases);
``tenc``/``tdec`` address the time branch.  ``first<k>``/``last<k>`` select the
first/last *k* blocks, ``<i>`` a single block by (non-negative) index, and a bare
module name selects every block.

Examples
--------
====================================  ===================================
strategy                              trains
====================================  ===================================
``dec_last1``                         last frequency-decoder block
``dec_last2``                         last two frequency-decoder blocks
``zdec_last1``                        last frequency-decoder block (alias)
``tdec_last1``                        last time-decoder block
``enc_first1+dec_last1``              first encoder block + last decoder block
``dec_all+tdec_all``                  the whole decoder of both branches
``dec_last2+tenc_first1``             last two decoder blocks + first tencoder
``dec_2``                             frequency-decoder block at index 2
``all``                               everything (full fine-tune)
====================================  ===================================
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch.nn as nn

__all__ = ["apply_freeze_strategy", "parse_strategy", "MODULE_ALIASES"]

# Selector module token -> HTDemucs attribute holding that branch's block list.
# ``z*`` = frequency (spectrogram) branch, ``t*`` = time (waveform) branch;
# the bare ``enc``/``dec`` default to the frequency branch, matching HTDemucs'
# own ``encoder``/``decoder`` attribute names.
MODULE_ALIASES: Dict[str, str] = {
    "enc":  "encoder",
    "zenc": "encoder",
    "dec":  "decoder",
    "zdec": "decoder",
    "tenc": "tencoder",
    "tdec": "tdecoder",
}

# One selector: <module>[_<range>], range = all | first<k> | last<k> | <index>.
_SELECTOR = re.compile(
    r"^(?P<mod>enc|zenc|dec|zdec|tenc|tdec)"
    r"(?:_(?P<range>all|first(?P<first>\d+)|last(?P<last>\d+)|(?P<index>\d+)))?$"
)


def _resolve_selector(model: nn.Module, token: str) -> Tuple[str, List[int], List[nn.Parameter]]:
    """Resolve one selector token into (group_label, indices, params)."""
    m = _SELECTOR.match(token)
    if not m:
        raise ValueError(
            f"Invalid selector {token!r}. Expected <module>[_<range>] with "
            f"module in {sorted(MODULE_ALIASES)} and range "
            f"in {{all, first<k>, last<k>, <index>}}."
        )
    attr   = MODULE_ALIASES[m.group("mod")]
    blocks = list(getattr(model, attr))
    n      = len(blocks)

    if m.group("first") is not None:
        k = int(m.group("first"))
        if not 1 <= k <= n:
            raise ValueError(f"{token!r}: first<k> needs 1 <= k <= {n} ({attr} has {n} blocks)")
        idx = list(range(k))
    elif m.group("last") is not None:
        k = int(m.group("last"))
        if not 1 <= k <= n:
            raise ValueError(f"{token!r}: last<k> needs 1 <= k <= {n} ({attr} has {n} blocks)")
        idx = list(range(n - k, n))
    elif m.group("index") is not None:
        i = int(m.group("index"))
        if not 0 <= i < n:
            raise ValueError(f"{token!r}: index needs 0 <= i < {n} ({attr} has {n} blocks)")
        idx = [i]
    else:                                   # bare module or explicit "all"
        idx = list(range(n))

    label  = f"{attr}[{','.join(map(str, idx))}]"
    params = [p for i in idx for p in blocks[i].parameters()]
    return label, idx, params


def parse_strategy(model: nn.Module, strategy: str) -> Dict[str, List[nn.Parameter]]:
    """Resolve ``strategy`` into ``{group_label: [params]}`` (no side effects).

    The returned groups are exactly the parameters that ``apply_freeze_strategy``
    would leave trainable; everything else on the model is meant to be frozen.
    For the ``"all"`` token this is every parameter of the model.
    """
    if strategy == "all":
        return {"all": list(model.parameters())}

    groups: Dict[str, List[nn.Parameter]] = {}
    for token in strategy.split("+"):
        token = token.strip()
        if not token:
            raise ValueError(f"Empty selector in strategy {strategy!r}")
        label, _, params = _resolve_selector(model, token)
        groups[label] = params
    return groups


def apply_freeze_strategy(model: nn.Module, strategy: str) -> Dict[str, List[nn.Parameter]]:
    """Set ``requires_grad`` on ``model`` according to ``strategy``.

    Freezes every parameter, then unfreezes the blocks named by ``strategy``
    (see the module docstring for the grammar).  The ``"all"`` token instead
    leaves the whole model trainable.

    Args:
        model:    an HTDemucs instance exposing ``encoder``/``decoder``/
                  ``tencoder``/``tdecoder`` module lists.
        strategy: a strategy string, e.g. ``"dec_last2"`` or
                  ``"enc_first1+dec_last1"``.

    Returns:
        ``{group_label: [params]}`` describing what stays trainable, for logging.
    """
    groups = parse_strategy(model, strategy)

    if strategy == "all":
        for p in model.parameters():
            p.requires_grad_(True)
        return groups

    for p in model.parameters():
        p.requires_grad_(False)
    for params in groups.values():
        for p in params:
            p.requires_grad_(True)
    return groups
