# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Miscellaneous Helpers."""

from typing import Type

import torch

__all__ = ['is_model_deepspeed', 'warning_on_one_line']


def is_model_deepspeed(model: torch.nn.Module) -> bool:
    """Whether ``model`` is an instance of a :class:`~deepspeed.DeepSpeedEngine`."""
    try:
        import deepspeed
    except ImportError:
        return False
    else:
        return isinstance(model, deepspeed.DeepSpeedEngine)


def warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    """Force Python warnings to consolidate into one line."""
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'
