"""Restore and release only objects owned by one strict native binding."""
from dataclasses import fields, is_dataclass


def checkpoint_restore(model):
    """Snapshot native instance attributes before shared scale hooks replace them."""
    names = ('gradient_checkpointing', '_gradient_checkpointing_func')
    saved = [(module, {name: vars(module)[name] for name in names if name in vars(module)})
             for module in model.modules()]
    def restore():
        for module, values in saved:
            for name in names:
                if name in values:
                    vars(module)[name] = values[name]
                else:
                    vars(module).pop(name, None)
    return restore


def release_tensors(value, seen=None):
    """Move owned state in place, including aliases inside frozen response rows.

    Do not traverse arbitrary user/runtime/model objects. Modules are moved by
    their explicit owner; this walk covers rows, containers and PG/optimizer
    bookkeeping, including nested optimizer history outside ``state``.
    """
    import torch
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        with torch.no_grad():
            value.grad = None
            value.data = value.detach().to('cpu')
    elif isinstance(value, torch.nn.Module):
        return
    elif isinstance(value, dict):
        for child in value.values():
            release_tensors(child, seen)
    elif isinstance(value, (tuple, list)):
        for child in value:
            release_tensors(child, seen)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            release_tensors(getattr(value, field.name), seen)
    elif type(value).__module__.startswith(('particlegan.', 'particle_sliders.', 'hndl.', 'torch.optim.')):
        for child in vars(value).values():
            release_tensors(child, seen)


def attempt_all(actions):
    """A failed hook/transfer must not skip independent release actions."""
    errors = []
    for action in actions:
        try:
            action()
        except BaseException as error:
            errors.append(error)
    if errors:
        raise BaseExceptionGroup('strict native owned cleanup failed', errors)
