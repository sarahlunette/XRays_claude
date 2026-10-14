"""Registry + factory pattern (revision plan 7.6).

New models and losses plug in by decorating a class/function with ``@MODELS.register("name")``;
the trainer builds them from a config string, so adding a component never edits core code
(open/closed principle).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(obj: Callable[..., T]) -> Callable[..., T]:
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj

        return deco

    def build(self, name: str, **params: Any) -> T:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name](**params)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


MODELS: Registry[Any] = Registry("model")
LOSSES: Registry[Any] = Registry("loss")
