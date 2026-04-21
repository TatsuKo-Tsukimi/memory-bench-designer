"""Items and item pool — the data-model layer of a scenario."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Item:
    id: str
    archetype: str
    content: str
    primary_theme: Optional[int]
    theme_affinity: Dict[int, float]
    importance: float
    created_session: int
    created_step: int
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None

    def affinity_to(self, theme: int) -> float:
        return self.theme_affinity.get(theme, 0.0)

    def is_active_at(self, session: int) -> bool:
        if self.superseded_by is None:
            return True
        return session < self._supersede_session

    _supersede_session: int = field(default=10**9, repr=False)


@dataclass
class ItemPool:
    items: Dict[str, Item] = field(default_factory=dict)
    arrivals: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)

    def add(self, item: Item) -> None:
        self.items[item.id] = item
        key = (item.created_session, item.created_step)
        self.arrivals.setdefault(key, []).append(item.id)

    def get(self, item_id: str) -> Item:
        return self.items[item_id]

    def arrivals_at(self, session: int, step: int) -> List[Item]:
        return [self.items[i] for i in self.arrivals.get((session, step), [])]

    def link_supersedes(self, old_id: str, new_id: str, at_session: int) -> None:
        old = self.items[old_id]
        new = self.items[new_id]
        old.superseded_by = new_id
        old._supersede_session = at_session
        new.supersedes = old_id

    def __len__(self) -> int:
        return len(self.items)
