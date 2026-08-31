from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4


@dataclass
class KnowledgeDocument:
    """
    Represents one stored knowledge document.

    A document contains the original content plus metadata describing
    where the knowledge came from.
    """

    content: str
    source: str
    source_type: str
    id: UUID
    created_at: datetime

    @classmethod
    def create(
        cls,
        content: str,
        source: str,
        source_type: str,
    ) -> "KnowledgeDocument":
        """
        Create a new knowledge document with a generated ID
        and current creation timestamp.
        """

        if not isinstance(content, str):
            raise ValueError(
                "content must be a string."
            )

        if not content.strip():
            raise ValueError(
                "content cannot be empty."
            )

        if not isinstance(source, str):
            raise ValueError(
                "source must be a string."
            )

        if not source.strip():
            raise ValueError(
                "source cannot be empty."
            )

        if not isinstance(source_type, str):
            raise ValueError(
                "source_type must be a string."
            )

        if not source_type.strip():
            raise ValueError(
                "source_type cannot be empty."
            )

        return cls(
            id=uuid4(),
            content=content,
            source=source,
            source_type=source_type,
            created_at=datetime.now(),
        )