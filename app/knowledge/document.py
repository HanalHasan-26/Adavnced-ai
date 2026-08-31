from __future__ import annotations

# Import dataclass support.
from dataclasses import dataclass, field

# Import datetime support.
from datetime import datetime

# Import UUID support.
from uuid import UUID, uuid4


@dataclass
class KnowledgeDocument:

    # ---------------------------------------------------------
    # CORE FIELDS
    # ---------------------------------------------------------

    # Unique document identifier.
    id: UUID

    # Actual knowledge content.
    content: str

    # Original source.
    source: str

    # Source type such as text, pdf, web, etc.
    source_type: str

    # Creation timestamp.
    created_at: datetime

    # ---------------------------------------------------------
    # EXTENDED FIELDS
    # ---------------------------------------------------------

    # Human-readable title.
    #
    # Default is provided for backward compatibility with the
    # previous KnowledgeDocument constructor.
    title: str = ""

    # Last modification timestamp.
    #
    # Optional here so older code that only supplies
    # created_at continues to work.
    updated_at: datetime | None = None

    # Additional document metadata.
    metadata: dict[str, str] = field(
        default_factory=dict
    )

    # ---------------------------------------------------------
    # POST INITIALIZATION
    # ---------------------------------------------------------

    def __post_init__(self) -> None:

        # Validate content.
        if not isinstance(
            self.content,
            str,
        ):
            raise ValueError(
                "content must be a string."
            )

        self.content = self.content.strip()

        if not self.content:
            raise ValueError(
                "content cannot be empty."
            )

        # Validate source.
        if not isinstance(
            self.source,
            str,
        ):
            raise ValueError(
                "source must be a string."
            )

        self.source = self.source.strip()

        if not self.source:
            raise ValueError(
                "source cannot be empty."
            )

        # Validate source type.
        if not isinstance(
            self.source_type,
            str,
        ):
            raise ValueError(
                "source_type must be a string."
            )

        self.source_type = (
            self.source_type.strip()
        )

        if not self.source_type:
            raise ValueError(
                "source_type cannot be empty."
            )

        # Validate title.
        if not isinstance(
            self.title,
            str,
        ):
            raise ValueError(
                "title must be a string."
            )

        self.title = self.title.strip()

        # Older documents don't have updated_at.
        #
        # Use created_at as the initial update timestamp.
        if self.updated_at is None:
            self.updated_at = self.created_at

        # Validate metadata.
        if not isinstance(
            self.metadata,
            dict,
        ):
            raise ValueError(
                "metadata must be a dictionary."
            )

        # Copy metadata so callers cannot accidentally
        # mutate the document through the original dictionary.
        self.metadata = dict(
            self.metadata
        )

        # Validate metadata contents.
        for key, value in self.metadata.items():

            if not isinstance(
                key,
                str,
            ):
                raise ValueError(
                    "metadata keys must be strings."
                )

            if not isinstance(
                value,
                str,
            ):
                raise ValueError(
                    "metadata values must be strings."
                )

    # ---------------------------------------------------------
    # CREATE
    # ---------------------------------------------------------

    @classmethod
    def create(
        cls,
        content: str,
        source: str,
        source_type: str,
        title: str = "",
        metadata: dict[str, str] | None = None,
    ) -> "KnowledgeDocument":

        # Validate metadata.
        if metadata is None:
            metadata = {}

        # Use one timestamp for both fields.
        now = datetime.now()

        # Create the document.
        return cls(
            id=uuid4(),
            content=content,
            source=source,
            source_type=source_type,
            created_at=now,
            title=title,
            updated_at=now,
            metadata=dict(metadata),
        )

    # ---------------------------------------------------------
    # UPDATE CONTENT
    # ---------------------------------------------------------

    def update_content(
        self,
        content: str,
    ) -> None:

        # Validate content.
        if not isinstance(
            content,
            str,
        ):
            raise ValueError(
                "content must be a string."
            )

        content = content.strip()

        if not content:
            raise ValueError(
                "content cannot be empty."
            )

        # Update content.
        self.content = content

        # Update modification timestamp.
        self.updated_at = datetime.now()

    # ---------------------------------------------------------
    # UPDATE TITLE
    # ---------------------------------------------------------

    def update_title(
        self,
        title: str,
    ) -> None:

        # Validate title.
        if not isinstance(
            title,
            str,
        ):
            raise ValueError(
                "title must be a string."
            )

        self.title = title.strip()

        # Update modification timestamp.
        self.updated_at = datetime.now()

    # ---------------------------------------------------------
    # UPDATE METADATA
    # ---------------------------------------------------------

    def update_metadata(
        self,
        metadata: dict[str, str],
    ) -> None:

        # Validate metadata.
        if not isinstance(
            metadata,
            dict,
        ):
            raise ValueError(
                "metadata must be a dictionary."
            )

        metadata_copy = dict(
            metadata
        )

        for key, value in metadata_copy.items():

            if not isinstance(
                key,
                str,
            ):
                raise ValueError(
                    "metadata keys must be strings."
                )

            if not isinstance(
                value,
                str,
            ):
                raise ValueError(
                    "metadata values must be strings."
                )

        # Replace metadata.
        self.metadata = metadata_copy

        # Update modification timestamp.
        self.updated_at = datetime.now()

    # ---------------------------------------------------------
    # SERIALIZATION
    # ---------------------------------------------------------

    def to_dict(
        self,
    ) -> dict[str, object]:

        # Make sure updated_at is available.
        updated_at = (
            self.updated_at
            or self.created_at
        )

        return {
            "id": str(self.id),
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "source_type": self.source_type,
            "created_at": (
                self.created_at.isoformat()
            ),
            "updated_at": (
                updated_at.isoformat()
            ),
            "metadata": dict(
                self.metadata
            ),
        }

    # ---------------------------------------------------------
    # DESERIALIZATION
    # ---------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> "KnowledgeDocument":

        # Validate input.
        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "data must be a dictionary."
            )

        try:

            document_id = UUID(
                str(data["id"])
            )

            content = str(
                data["content"]
            )

            source = str(
                data["source"]
            )

            source_type = str(
                data["source_type"]
            )

            created_at = datetime.fromisoformat(
                str(data["created_at"])
            )

            # Title is optional for backward compatibility.
            title = str(
                data.get(
                    "title",
                    "",
                )
            )

            # Older serialized documents may not have
            # updated_at.
            raw_updated_at = data.get(
                "updated_at"
            )

            if raw_updated_at is None:

                updated_at = created_at

            else:

                updated_at = datetime.fromisoformat(
                    str(raw_updated_at)
                )

            # Metadata is optional.
            raw_metadata = data.get(
                "metadata",
                {},
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:

            raise ValueError(
                "invalid knowledge document data."
            ) from error

        # Validate metadata.
        if not isinstance(
            raw_metadata,
            dict,
        ):
            raise ValueError(
                "metadata must be a dictionary."
            )

        metadata: dict[str, str] = {}

        for key, value in raw_metadata.items():

            if not isinstance(
                key,
                str,
            ):
                raise ValueError(
                    "metadata keys must be strings."
                )

            if not isinstance(
                value,
                str,
            ):
                raise ValueError(
                    "metadata values must be strings."
                )

            metadata[key] = value

        return cls(
            id=document_id,
            content=content,
            source=source,
            source_type=source_type,
            created_at=created_at,
            title=title,
            updated_at=updated_at,
            metadata=metadata,
        )