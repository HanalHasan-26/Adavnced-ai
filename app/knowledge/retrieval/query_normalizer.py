# Import the regular expression module.
import re


# Create a component responsible for normalizing search queries.
class QueryNormalizer:

    # Normalize a search query.
    def normalize(self, query: str) -> str:

        # Reject non-string values.
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        # Convert the query to lowercase.
        query = query.lower()

        # Replace punctuation and special characters with spaces.
        query = re.sub(
            r"[^\w\s]",
            " ",
            query,
        )

        # Collapse multiple whitespace characters into one space.
        query = re.sub(
            r"\s+",
            " ",
            query,
        )

        # Remove whitespace from both ends.
        return query.strip()