# Create a component responsible for converting
# normalized queries into searchable terms.
class QueryTokenizer:

    # Convert a query into individual terms.
    def tokenize(self, query: str) -> list[str]:

        # Remove unnecessary whitespace.
        query = query.strip()

        # Return no terms for an empty query.
        if not query:
            return []

        # Split the query into individual words.
        terms = query.split()

        # Remove duplicate terms while preserving order.
        unique_terms = list(dict.fromkeys(terms))

        # Return the searchable terms.
        return unique_terms