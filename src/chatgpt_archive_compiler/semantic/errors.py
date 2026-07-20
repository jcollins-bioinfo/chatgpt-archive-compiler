"""Safe exception type for semantic-atlas construction failures."""


class SemanticAtlasError(RuntimeError):
    """Raised when a provider or local semantic stage violates an invariant."""
