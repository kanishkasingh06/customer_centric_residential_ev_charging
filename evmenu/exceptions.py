"""Domain-specific exceptions for the EV menu package."""


class EVMenuError(Exception):
    """Base class for all package-specific exceptions."""


class PhysicalConstraintError(EVMenuError, ValueError):
    """Raised when supplied data violate a physical EV constraint."""


class SignalValidationError(EVMenuError, ValueError):
    """Raised when a planning signal is malformed or internally inconsistent."""


class SchemaValidationError(EVMenuError, ValueError):
    """Raised when a schema object contains invalid or inconsistent data."""
