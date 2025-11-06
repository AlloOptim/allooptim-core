"""Default Pydantic configuration for AlloOptim models.

This module provides the default configuration settings used by Pydantic
models throughout the AlloOptim codebase. These settings ensure consistent
validation behavior, type safety, and data integrity.

Configuration includes:
- Assignment validation for data integrity
- Strict extra field handling to prevent typos
- Support for arbitrary types (pandas DataFrames, etc.)
- Enum value usage for cleaner serialization
- Mutable models for flexibility
"""

from pydantic import ConfigDict

DEFAULT_PYDANTIC_CONFIG = ConfigDict(
    validate_assignment=True,
    extra="forbid",
    arbitrary_types_allowed=True,
    use_enum_values=True,
    frozen=False,
)
