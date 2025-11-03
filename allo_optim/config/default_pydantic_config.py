from pydantic import ConfigDict

DEFAULT_PYDANTIC_CONFIG = ConfigDict(
	validate_assignment=True,
	extra="forbid",
	arbitrary_types_allowed=True,
	use_enum_values=True,
	frozen=True,
)
