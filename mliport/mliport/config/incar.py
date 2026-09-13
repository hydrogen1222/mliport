"""INCAR-style configuration parser.

Moved here from the top-level :mod:`mliport.config` module so the configuration
package owns all parsing code (plan section 17.7). The behaviour is unchanged;
the legacy module re-exports this class for backward compatibility.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from mliport.config.provenance import SourceLocation


class IncarConfig(dict):
    """VASP-style INCAR configuration parser.

    The lexer removes syntax-level quotes/comments but deliberately keeps raw
    value tokens as strings. The shared :mod:`mliport.config.schema` performs
    field-directed typing later, so ``HEAD = 0`` remains the string ``"0"``
    while ``CPU_THREADS = 1`` becomes an integer in the resolved config.

    Example:
        >>> config = IncarConfig.from_file("INCAR.mliport")
        >>> print(config["CALC_TYPE"])  # "SP"
        >>> print(config.get("FMAX", 0.05))  # 0.05 with default
    """

    TRUE_VALUES = {".true.", ".t.", "true", "t", "yes", "y", "1"}
    FALSE_VALUES = {".false.", ".f.", "false", "f", "no", "n", "0"}

    def __init__(
        self,
        *args,
        source_name: str = "<mapping>",
        base_dir: str | Path | None = None,
        **kwargs,
    ):
        """Initialize configuration with optional initial values."""
        super().__init__(*args, **kwargs)
        self._comments: dict[str, str] = {}
        self._origins: dict[str, SourceLocation] = {}
        self.source_name = source_name
        self.base_dir = SourceLocation.synthetic(source_name, base_dir).base_dir

    @classmethod
    def from_file(cls, filepath: str | Path) -> IncarConfig:
        """Parse INCAR file and return configuration object.

        Args:
            filepath: Path to INCAR file

        Returns:
            IncarConfig instance with parsed values

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file has invalid format
        """
        path = Path(filepath).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        return cls._from_content(
            path.read_text(encoding="utf-8"),
            source_name=str(path),
            base_dir=path.parent,
            source_path=path,
        )

    @classmethod
    def from_string(
        cls,
        content: str,
        *,
        source_name: str = "<string>",
        base_dir: str | Path | None = None,
    ) -> IncarConfig:
        """Parse configuration from string content.

        Args:
            content: String containing INCAR-style configuration

        Returns:
            IncarConfig instance with parsed values
        """
        return cls._from_content(
            content,
            source_name=source_name,
            base_dir=base_dir,
            source_path=None,
        )

    @classmethod
    def _from_content(
        cls,
        content: str,
        *,
        source_name: str,
        base_dir: str | Path | None,
        source_path: Path | None,
    ) -> IncarConfig:
        config = cls(source_name=source_name, base_dir=base_dir)
        for line_num, raw_line in enumerate(content.splitlines(), 1):
            try:
                statement, comment = cls._split_comment(raw_line)
                statement = statement.strip()
                if not statement:
                    continue
                if "=" not in statement:
                    raise ValueError("expected exactly one KEY = VALUE assignment")
                key, _, raw_value = statement.partition("=")
                key = key.strip().upper()
                if not re.fullmatch(r"[A-Z_][A-Z0-9_-]*", key):
                    raise ValueError(f"invalid option key {key!r}")
                value = cls._strip_syntax_quotes(raw_value.strip())
            except ValueError as exc:
                location = (
                    f"line {line_num} in {source_name}"
                    if source_path is not None
                    else f"line {line_num}"
                )
                raise ValueError(f"Error parsing {location}: {exc}") from exc

            config[key] = value
            if comment:
                config._comments[key] = comment.strip()
            if source_path is not None:
                config._origins[key] = SourceLocation.from_file(source_path, line_num)
            else:
                config._origins[key] = SourceLocation(
                    base_dir=config.base_dir,
                    line=line_num,
                    label=source_name,
                )
        return config

    @staticmethod
    def _split_comment(line: str) -> tuple[str, str]:
        quote: str | None = None
        escaped = False
        for index, char in enumerate(line):
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in {'"', "'"}:
                quote = char
            elif char in {"#", "!"}:
                return line[:index], line[index + 1 :]
            elif char == ";":
                raise ValueError(
                    "semicolon-separated assignments are unsupported; use one per line"
                )
        if quote is not None:
            raise ValueError("unterminated quoted value")
        return line, ""

    @staticmethod
    def _strip_syntax_quotes(value: str) -> str:
        if value and value[0] in {'"', "'"}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError("quoted value must occupy the complete value token")
            quote = value[0]
            body = value[1:-1]
            body = body.replace(f"\\{quote}", quote).replace("\\\\", "\\")
            return body
        return value

    @staticmethod
    def _format_value(value: object) -> str:
        if isinstance(value, bool):
            return ".TRUE." if value else ".FALSE."
        text = str(value)
        if text != text.strip() or any(
            char in text for char in ("#", "!", ";", "'", '"')
        ):
            escaped = text.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return text

    def origin(self, key: str) -> SourceLocation | None:
        """Return the source location for ``key`` (case-insensitive)."""
        return self._origins.get(key.upper())

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value with default."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in self.TRUE_VALUES:
                return True
            if value_lower in self.FALSE_VALUES:
                return False
        raise ValueError(f"Cannot convert {key}={value!r} to boolean")

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer value with default."""
        value = self.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"Cannot convert {key}={value!r} to integer")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            if math.isfinite(value) and value.is_integer():
                return int(value)
            raise ValueError(f"Cannot convert {key}={value!r} to integer")
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            return int(value)
        raise ValueError(f"Cannot convert {key}={value!r} to integer")

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float value with default."""
        value = self.get(key, default)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
        elif isinstance(value, str):
            try:
                numeric = float(value.strip().replace("D", "E").replace("d", "e"))
            except ValueError as exc:
                raise ValueError(f"Cannot convert {key}={value!r} to float") from exc
        else:
            raise ValueError(f"Cannot convert {key}={value!r} to float")
        if not math.isfinite(numeric):
            raise ValueError(f"Cannot convert {key}={value!r} to finite float")
        return numeric

    def get_str(self, key: str, default: str | None = "") -> str | None:
        """Get string value with default."""
        value = self.get(key, default)
        return None if value is None else str(value)

    def write(self, filepath: str | Path) -> None:
        """Write configuration to file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_string())

    def to_string(self) -> str:
        """Convert configuration to INCAR-formatted string."""
        lines = []
        lines.append("# mliport Calculation Settings")
        lines.append("")

        # Group by categories for readability
        categories = {
            "CALC_TYPE": "Calculation Type",
            "TASK": "Task Selection",
            "MODEL_TYPE": "Model Settings",
            "MODEL_PATH": "Model Settings",
            "MODEL": "Model Settings",
            "DEVICE": "Device Settings",
            "INFERENCE_MODE": "Inference Settings",
            "DEFAULT_DTYPE": "Inference Settings",
            "HEAD": "Inference Settings",
            "OPT_ALGO": "Optimization",
            "FMAX": "Optimization",
            "MAX_STEPS": "Optimization",
            "CELL_OPT": "Optimization",
            "FIX_SYMMETRY": "Optimization",
            "MD_ENSEMBLE": "Molecular Dynamics",
            "TEMPERATURE": "Molecular Dynamics",
            "TIMESTEP": "Molecular Dynamics",
            "STEPS": "Molecular Dynamics",
            "THERMOSTAT": "Molecular Dynamics",
            "FRICTION": "Molecular Dynamics",
            "BUSSI_TAU": "Molecular Dynamics",
            "NHC_TDAMP": "Molecular Dynamics",
            "NHC_TCHAIN": "Molecular Dynamics",
            "NHC_TLOOP": "Molecular Dynamics",
            "COM_POLICY": "Molecular Dynamics",
        }

        current_category = None

        for key in sorted(self.keys()):
            value = self[key]

            # Determine category
            category = categories.get(key, "Other")
            if category != current_category:
                lines.append(f"# {category}")
                current_category = category

            # Format value
            formatted_value = self._format_value(value)

            # Add comment if present
            comment = self._comments.get(key, "")
            if comment:
                lines.append(f"{key:20s} = {formatted_value:20s} # {comment}")
            else:
                lines.append(f"{key:20s} = {formatted_value}")

        return "\n".join(lines) + "\n"

    def validate(self, required_keys: list[str] | None = None) -> list[str]:
        """Validate configuration.

        Args:
            required_keys: List of required keys

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if required_keys:
            for key in required_keys:
                if key not in self:
                    errors.append(f"Required key '{key}' is missing")

        # Keep the historical domain-specific messages below, but route all
        # other known fields through the shared typed schema. Unknown keys are
        # handled after layering, once STRICT_CONFIG has itself been resolved.
        from mliport.config.schema import get_schema  # noqa: PLC0415

        custom_validation = {
            "CALCULATION",
            "CALC_TYPE",
            "DEFAULT_DTYPE",
            "DEVICE",
            "MD_ENSEMBLE",
            "MODEL_TYPE",
            "OPT_ALGO",
            "TASK",
            "THERMOSTAT",
        }
        schema_values = {
            key: value for key, value in self.items() if key not in custom_validation
        }
        errors.extend(
            get_schema().validate_dict(
                schema_values,
                strict=False,
                context=self.source_name,
            )
        )

        # Validate specific keys
        valid_model_types = {"uma", "fairchem", "mace", "dpa", "grace"}
        if "MODEL_TYPE" in self:
            mt = self.get_str("MODEL_TYPE").lower()
            if mt not in valid_model_types:
                errors.append(
                    f"Invalid MODEL_TYPE '{mt}'. "
                    f"Must be one of: {', '.join(valid_model_types - {'fairchem'})}"
                )

        # UMA tasks + generic bulk/molecule (non-UMA engines)
        valid_tasks = {
            "omat",
            "omol",
            "oc20",
            "oc25",
            "odac",
            "omc",
            "bulk",
            "molecule",
        }
        if "TASK" in self:
            task = self.get_str("TASK").lower()
            if task not in valid_tasks:
                errors.append(
                    f"Invalid TASK '{task}'. Must be one of: {', '.join(valid_tasks)}"
                )
        # Only the calculation types the engine can actually execute. Keep this
        # list in sync with CalculationEngine.VALID_CALC_TYPES and the schema
        # choices (schema.py): phonon/analyze were previously accepted here but
        # rejected at engine construction with a confusing late error.
        valid_calc_types = {"sp", "opt", "md", "neb", "batch"}
        calc_declarations = {
            key: self.get_str(key).lower()
            for key in ("CALC_TYPE", "CALCULATION")
            if key in self
        }
        if len(set(calc_declarations.values())) > 1:
            errors.append(
                "Conflicting CALC_TYPE/CALCULATION declarations: "
                + ", ".join(
                    f"{key}={value!r}" for key, value in calc_declarations.items()
                )
            )
        if calc_declarations:
            calc_type = next(iter(calc_declarations.values()))
            if calc_type not in valid_calc_types:
                declaration_name = (
                    "CALC_TYPE" if "CALC_TYPE" in calc_declarations else "CALCULATION"
                )
                errors.append(
                    f"Invalid {declaration_name} '{calc_type}'. "
                    f"Must be one of: {', '.join(valid_calc_types)}"
                )

        valid_devices = {"cpu", "cuda", "gpu"}
        if "DEVICE" in self:
            device = self.get_str("DEVICE").lower()
            # Allow cuda:N device strings (plan section 17.2).
            is_device = device in valid_devices or device.startswith("cuda:")
            if not is_device:
                errors.append(
                    f"Invalid DEVICE '{device}'. "
                    f"Must be one of: {', '.join(valid_devices)} or cuda:N"
                )

        # Only the optimizers the engine actually implements (OptimizationRunner.OPTIMIZERS):
        # gpmin/mdmin were previously accepted here but raised at runner construction.
        valid_optimizers = {"fire", "bfgs", "lbfgs"}
        if "OPT_ALGO" in self:
            optimizer = self.get_str("OPT_ALGO").lower()
            if optimizer not in valid_optimizers:
                errors.append(
                    f"Invalid OPT_ALGO '{optimizer}'. "
                    f"Must be one of: {', '.join(valid_optimizers)}"
                )

        valid_md_ensembles = {"nve", "nvt"}
        if "MD_ENSEMBLE" in self:
            ensemble = self.get_str("MD_ENSEMBLE").lower()
            if ensemble not in valid_md_ensembles:
                errors.append(
                    f"Invalid MD_ENSEMBLE '{ensemble}'. "
                    f"Must be one of: {', '.join(valid_md_ensembles)}"
                )

        if "THERMOSTAT" in self:
            thermostat = self.get_str("THERMOSTAT").lower()
            valid_thermostats = {"langevin", "bussi", "nhc"}
            if thermostat not in valid_thermostats:
                errors.append(
                    f"Invalid THERMOSTAT '{thermostat}'. "
                    f"Must be one of: {', '.join(valid_thermostats)}"
                )

        # Validate MACE dtype if present.
        if "DEFAULT_DTYPE" in self:
            dtype = self.get_str("DEFAULT_DTYPE").lower()
            if dtype not in {"float32", "float64"}:
                errors.append(
                    f"Invalid DEFAULT_DTYPE '{dtype}'. Must be float32 or float64"
                )

        return errors
