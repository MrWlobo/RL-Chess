from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class PositionsConfig(BaseModel):
    path: Path

    @field_validator("path")
    @classmethod
    def check_path_exists(cls, v: Path) -> Path:
        v = Path(__file__).parent / v
        if not v.exists():
            raise ValueError(f"EPD file not found at: {v.absolute()}")
        return v


class MaiaExeConfig(BaseModel):
    lc0_path: Path

    @field_validator("lc0_path")
    @classmethod
    def check_path_exists(cls, v: Path) -> Path:
        v = Path(__file__).parent / v
        if not v.exists():
            raise ValueError(f"exe file not found at: {v.absolute()}")
        return v


class MaiaConfig(BaseModel):
    exe: MaiaExeConfig
    weights: dict[str, Path]

    @field_validator("weights")
    @classmethod
    def check_path_exists(cls, v: dict[str, Path]) -> dict[str, Path]:
        for key, p in v.items():
            p = Path(__file__).parent / p
            v[key] = p
            if not p.exists():
                raise ValueError(f"weights file not found at: {p.absolute()}")
        return v


class BenchmarkConfig(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file=Path(__file__).parent / "config.toml"
    )

    positions: PositionsConfig
    maia: MaiaConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
        )
