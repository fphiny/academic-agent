from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SendMailSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEND_MAIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    smtp_host: str = Field(default="smtp.naver.com")
    smtp_port: int = Field(default=587)
    smtp_use_tls: bool = Field(default=True)
    smtp_use_ssl: bool = Field(default=False)

    smtp_username: str = Field(default="")
    smtp_password: str = Field(default="")

    from_email: str = Field(default="")
    from_name: str = Field(default="Hallym Guide")

    timeout_seconds: int = Field(default=20)

    @field_validator(
        "smtp_host",
        "smtp_username",
        "smtp_password",
        "from_email",
        "from_name",
        mode="before",
    )
    @classmethod
    def strip_string_values(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().replace("\r", "")
        return v

    @field_validator("smtp_use_tls", "smtp_use_ssl", mode="before")
    @classmethod
    def parse_bool_values(cls, v: Any) -> Any:
        if isinstance(v, str):
            value = v.strip().replace("\r", "").lower()
            if value in {"true", "1", "yes", "y", "on"}:
                return True
            if value in {"false", "0", "no", "n", "off"}:
                return False
        return v

    @field_validator("smtp_port", "timeout_seconds", mode="before")
    @classmethod
    def parse_int_values(cls, v: Any) -> Any:
        if isinstance(v, str):
            value = v.strip().replace("\r", "")
            if value:
                return int(value)
        return v

    def validate_required(self) -> None:
        missing: list[str] = []

        if not self.smtp_host:
            missing.append("SEND_MAIL_SMTP_HOST")
        if not self.smtp_username:
            missing.append("SEND_MAIL_SMTP_USERNAME")
        if not self.smtp_password:
            missing.append("SEND_MAIL_SMTP_PASSWORD")
        if not self.from_email:
            missing.append("SEND_MAIL_FROM_EMAIL")

        if missing:
            raise ValueError(
                "메일 설정이 누락되었습니다: " + ", ".join(missing)
            )


@lru_cache(maxsize=1)
def get_send_mail_settings() -> SendMailSettings:
    return SendMailSettings()