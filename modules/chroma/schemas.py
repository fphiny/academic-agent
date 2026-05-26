from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FetchHtmlRequest(BaseModel):
    url: str


class EmbedUrlsRequest(BaseModel):
    urls: list[str] = Field(default_factory=list)
    element_id: list[str] = Field(default_factory=list)
    element_class: list[str] = Field(default_factory=list)
    index: str
    meta: dict[str, Any] = Field(default_factory=dict)


class ScrapeRequest(EmbedUrlsRequest):
    pass