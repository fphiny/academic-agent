from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AliasConfig:
    alias_file: str = "./data/aliases.json"


class AliasStore:
    """
    컬렉션 alias 저장소

    예:
        {
            "kb_current": "kb_v3",
            "faq_current": "faq_20260307"
        }
    """

    def __init__(self, config: Optional[AliasConfig] = None):
        self.config = config or AliasConfig()
        self._lock = threading.Lock()
        self._ensure_file()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _ensure_file(self) -> None:
        directory = os.path.dirname(self.config.alias_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.config.alias_file):
            with open(self.config.alias_file, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

    def _read_all(self) -> Dict[str, str]:
        self._ensure_file()

        with self._lock:
            with open(self.config.alias_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}

        if not isinstance(data, dict):
            return {}

        cleaned: Dict[str, str] = {}
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, str):
                cleaned[k] = v

        return cleaned

    def _write_all(self, data: Dict[str, str]) -> None:
        self._ensure_file()

        with self._lock:
            with open(self.config.alias_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def _validate_name(self, value: str, field_name: str) -> str:
        if value is None:
            raise ValueError(f"{field_name} is required")

        value = str(value).strip()
        if not value:
            raise ValueError(f"{field_name} is required")

        return value

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def set_alias(self, alias: str, collection_name: str) -> None:
        """
        alias 등록/수정
        """
        alias = self._validate_name(alias, "alias")
        collection_name = self._validate_name(collection_name, "collection_name")

        data = self._read_all()
        data[alias] = collection_name
        self._write_all(data)

    def delete_alias(self, alias: str) -> bool:
        """
        alias 삭제
        반환:
            True  -> 삭제됨
            False -> 원래 없음
        """
        alias = self._validate_name(alias, "alias")

        data = self._read_all()
        existed = alias in data

        if existed:
            del data[alias]
            self._write_all(data)

        return existed

    def get_alias(self, alias: str) -> Optional[str]:
        """
        alias가 가리키는 실제 collection 이름 반환
        """
        alias = self._validate_name(alias, "alias")
        data = self._read_all()
        return data.get(alias)

    def list_aliases(self) -> Dict[str, str]:
        """
        전체 alias 매핑 반환
        """
        return self._read_all()

    def has_alias(self, alias: str) -> bool:
        alias = self._validate_name(alias, "alias")
        data = self._read_all()
        return alias in data

    def resolve_alias(self, name: str) -> str:
        """
        alias가 있으면 실제 collection 이름 반환
        없으면 원래 name 그대로 반환
        """
        name = self._validate_name(name, "name")
        data = self._read_all()
        return data.get(name, name)

    def clear_aliases(self) -> None:
        """
        전체 alias 삭제
        """
        self._write_all({})


# ----------------------------------------------------------------------
# 전역 인스턴스
# ----------------------------------------------------------------------

_default_store: Optional[AliasStore] = None


def get_alias_store(alias_file: str = "./data/aliases.json") -> AliasStore:
    global _default_store

    if _default_store is None:
        _default_store = AliasStore(
            config=AliasConfig(alias_file=alias_file)
        )

    return _default_store


# ----------------------------------------------------------------------
# 편의 함수
# ----------------------------------------------------------------------

def set_alias(alias: str, collection_name: str) -> None:
    get_alias_store().set_alias(alias, collection_name)


def delete_alias(alias: str) -> bool:
    return get_alias_store().delete_alias(alias)


def get_alias(alias: str) -> Optional[str]:
    return get_alias_store().get_alias(alias)


def list_aliases() -> Dict[str, str]:
    return get_alias_store().list_aliases()


def has_alias(alias: str) -> bool:
    return get_alias_store().has_alias(alias)


def resolve_alias(name: str) -> str:
    return get_alias_store().resolve_alias(name)


def clear_aliases() -> None:
    get_alias_store().clear_aliases()