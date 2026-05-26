from __future__ import annotations

from typing import List

from modules.agent_speed.tools.base import ToolResult

from .service import SendMailError, send_mail


def run(
    to: str | List[str],
    subject: str,
    body: str,
    cc: str | List[str] | None = None,
    bcc: str | List[str] | None = None,
    html_body: str | None = None,
) -> ToolResult:
    try:
        result = send_mail(
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            html_body=html_body,
        )
        return ToolResult(
            name="send_mail",
            ok=True,
            data=result,
        )
    except SendMailError as e:
        return ToolResult(
            name="send_mail",
            ok=False,
            data={"error": str(e)},
        )
    except Exception as e:
        return ToolResult(
            name="send_mail",
            ok=False,
            data={"error": f"unexpected send_mail error: {str(e)}"},
        )