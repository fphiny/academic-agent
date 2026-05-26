from __future__ import annotations

import mimetypes
import smtplib
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Iterable, List, Sequence

from modules.agent_speed.tools.base import ToolResult

from .settings import get_send_mail_settings


class SendMailError(Exception):
    pass


def _normalize_recipients(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    else:
        items = [str(x).strip() for x in value]

    recipients = [x for x in items if x]
    if not recipients:
        raise SendMailError("수신자 이메일이 비어 있습니다.")
    return recipients


def _attach_files(
    message: EmailMessage,
    attachments: Iterable[str | Path] | None,
) -> None:
    if not attachments:
        return

    for item in attachments:
        path = Path(item)
        if not path.exists():
            raise SendMailError(f"첨부파일을 찾을 수 없습니다: {path}")

        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type:
            maintype, subtype = mime_type.split("/", 1)
        else:
            maintype, subtype = "application", "octet-stream"

        with path.open("rb") as f:
            data = f.read()

        message.add_attachment(
            data,
            maintype=maintype,
            subtype=subtype,
            filename=path.name,
        )


def build_email_message(
    *,
    to: str | Sequence[str],
    subject: str,
    body: str,
    cc: str | Sequence[str] | None = None,
    bcc: str | Sequence[str] | None = None,
    html_body: str | None = None,
    attachments: Iterable[str | Path] | None = None,
) -> tuple[EmailMessage, list[str]]:
    settings = get_send_mail_settings()
    settings.validate_required()

    to_list = _normalize_recipients(to)
    cc_list = _normalize_recipients(cc) if cc else []
    bcc_list = _normalize_recipients(bcc) if bcc else []

    msg = EmailMessage()
    msg["Subject"] = subject.strip()
    msg["From"] = formataddr((settings.from_name, settings.from_email))
    msg["To"] = ", ".join(to_list)

    if cc_list:
        msg["Cc"] = ", ".join(cc_list)

    msg.set_content(body or "")

    if html_body:
        msg.add_alternative(html_body, subtype="html")

    _attach_files(msg, attachments)

    all_recipients = to_list + cc_list + bcc_list
    return msg, all_recipients


def send_mail(
    *,
    to: str | Sequence[str],
    subject: str,
    body: str,
    cc: str | Sequence[str] | None = None,
    bcc: str | Sequence[str] | None = None,
    html_body: str | None = None,
    attachments: Iterable[str | Path] | None = None,
) -> dict:
    settings = get_send_mail_settings()
    settings.validate_required()

    message, all_recipients = build_email_message(
        to=to,
        subject=subject,
        body=body,
        cc=cc,
        bcc=bcc,
        html_body=html_body,
        attachments=attachments,
    )

    try:
        if settings.smtp_use_ssl:
            with smtplib.SMTP_SSL(
                host=settings.smtp_host,
                port=settings.smtp_port,
                timeout=settings.timeout_seconds,
            ) as server:
                server.login(settings.smtp_username, settings.smtp_password)
                server.send_message(message, to_addrs=all_recipients)
        else:
            with smtplib.SMTP(
                host=settings.smtp_host,
                port=settings.smtp_port,
                timeout=settings.timeout_seconds,
            ) as server:
                server.ehlo()

                if settings.smtp_use_tls:
                    server.starttls()
                    server.ehlo()

                server.login(settings.smtp_username, settings.smtp_password)
                server.send_message(message, to_addrs=all_recipients)

    except smtplib.SMTPAuthenticationError as e:
        raise SendMailError(
            "SMTP 인증 실패: 네이버 SMTP 설정 또는 애플리케이션 비밀번호를 확인하세요."
        ) from e
    except smtplib.SMTPException as e:
        raise SendMailError(f"메일 전송 실패: {str(e)}") from e
    except OSError as e:
        raise SendMailError(f"SMTP 연결 실패: {str(e)}") from e

    return {
        "ok": True,
        "message": "메일이 전송되었습니다.",
        "subject": subject,
        "to": all_recipients,
    }


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