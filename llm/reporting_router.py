from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Request, Query, Response, HTTPException, Depends

from llm.reporting import fetch_messages_for_day, render_html, render_json, render_csv
from llm.router import verify_credentials  # reuse Basic Auth

router = APIRouter(tags=["Reports"])

@router.get("/reports/daily")
def reports_daily(
    request: Request,
    date: Optional[str] = Query(default=None, description="YYYY-MM-DD (HKT). Defaults to today."),
    format: Optional[str] = Query(default="html", regex="^(html|json|csv)$"),
    _auth: bool = Depends(verify_credentials),
):
    """
    Live daily chat transcript (HKT), grouped by session.
    Authenticated via the same Basic Auth as /chat.

    Examples:
    - /reports/daily?date=2025-12-05
    - /reports/daily?date=2025-12-05&format=json
    - /reports/daily?format=csv
    """
    try:
        messages, day = fetch_messages_for_day(date)
        if format == "json":
            return render_json(messages, day)
        if format == "csv":
            data = render_csv(messages)
            return Response(
                content=data,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="chat_transcript_{day.strftime("%Y-%m-%d")}.csv"'
                },
            )
        # default html
        html_body = render_html(messages, day)
        return Response(content=html_body, media_type="text/html; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build report: {e}")