from fastapi import APIRouter, HTTPException

from app.database import (
    delete_strike_label,
    get_analysis_run,
    list_strike_labels,
    upsert_strike_label,
)
from app.models import StrikeLabel, StrikeLabelRequest, StrikeLabelsResponse

router = APIRouter()


def _row_to_label(row) -> StrikeLabel:
    return StrikeLabel(
        id=row["id"],
        analysis_id=row["analysis_id"],
        time_s=float(row["time_s"]),
        source=row["source"],
        is_strike=bool(row["is_strike"]),
        algorithm_validated=(
            None if row["algorithm_validated"] is None else bool(row["algorithm_validated"])
        ),
        comment=row["comment"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


@router.get("/strike-labels/{analysis_id}", response_model=StrikeLabelsResponse)
async def get_strike_labels(analysis_id: str) -> StrikeLabelsResponse:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    rows = await list_strike_labels(analysis_id)
    return StrikeLabelsResponse(
        analysis_id=analysis_id,
        labels=[_row_to_label(r) for r in rows],
    )


@router.post("/strike-labels/{analysis_id}", response_model=StrikeLabel)
async def post_strike_label(analysis_id: str, payload: StrikeLabelRequest) -> StrikeLabel:
    analysis = await get_analysis_run(analysis_id)
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    row = await upsert_strike_label(
        analysis_id,
        time_s=payload.time_s,
        source=payload.source,
        is_strike=payload.is_strike,
        algorithm_validated=payload.algorithm_validated,
        comment=payload.comment,
    )
    return _row_to_label(row)


@router.delete("/strike-labels/{label_id}")
async def remove_strike_label(label_id: str) -> dict:
    deleted = await delete_strike_label(label_id)
    if not deleted:
        raise HTTPException(404, "label not found")
    return {"deleted": True, "id": label_id}
