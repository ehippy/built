from fastapi import APIRouter, HTTPException, Query, status

from built.api.deps import RequireApiKey, SessionDep
from built.api.schemas import CardColumnVisitOut, CardCreate, CardEventOut, CardOut
from built.services import card_service
from built.services.project_service import NotFoundError

router = APIRouter(prefix="/api/v1", tags=["cards"])


@router.post("/projects/{project_id}/cards", response_model=CardOut, status_code=status.HTTP_201_CREATED)
async def create_card(project_id: str, body: CardCreate, session: SessionDep, _: RequireApiKey) -> CardOut:
    try:
        card = await card_service.create_card(session, project_id, **body.model_dump())
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return CardOut.model_validate(card)


@router.get("/projects/{project_id}/cards", response_model=list[CardOut])
async def list_cards(project_id: str, session: SessionDep) -> list[CardOut]:
    cards = await card_service.list_cards(session, project_id)
    return [CardOut.model_validate(c) for c in cards]


@router.get("/cards/{card_id}", response_model=CardOut)
async def get_card(card_id: str, session: SessionDep) -> CardOut:
    try:
        card = await card_service.get_card(session, card_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return CardOut.model_validate(card)


@router.get("/cards/{card_id}/column-visits", response_model=list[CardColumnVisitOut])
async def list_column_visits(card_id: str, session: SessionDep) -> list[CardColumnVisitOut]:
    try:
        visits = await card_service.list_column_visits(session, card_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return [CardColumnVisitOut.model_validate(v) for v in visits]


@router.get("/cards/{card_id}/events", response_model=list[CardEventOut])
async def list_events(
    card_id: str,
    session: SessionDep,
    since_seq: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
) -> list[CardEventOut]:
    try:
        events = await card_service.list_events(session, card_id, since_seq=since_seq, limit=limit)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return [CardEventOut.model_validate(e) for e in events]


@router.post("/cards/{card_id}/retry", response_model=CardOut)
async def retry_card(card_id: str, session: SessionDep, _: RequireApiKey) -> CardOut:
    """The one human touchpoint outside the autonomous pipeline: un-stick a blocked or
    failed card, with a fresh safety-valve budget."""
    try:
        card = await card_service.retry_card(session, card_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return CardOut.model_validate(card)


@router.post("/cards/{card_id}/cancel", response_model=CardOut)
async def cancel_card(card_id: str, session: SessionDep, _: RequireApiKey) -> CardOut:
    try:
        card = await card_service.cancel_card(session, card_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return CardOut.model_validate(card)
