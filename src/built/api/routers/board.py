from fastapi import APIRouter

from built.api.deps import SessionDep
from built.api.schemas import BoardOut, CardOut
from built.domain.enums import Column
from built.services import card_service

router = APIRouter(prefix="/api/v1", tags=["board"])


@router.get("/projects/{project_id}/board", response_model=BoardOut)
async def get_board(project_id: str, session: SessionDep, include_archived: bool = False) -> BoardOut:
    board = await card_service.get_board(session, project_id, include_archived=include_archived)
    return BoardOut(
        pm=[CardOut.model_validate(c) for c in board[Column.PM]],
        developer=[CardOut.model_validate(c) for c in board[Column.DEVELOPER]],
        tester=[CardOut.model_validate(c) for c in board[Column.TESTER]],
        reviewer=[CardOut.model_validate(c) for c in board[Column.REVIEWER]],
        deployer=[CardOut.model_validate(c) for c in board[Column.DEPLOYER]],
    )
