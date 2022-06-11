from __future__ import annotations
from typing import Any, Optional
import dataclasses


@dataclasses.dataclass(frozen=True)
class BudgetObject:
    """
    BAMタスクにおける予算リストの配布用フォーマット．
    地方議会と国会の両方を持つ．

    地方議会（local）は，キーが自治体コード（localGovernmentCode），値がその自治体の予算項目リストとなる辞書型である．
    国会（diet）は，予算項目リストである．
    """

    local: dict[str, list[BudgetItem]]
    diet: list[BudgetItem]

    @staticmethod
    def from_dict(d: dict[str, Any]):
        return BudgetObject(
            local={k: [BudgetItem(**x) for x in v]
                   for k, v in d["local"].items()},
            diet=[BudgetItem(**x) for x in d["diet"]],
        )

@dataclasses.dataclass(frozen=True)
class BudgetItem:
    """
    予算項目一つ分を保持する．

    budgetIdの命名規則は以下の通り．
    ID-[year]-[localGovernmentCode]-00-[index]
    例：ID-2020-401307-00-000001
    """

    budgetId: str
    budgetTitle: str
    url: Optional[str]
    budgetItem: str
    budget: str
    categories: list[str]
    typesOfAccount: Optional[str]
    department: str
    budgetLastYear: Optional[str]
    description: str
    budgetDifference: Optional[str]