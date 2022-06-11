from __future__ import annotations
from typing import Any, Optional
import dataclasses

@dataclasses.dataclass
class MoneyExpression:
    """
    金額表現，関連する予算のID，議論ラベルを保持する．
    """

    moneyExpression: str
    relatedID: Optional[list[str]]
    argumentClass: Optional[str]


@dataclasses.dataclass(frozen=True)
class LProceedingItem:
    """
    地方議会会議録の一つ分の発言を保持する．
    発言には複数の金額表現が含まれる場合がある．
    """

    speakerPosition: str
    speaker: str
    utterance: str
    moneyExpressions: list[MoneyExpression]

    @staticmethod
    def from_dict(d: dict[str, Any]):
        return LProceedingItem(
            speakerPosition=d["speakerPosition"],
            speaker=d["speaker"],
            utterance=d["utterance"],
            moneyExpressions=[MoneyExpression(**m)
                              for m in d["moneyExpressions"]],
        )


@dataclasses.dataclass(frozen=True)
class LProceedingObject:
    """
    地方議会会議録の一つ分の会議を保持する．
    一つ分の会議は発言オブジェクトのリストを持つ．
    """

    date: str
    localGovernmentCode: str
    localGovernmentName: str
    proceedingTitle: str
    url: str
    proceeding: list[LProceedingItem]

    @staticmethod
    def from_dict(d: dict[str, Any]):
        return LProceedingObject(
            date=d["date"],
            localGovernmentCode=d["localGovernmentCode"],
            localGovernmentName=d["localGovernmentName"],
            proceedingTitle=d["proceedingTitle"],
            url=d["url"],
            proceeding=[LProceedingItem.from_dict(x) for x in d["proceeding"]],
        )


@dataclasses.dataclass(frozen=True)
class DSpeechRecord:
    """
    国会会議録の一つ分の発言を保持する．
    発言には複数の金額表現が含まれる場合がある．
    """

    speechID: str
    speechOrder: int
    speaker: str
    speakerYomi: Optional[str]
    speakerGroup: Optional[str]
    speakerPosition: Optional[str]
    speakerRole: Optional[str]
    speech: str
    startPage: int
    createTime: str
    updateTime: str
    speechURL: str
    moneyExpressions: list[MoneyExpression]

    @staticmethod
    def from_dict(d: dict[str, Any]):
        return DSpeechRecord(
            speechID=d["speechID"],
            speechOrder=d["speechOrder"],
            speaker=d["speaker"],
            speakerYomi=d["speakerYomi"],
            speakerGroup=d["speakerGroup"],
            speakerPosition=d["speakerPosition"],
            speakerRole=d["speakerRole"],
            speech=d["speech"],
            startPage=d["startPage"],
            createTime=d["createTime"],
            updateTime=d["updateTime"],
            speechURL=d["speechURL"],
            moneyExpressions=[MoneyExpression(**m)
                              for m in d["moneyExpressions"]],
        )


@dataclasses.dataclass(frozen=True)
class DProceedingObject:
    """
    国会会議録の一つ分の会議を保持する．
    一つ分の会議は発言オブジェクトのリストを持つ．
    """

    issueID: str
    imageKind: str
    searchObject: int
    session: int
    nameOfHouse: str
    nameOfMeeting: str
    issue: str
    date: str
    closing: Optional[str]
    speechRecord: list[DSpeechRecord]
    meetingURL: str
    pdfURL: str

    @staticmethod
    def from_dict(d: dict[str, Any]):
        return DProceedingObject(
            issueID=d["issueID"],
            imageKind=d["imageKind"],
            searchObject=d["searchObject"],
            session=d["session"],
            nameOfHouse=d["nameOfHouse"],
            nameOfMeeting=d["nameOfMeeting"],
            issue=d["issue"],
            date=d["date"],
            closing=d["closing"],
            speechRecord=[DSpeechRecord.from_dict(
                x) for x in d["speechRecord"]],
            meetingURL=d["meetingURL"],
            pdfURL=d["pdfURL"],
        )


@dataclasses.dataclass(frozen=True)
class MinutesObject:
    """
    BAMタスクにおける会議録データのフォーマット．
    地方議会と国会の両方を持つ．
    """

    local: list[LProceedingObject]
    diet: list[DProceedingObject]

    @staticmethod
    def from_dict(d: dict[str, Any]):
        return MinutesObject(
            local=[LProceedingObject.from_dict(x) for x in d["local"]],
            diet=[DProceedingObject.from_dict(x) for x in d["diet"]],
        )