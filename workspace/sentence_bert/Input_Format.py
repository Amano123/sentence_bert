from typing import Union, List


class Input_Format:
    """
    Sentence-BERTに入力するデータ形式を指定
    """
    def __init__(self, guid: str, texts: List[str], label: Union[int, float]):
        """
        # guid
            番号
        # texts
            テキスト
        # label
            ラベル
        """
        self.guid = guid
        self.texts = [text.strip() for text in texts]
        self.label = label