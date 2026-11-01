"""LLM(Claude API)による自然言語タスクプランナー。

シーン状態(ブロック・トレイの色と位置)と日本語の指示文を Claude に渡し、
「どのブロックをどのトレイに運ぶか」の実行計画(構造化出力)を推論させる。
学習は行わず、推論のみで AI を活用する。
"""

import json
from typing import List

import anthropic
from pydantic import BaseModel

MODEL = "claude-opus-5"


class PickPlace(BaseModel):
    """1回のピック&プレース動作。"""

    cube_id: str
    tray_id: str
    reason: str  # なぜこの動作を行うか(日本語・簡潔に)


class SortPlan(BaseModel):
    """ロボットが実行する計画全体。"""

    steps: List[PickPlace]
    summary: str  # 指示をどう解釈したかの説明(日本語・60文字以内)


SYSTEM = """あなたはロボットアームのタスクプランナーです。
テーブル上のブロック(cubes)とトレイ(trays)の状態がJSONで与えられます。
ユーザーの日本語指示に従い、ピック&プレースの実行計画を作成してください。

ルール:
- cube_id / tray_id は必ずシーン状態に存在するIDを使うこと。
- 指示に含まれる制約(触ってはいけない対象など)は厳守すること。
- 動かす必要のないブロックは計画に含めないこと。
- summary には指示の解釈と除外したものの理由を簡潔に書くこと。"""


def make_plan(scene_state: dict, instruction: str) -> SortPlan:
    """Claude API を呼び出して実行計画を得る。"""
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY を環境変数から読む
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    "## シーン状態\n"
                    f"{json.dumps(scene_state, ensure_ascii=False, indent=2)}\n\n"
                    f"## 指示\n{instruction}"
                ),
            }
        ],
        output_format=SortPlan,
    )
    return response.parsed_output


def make_plan_offline(scene_state: dict, instruction: str) -> SortPlan:
    """API キーなしで動作確認するためのルールベース代替(AI不使用)。

    「色ごとに同じ色のトレイへ、緑は触らない」という固定ルールのみ。
    提出用の実行には使わない。
    """
    tray_by_color = {t["color"]: t["id"] for t in scene_state["trays"]}
    steps = [
        PickPlace(
            cube_id=c["id"],
            tray_id=tray_by_color[c["color"]],
            reason=f"{c['color']}のブロックを{c['color']}のトレイへ",
        )
        for c in scene_state["cubes"]
        if c["color"] != "green" and c["color"] in tray_by_color
    ]
    return SortPlan(steps=steps, summary="(オフライン動作確認モード: 固定ルールで仕分け)")
