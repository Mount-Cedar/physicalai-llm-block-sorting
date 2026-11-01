"""AI×ロボティクス課題: 自然言語指示によるブロック仕分けタスク。

Genesis シミュレータ上の Franka Panda アームが、日本語の指示文を
LLM(Claude API)に解釈させて得た実行計画に従い、色付きブロックを
トレイへ仕分ける。実行の様子は字幕付き mp4 として録画される。

実行例:
    python src/sorting_task.py                # Claude API でプランニング(要 ANTHROPIC_API_KEY)
    python src/sorting_task.py --offline      # 動作確認用(AI不使用の固定ルール)
"""

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # ヘッドレス描画

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_planner import SortPlan, make_plan, make_plan_offline

# ---------------------------------------------------------------- シーン定義

CUBE_SIZE = 0.045
COLORS_RGB = {
    "red": (0.85, 0.15, 0.15),
    "blue": (0.15, 0.30, 0.85),
    "green": (0.15, 0.70, 0.25),
}

CUBES = [
    {"id": "red_cube_1", "color": "red", "pos": [0.52, -0.06]},
    {"id": "red_cube_2", "color": "red", "pos": [0.66, 0.08]},
    {"id": "blue_cube_1", "color": "blue", "pos": [0.58, -0.20]},
    {"id": "blue_cube_2", "color": "blue", "pos": [0.70, -0.07]},
    {"id": "green_cube_1", "color": "green", "pos": [0.45, 0.10]},
]

TRAYS = [
    {"id": "red_tray", "color": "red", "pos": [0.38, 0.38]},
    {"id": "blue_tray", "color": "blue", "pos": [0.62, 0.38]},
]

INSTRUCTION = (
    "テーブルの上のブロックを、同じ色のトレイに仕分けしてください。"
    "ただし緑のブロックは壊れやすいので、絶対に触らないでください。"
)

# 把持時のハンド高さ(キューブ中心 z=CUBE_SIZE/2 に対するオフセット)
HAND_OFFSET = 0.105
GRASP_Z = CUBE_SIZE / 2 + HAND_OFFSET
HOVER_Z = GRASP_Z + 0.20
PLACE_Z = GRASP_Z + 0.06  # トレイ上で放す高さ
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # グリッパを真下に向ける

FPS = 25
RENDER_EVERY = 4  # dt=0.01 × 4 = 0.04s/フレーム → 25fps で等速再生


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def wrap_jp(text, width=44):
    """日本語テキストを指定文字数で折り返す(全角基準の簡易版)。"""
    lines, cur, w = [], "", 0
    for ch in text:
        cw = 0.5 if ord(ch) < 0x3000 else 1.0
        if w + cw > width:
            lines.append(cur)
            cur, w = "", 0
        cur += ch
        w += cw
    if cur:
        lines.append(cur)
    return lines


def find_jp_font():
    for pat in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK*.ttc",
    ]:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


class CaptionRenderer:
    """PILでフレームに字幕(上段: 指示、下段: 現在の動作)を焼き込む。"""

    def __init__(self, width, height):
        from PIL import ImageFont

        self.w, self.h = width, height
        font_path = find_jp_font()
        size_l, size_s = int(height * 0.040), int(height * 0.030)
        if font_path:
            self.font_l = ImageFont.truetype(font_path, size_l)
            self.font_s = ImageFont.truetype(font_path, size_s)
        else:
            self.font_l = self.font_s = ImageFont.load_default()

    def draw(self, frame, top_lines=(), bottom_lines=(), badge=None):
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame).convert("RGBA")
        band = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(band)
        pad = int(self.h * 0.015)
        lh_l = int(self.h * 0.055)
        lh_s = int(self.h * 0.042)
        if top_lines:
            d.rectangle([0, 0, self.w, pad * 2 + lh_l * len(top_lines)], fill=(0, 0, 0, 150))
            for i, line in enumerate(top_lines):
                d.text((pad * 2, pad + i * lh_l), line, font=self.font_l, fill=(255, 255, 255, 255))
        if bottom_lines:
            y0 = self.h - (pad * 2 + lh_s * len(bottom_lines))
            d.rectangle([0, y0, self.w, self.h], fill=(0, 0, 0, 150))
            for i, line in enumerate(bottom_lines):
                d.text((pad * 2, y0 + pad + i * lh_s), line, font=self.font_s, fill=(255, 255, 160, 255))
        if badge:
            bb = d.textbbox((0, 0), badge, font=self.font_s)
            bw = bb[2] - bb[0] + pad * 2
            y0 = pad * 2 + (lh_l * len(top_lines) if top_lines else 0)
            d.rectangle([self.w - bw - pad, y0 + pad, self.w - pad, y0 + pad + lh_s + pad], fill=(30, 60, 140, 170))
            d.text((self.w - bw, y0 + pad * 2), badge, font=self.font_s, fill=(255, 255, 255, 255))
        return np.asarray(Image.alpha_composite(img, band).convert("RGB"))


class SortingSim:
    def __init__(self, res=(1280, 720), out_path="output/video.mp4", use_llm=True):
        import genesis as gs
        import imageio

        self.gs = gs
        self.use_llm = use_llm
        gs.init(backend=gs.cpu, logging_level="warning")

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
        )
        self.cube_entities = {}
        for c in CUBES:
            self.cube_entities[c["id"]] = self.scene.add_entity(
                gs.morphs.Box(
                    size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                    pos=(c["pos"][0], c["pos"][1], CUBE_SIZE / 2),
                ),
                surface=gs.surfaces.Default(color=COLORS_RGB[c["color"]]),
            )
        for t in TRAYS:
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.14, 0.14, 0.01),
                    pos=(t["pos"][0], t["pos"][1], 0.005),
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=COLORS_RGB[t["color"]]),
            )
        self.cam = self.scene.add_camera(
            res=res, pos=(1.7, -1.1, 1.15), lookat=(0.45, 0.05, 0.12), fov=35, GUI=False
        )
        self.scene.build()

        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )
        self.ee = self.franka.get_link("hand")

        home = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.785, 0.04, 0.04])
        self.franka.set_qpos(home)
        self.franka.control_dofs_position(home[:7], self.motors_dof)
        self.franka.control_dofs_position(home[7:], self.fingers_dof)

        self.writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8)
        self.captions = CaptionRenderer(*res)
        self.top_lines = []
        self.bottom_lines = []
        self.badge = "AI: Claude API (claude-opus-5)" if use_llm else "オフライン動作確認"
        self.step_count = 0
        self.frames_written = 0
        self.tray_fill = {t["id"]: 0 for t in TRAYS}

    # ------------------------------------------------------------ 低レベル制御

    def step(self, n=1):
        for _ in range(n):
            self.scene.step()
            self.step_count += 1
            if self.step_count % RENDER_EVERY == 0:
                self._write_frame()

    def _write_frame(self):
        out = self.cam.render()
        rgb = out[0] if isinstance(out, tuple) else out
        frame = self.captions.draw(
            np.asarray(rgb)[..., :3].astype(np.uint8),
            self.top_lines,
            self.bottom_lines,
            self.badge,
        )
        self.writer.append_data(frame)
        self.frames_written += 1

    def hold(self, seconds):
        """現在の制御目標のままシミュレーションを進める(字幕表示用)。"""
        self.step(int(seconds / 0.01))

    def ik(self, pos):
        q = self.franka.inverse_kinematics(link=self.ee, pos=np.array(pos), quat=DOWN_QUAT)
        return to_np(q)[:7]

    def move_to(self, pos, seconds=1.2):
        """関節空間で線形補間しながら目標位置へ移動する。"""
        target = self.ik(pos)
        start = to_np(self.franka.get_qpos())[:7]
        n = max(int(seconds / 0.01), 1)
        for i in range(n):
            a = (i + 1) / n
            self.franka.control_dofs_position(start * (1 - a) + target * a, self.motors_dof)
            self.step()

    def gripper_open(self):
        self.franka.control_dofs_position(np.array([0.04, 0.04]), self.fingers_dof)
        self.step(40)

    def gripper_grasp(self):
        self.franka.control_dofs_force(np.array([-1.5, -1.5]), self.fingers_dof)
        self.step(60)

    # ------------------------------------------------------------ タスク実行

    def scene_state(self):
        return {
            "cubes": [{"id": c["id"], "color": c["color"], "pos": c["pos"]} for c in CUBES],
            "trays": [{"id": t["id"], "color": t["color"], "pos": t["pos"]} for t in TRAYS],
        }

    def pick_and_place(self, cube_id, tray_id, caption):
        cube = self.cube_entities[cube_id]
        cx, cy, _ = to_np(cube.get_pos()).tolist()
        tray = next(t for t in TRAYS if t["id"] == tray_id)
        k = self.tray_fill[tray_id]
        tx = tray["pos"][0] + (0.03 if k % 2 else -0.03)
        ty = tray["pos"][1] + (0.03 if k >= 2 else -0.03) * (1 if k % 2 else -1)
        self.tray_fill[tray_id] += 1

        self.bottom_lines = caption
        self.gripper_open()
        self.move_to([cx, cy, HOVER_Z], seconds=1.4)
        self.move_to([cx, cy, GRASP_Z], seconds=1.0)
        self.gripper_grasp()
        self.move_to([cx, cy, HOVER_Z], seconds=0.9)
        self.move_to([tx, ty, HOVER_Z], seconds=1.6)
        self.move_to([tx, ty, PLACE_Z], seconds=1.0)
        self.gripper_open()
        self.move_to([tx, ty, HOVER_Z], seconds=0.9)

    def run(self):
        # --- イントロ(タイトル表示+カメラが寄る)
        self.top_lines = ["AI×ロボティクス課題: 自然言語指示によるブロック仕分け"]
        self.bottom_lines = ["LLM (Claude API) がタスクをプランニング / Genesis シミュレータ (CPU)"]
        n_intro = int(5.0 / 0.01)
        p0, p1 = np.array([2.3, -1.6, 1.6]), np.array([1.7, -1.1, 1.15])
        for i in range(n_intro):
            a = (i + 1) / n_intro
            self.cam.set_pose(pos=tuple(p0 * (1 - a) + p1 * a), lookat=(0.45, 0.05, 0.12))
            self.step()

        # --- LLM プランニング
        sentences = [s + "。" for s in INSTRUCTION.rstrip("。").split("。")]
        self.top_lines = [f"指示:「{sentences[0]}"] + [f"    {s}" for s in sentences[1:-1]] + [f"    {sentences[-1]}」"]
        self.bottom_lines = ["シーン状態(JSON)と指示文を LLM に送信し、実行計画を推論中..."]
        self.hold(3.0)

        state = self.scene_state()
        if self.use_llm:
            plan: SortPlan = make_plan(state, INSTRUCTION)
        else:
            plan = make_plan_offline(state, INSTRUCTION)
        print("=== LLM が生成した計画 ===")
        print(plan.model_dump_json(indent=2))
        with open("output/plan.json", "w") as f:
            f.write(plan.model_dump_json(indent=2))

        self.bottom_lines = wrap_jp(f"LLMの解釈: {plan.summary}") + [
            f"  計画{i + 1}: {s.cube_id} → {s.tray_id}({s.reason}" + ")"
            for i, s in enumerate(plan.steps)
        ]
        self.hold(9.0)

        # --- 計画の実行
        for i, s in enumerate(plan.steps):
            self.pick_and_place(
                s.cube_id,
                s.tray_id,
                [f"Step {i + 1}/{len(plan.steps)}: {s.cube_id} → {s.tray_id}", f"  理由: {s.reason}"],
            )

        # --- アウトロ(最低でも動画全体が65秒になるまで表示)
        untouched = [c["id"] for c in CUBES if c["id"] not in {s.cube_id for s in plan.steps}]
        self.bottom_lines = [
            "タスク完了: LLM の計画どおりに仕分けしました。",
            f"指示の制約に従い {', '.join(untouched)} には触れていません。" if untouched else "",
        ]
        remain = max(65.0 - self.frames_written / FPS, 4.0)
        self.hold(remain)

        self.writer.close()
        print(f"動画を保存しました: {self.frames_written} frames ({self.frames_written / FPS:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="AI不使用の固定ルールで動作確認")
    ap.add_argument("--out", default="output/video.mp4")
    ap.add_argument("--res", default="1280x720")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    w, h = (int(v) for v in args.res.split("x"))
    sim = SortingSim(res=(w, h), out_path=args.out, use_llm=not args.offline)
    sim.run()


if __name__ == "__main__":
    main()
