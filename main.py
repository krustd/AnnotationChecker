"""标注质量检查工具 - 用于检查姿态标注的质量和一致性。"""

import json
import random
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("QtAgg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# ── 常量 ──────────────────────────────────────────────────────────────
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]

SKELETON_COLORS = {
    # 头部/躯干中线 - 绿色
    (0, 1): (100, 200, 100, 150), (0, 2): (100, 200, 100, 150),
    (5, 6): (100, 200, 100, 150), (11, 12): (100, 200, 100, 150),
    # 左侧 - 暖色
    (1, 3): (255, 120, 100, 150), (5, 7): (255, 120, 100, 150),
    (7, 9): (255, 150, 80, 150), (5, 11): (230, 100, 70, 150),
    (11, 13): (230, 120, 80, 150), (13, 15): (240, 150, 90, 150),
    # 右侧 - 冷色
    (2, 4): (80, 160, 255, 150), (6, 8): (80, 160, 255, 150),
    (8, 10): (110, 190, 255, 150), (6, 12): (70, 120, 230, 150),
    (12, 14): (90, 140, 240, 150), (14, 16): (120, 170, 245, 150),
}

KEYPOINT_COLORS = {
    0: (100, 220, 100),    # nose
    1: (255, 120, 120),    # left_eye
    2: (100, 180, 255),    # right_eye
    3: (255, 80, 80),      # left_ear
    4: (60, 140, 255),     # right_ear
    5: (255, 100, 50),     # left_shoulder
    6: (80, 120, 255),     # right_shoulder
    7: (255, 140, 60),     # left_elbow
    8: (100, 160, 240),    # right_elbow
    9: (255, 180, 80),     # left_wrist
    10: (130, 200, 255),   # right_wrist
    11: (220, 80, 60),     # left_hip
    12: (80, 80, 220),     # right_hip
    13: (230, 120, 80),    # left_knee
    14: (100, 120, 230),   # right_knee
    15: (240, 160, 100),   # left_ankle
    16: (140, 160, 240),   # right_ankle
}

SCORE_NAMES = ("novelty", "environment_interaction", "person_fit")
SCORE_LABELS = {"novelty": "姿势新奇度", "environment_interaction": "环境互动性", "person_fit": "人物契合度"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── 数据加载 ──────────────────────────────────────────────────────────
def load_project(project_dir: Path) -> Tuple[Dict[str, dict], Dict[str, Path]]:
    """加载项目的所有标注和图像路径。返回 (annotations_dict, image_paths_dict)"""
    annotations = {}
    image_paths = {}

    img_dir = project_dir / "images"
    json_dir = project_dir / "annotations"

    if not json_dir.exists():
        return annotations, image_paths

    # 建立图像路径索引
    if img_dir.exists():
        for f in img_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths[f.stem] = f

    # 加载所有标注
    for jf in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text("utf-8"))
            if isinstance(data, list) and data:
                data = data[0]
            annotations[jf.stem] = data
        except Exception:
            continue

    return annotations, image_paths


_IGNORE_DIRS = {"inpainting"}


def _extract_images_from_archive(archive_path: Path, dest_dir: Path):
    """从压缩包（zip/tar）中提取图像文件，忽略inpainting目录和非图像文件。"""
    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = Path(info.filename).parts
                if _IGNORE_DIRS & {p.lower() for p in parts}:
                    continue
                name = Path(info.filename).name
                if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                    dest = dest_dir / name
                    dest.write_bytes(zf.read(info.filename))
    elif suffix in (".tar", ".gz", ".tgz", ".bz2", ".xz"):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                parts = Path(member.name).parts
                if _IGNORE_DIRS & {p.lower() for p in parts}:
                    continue
                name = Path(member.name).name
                if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                    data = tf.extractfile(member).read()
                    dest = dest_dir / name
                    dest.write_bytes(data)


def _extract_jsons_from_archive(archive_path: Path, dest_dir: Path):
    """从压缩包（zip/tar）中提取json文件。"""
    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if Path(name).suffix.lower() == ".json":
                    dest = dest_dir / name
                    dest.write_bytes(zf.read(info.filename))
    elif suffix in (".tar", ".gz", ".tgz", ".bz2", ".xz"):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if Path(name).suffix.lower() == ".json":
                    data = tf.extractfile(member).read()
                    dest = dest_dir / name
                    dest.write_bytes(data)


def load_from_archives(
    image_archive_path: Path, annotation_archive_path: Path, tmp_dir: Path
) -> Tuple[Dict[str, dict], Dict[str, Path]]:
    """从两个压缩包加载：图片包中提取图像（忽略json），标注包中提取json。"""
    annotations = {}
    image_paths = {}

    img_dir = tmp_dir / "images"
    json_dir = tmp_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    _extract_images_from_archive(image_archive_path, img_dir)
    _extract_jsons_from_archive(annotation_archive_path, json_dir)

    # 建立图像路径索引
    for f in img_dir.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths[f.stem] = f

    # 加载所有标注
    for jf in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text("utf-8"))
            if isinstance(data, list) and data:
                data = data[0]
            annotations[jf.stem] = data
        except Exception:
            continue

    return annotations, image_paths


def group_skipped_by_reason(
    annotations: Dict[str, dict], image_paths: Dict[str, Path]
) -> Dict[str, List[str]]:
    """从 JSON 标注的 skip_reason 字段分类跳过的图像，返回 {原因: [stem, ...]}。"""
    result: Dict[str, List[str]] = {}
    for stem, ann in annotations.items():
        reason = ann.get("skip_reason", "")
        if reason:
            result.setdefault(reason, []).append(stem)
    return result


# ── 绘制关键点到图像上 ──────────────────────────────────────────────
def draw_pose_on_pixmap(pixmap: QPixmap, annotation: dict) -> QPixmap:
    """在 QPixmap 上绘制骨骼和关键点，返回新的 QPixmap。
    如果关键点超出图像边界，自动扩展画布以容纳所有点。"""
    kps = annotation.get("keypoints", [])
    visibility = annotation.get("visibility", [])
    if not kps:
        return pixmap.copy()

    img_w, img_h = pixmap.width(), pixmap.height()

    # 计算所有关键点的边界，确定画布需要多大
    all_xs = [x for x, y in kps]
    all_ys = [y for x, y in kps]

    margin = max(20, int(min(img_w, img_h) * 0.02))
    min_x = min(0, min(all_xs) - margin)
    min_y = min(0, min(all_ys) - margin)
    max_x = max(img_w, max(all_xs) + margin)
    max_y = max(img_h, max(all_ys) + margin)

    # 偏移量：原图在新画布上的位置
    off_x = int(-min_x)
    off_y = int(-min_y)
    canvas_w = int(max_x - min_x)
    canvas_h = int(max_y - min_y)

    # 创建扩展画布，将原图绘制到正确位置
    result = QPixmap(canvas_w, canvas_h)
    result.fill(QColor(30, 30, 30))
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.drawPixmap(off_x, off_y, pixmap)

    radius = 5.0
    pen_width = 1.5

    # 绘制骨骼
    for s, e in SKELETON:
        if s < len(kps) and e < len(kps):
            sx, sy = kps[s]
            ex, ey = kps[e]
            r, g, b, a = SKELETON_COLORS.get((s, e), (100, 200, 100, 150))
            painter.setPen(QPen(QColor(r, g, b, a), 2))
            painter.drawLine(
                QPointF(sx + off_x, sy + off_y),
                QPointF(ex + off_x, ey + off_y),
            )

    # 绘制关键点
    for i, (x, y) in enumerate(kps):
        vis = visibility[i] if i < len(visibility) else 1
        r, g, b = KEYPOINT_COLORS.get(i, (200, 200, 200))
        fill_color = QColor(r, g, b)
        px, py = x + off_x, y + off_y

        if vis == 1:
            # 可见 - 实心圆
            painter.setPen(QPen(QColor(0, 0, 0), pen_width))
            painter.setBrush(QBrush(fill_color))
            painter.drawEllipse(QPointF(px, py), radius, radius)
        else:
            # 遮挡 - X 标记
            cross_size = radius * 0.9
            cross_pen = QPen(fill_color, pen_width * 2)
            cross_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(cross_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(
                QPointF(px - cross_size, py - cross_size),
                QPointF(px + cross_size, py + cross_size),
            )
            painter.drawLine(
                QPointF(px - cross_size, py + cross_size),
                QPointF(px + cross_size, py - cross_size),
            )

    painter.end()
    return result


# ── 图像缩略图 Widget ──────────────────────────────────────────────
class ImageThumbnail(QLabel):
    """可点击的图像缩略图标签。"""
    clicked = Signal(str)

    def __init__(self, size: int = 200):
        super().__init__()
        self.image_path = ""
        self.thumb_size = size
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #555; background: #222;")

    def set_image(self, path: str, label_text: str = "", annotation: dict = None):
        self.image_path = path
        pm = QPixmap(path)
        if pm.isNull():
            self.setText("加载失败")
            return
        if annotation:
            pm = draw_pose_on_pixmap(pm, annotation)
        pm = pm.scaled(self.thumb_size, self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pm)
        self.setToolTip(label_text or Path(path).name)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image_path:
            self.clicked.emit(self.image_path)


# ── 大图预览窗口 ──────────────────────────────────────────────────
class ImagePreviewWindow(QMainWindow):
    """全尺寸图像预览窗口。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像预览")
        self.resize(1000, 800)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background: #111;")
        scroll = QScrollArea()
        scroll.setWidget(self.label)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

    def show_image(self, path: str, annotation: dict = None):
        pm = QPixmap(path)
        if pm.isNull():
            return
        if annotation:
            pm = draw_pose_on_pixmap(pm, annotation)
        self.label.setPixmap(pm)
        self.setWindowTitle(Path(path).name)
        self.show()
        self.raise_()


# ── Tab 1: 跳过原因分类浏览 ──────────────────────────────────────
class SkipReasonTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.preview_window = ImagePreviewWindow(self)
        layout = QVBoxLayout(self)

        # 控制栏
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("跳过类别:"))
        self.category_combo = QComboBox()
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        ctrl.addWidget(self.category_combo)
        self.count_label = QLabel()
        ctrl.addWidget(self.count_label)
        self.page_label = QLabel()
        ctrl.addWidget(self.page_label)
        self.prev_btn = QPushButton("上一页")
        self.prev_btn.clicked.connect(lambda: self._change_page(-1))
        ctrl.addWidget(self.prev_btn)
        self.next_btn = QPushButton("下一页")
        self.next_btn.clicked.connect(lambda: self._change_page(1))
        ctrl.addWidget(self.next_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # 图像网格
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(4)
        scroll = QScrollArea()
        scroll.setWidget(self.grid_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        self.skip_data: Dict[str, List[str]] = {}  # {reason: [stem, ...]}
        self.image_paths: Dict[str, Path] = {}
        self.current_stems: List[str] = []
        self.current_page = 0
        self.cols = 6
        self.rows = 4
        self.per_page = self.cols * self.rows

    def load_data(self, annotations: Dict[str, dict], image_paths: Dict[str, Path]):
        self.image_paths = image_paths
        self.skip_data = group_skipped_by_reason(annotations, image_paths)
        self.category_combo.clear()
        for cat, stems in sorted(self.skip_data.items()):
            self.category_combo.addItem(f"{cat} ({len(stems)})")
        if not self.skip_data:
            self.count_label.setText("没有跳过的图像")

    def _on_category_changed(self, text: str):
        cat = text.split(" (")[0] if " (" in text else text
        self.current_stems = self.skip_data.get(cat, [])
        self.current_page = 0
        self._show_page()

    def _change_page(self, delta: int):
        max_page = max(0, (len(self.current_stems) - 1) // self.per_page)
        self.current_page = max(0, min(max_page, self.current_page + delta))
        self._show_page()

    def _show_page(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        start = self.current_page * self.per_page
        end = min(start + self.per_page, len(self.current_stems))
        total_pages = max(1, (len(self.current_stems) + self.per_page - 1) // self.per_page)
        self.page_label.setText(f"第 {self.current_page + 1}/{total_pages} 页")
        self.count_label.setText(f"共 {len(self.current_stems)} 张")

        for idx, stem in enumerate(self.current_stems[start:end]):
            row, col = divmod(idx, self.cols)
            thumb = ImageThumbnail(size=180)
            img_path = self.image_paths.get(stem)
            if img_path:
                thumb.set_image(str(img_path), stem)
                thumb.clicked.connect(lambda p=str(img_path): self.preview_window.show_image(p))
            else:
                thumb.setText(f"{stem}\n(无图像)")
            self.grid_layout.addWidget(thumb, row, col)


# ── Tab 2: 分数分布直方图 ──────────────────────────────────────────
class ScoreDistributionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(12, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

    def load_data(self, annotations: Dict[str, dict]):
        self.figure.clear()
        scores_data = {name: [] for name in SCORE_NAMES}

        for ann in annotations.values():
            if ann.get("skip_reason", ""):
                continue
            for name in SCORE_NAMES:
                v = ann.get(name, -1)
                if v >= 0:
                    scores_data[name].append(v)

        axes = self.figure.subplots(1, 3)
        stats_parts = []
        for i, name in enumerate(SCORE_NAMES):
            ax = axes[i]
            data = scores_data[name]
            if data:
                bins = np.arange(-0.5, 6.5, 1)
                ax.hist(data, bins=bins, color="#4a90d9", edgecolor="white", alpha=0.85)
                ax.set_xticks(range(6))
                mean_v = np.mean(data)
                std_v = np.std(data)
                ax.axvline(mean_v, color="red", linestyle="--", label=f"均值={mean_v:.2f}")
                ax.legend(fontsize=8)
                stats_parts.append(f"{SCORE_LABELS[name]}: N={len(data)}, 均值={mean_v:.2f}, 标准差={std_v:.2f}")
            ax.set_title(SCORE_LABELS[name], fontsize=11)
            ax.set_xlabel("分数")
            ax.set_ylabel("数量")

        self.figure.tight_layout()
        self.canvas.draw()
        self.stats_label.setText("  |  ".join(stats_parts))


# ── Tab 3: 评分图像矩阵 ──────────────────────────────────────────
class ScoreMatrixTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.preview_window = ImagePreviewWindow(self)
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("评分项:"))
        self.score_combo = QComboBox()
        for name in SCORE_NAMES:
            self.score_combo.addItem(SCORE_LABELS[name], name)
        ctrl.addWidget(self.score_combo)
        self.resample_btn = QPushButton("重新抽样")
        self.resample_btn.clicked.connect(self._resample)
        ctrl.addWidget(self.resample_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # 列标题
        header = QHBoxLayout()
        for s in range(6):
            lbl = QLabel(f"分数 = {s}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedWidth(185)
            header.addWidget(lbl)
        header.addStretch()
        layout.addLayout(header)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(4)
        scroll = QScrollArea()
        scroll.setWidget(self.grid_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        self.annotations: Dict[str, dict] = {}
        self.image_paths: Dict[str, Path] = {}

    def load_data(self, annotations: Dict[str, dict], image_paths: Dict[str, Path]):
        self.annotations = annotations
        self.image_paths = image_paths
        self._resample()

    def _resample(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        score_key = self.score_combo.currentData()
        if not score_key:
            return

        # 按分数分组
        by_score: Dict[int, List[str]] = {s: [] for s in range(6)}
        for stem, ann in self.annotations.items():
            if ann.get("skip_reason", ""):
                continue
            v = ann.get(score_key, -1)
            if 0 <= v <= 5 and stem in self.image_paths:
                by_score[v].append(stem)

        # 4行6列: 每个分数随机抽4张
        for col in range(6):
            candidates = by_score[col]
            sampled = random.sample(candidates, min(4, len(candidates)))
            for row in range(4):
                if row < len(sampled):
                    stem = sampled[row]
                    thumb = ImageThumbnail(size=180)
                    img_path = str(self.image_paths[stem])
                    thumb.set_image(img_path, f"{SCORE_LABELS[score_key]}={col}\n{stem}")
                    thumb.clicked.connect(lambda p=img_path: self.preview_window.show_image(p))
                    self.grid_layout.addWidget(thumb, row, col)
                else:
                    placeholder = QLabel("无图像")
                    placeholder.setFixedSize(180, 180)
                    placeholder.setAlignment(Qt.AlignCenter)
                    placeholder.setStyleSheet("border: 1px solid #555; background: #222; color: #888;")
                    self.grid_layout.addWidget(placeholder, row, col)


# ── Tab 4: 修正图像抽查 ──────────────────────────────────────────
class CorrectionCheckTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # 控制栏
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("抽样数量:"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 100)
        self.sample_spin.setValue(20)
        ctrl.addWidget(self.sample_spin)
        self.resample_btn = QPushButton("重新抽样")
        self.resample_btn.clicked.connect(self._resample)
        ctrl.addWidget(self.resample_btn)

        self.prev_btn = QPushButton("◀ 上一张")
        self.prev_btn.clicked.connect(lambda: self._change_image(-1))
        ctrl.addWidget(self.prev_btn)
        self.next_btn = QPushButton("下一张 ▶")
        self.next_btn.clicked.connect(lambda: self._change_image(1))
        ctrl.addWidget(self.next_btn)

        self.index_label = QLabel()
        ctrl.addWidget(self.index_label)
        self.legend_label = QLabel("图例: ● 实心圆=可见 | ✕ 叉号=遮挡 | 暖色=左侧 | 冷色=右侧")
        self.legend_label.setStyleSheet("color: #aaa; margin-left: 20px;")
        ctrl.addWidget(self.legend_label)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # 文件名标签
        self.name_label = QLabel()
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("font-size: 14px; padding: 4px;")
        layout.addWidget(self.name_label)

        # 图像显示区域 - 不限制大小，用 ScrollArea 包裹
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #111;")
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        self.annotations: Dict[str, dict] = {}
        self.image_paths: Dict[str, Path] = {}
        self.sampled_stems: List[str] = []
        self.current_index = 0

    def load_data(self, annotations: Dict[str, dict], image_paths: Dict[str, Path]):
        self.annotations = annotations
        self.image_paths = image_paths
        self._resample()

    def _resample(self):
        # 只选取未跳过且有有效关键点的图像
        candidates = []
        for stem, ann in self.annotations.items():
            if ann.get("skip_reason", ""):
                continue
            if stem not in self.image_paths:
                continue
            kps = ann.get("keypoints", [])
            if kps:
                candidates.append(stem)

        n = min(self.sample_spin.value(), len(candidates))
        self.sampled_stems = random.sample(candidates, n) if candidates else []
        self.current_index = 0
        self._show_current()

    def _change_image(self, delta: int):
        if not self.sampled_stems:
            return
        self.current_index = max(0, min(len(self.sampled_stems) - 1, self.current_index + delta))
        self._show_current()

    def _show_current(self):
        if not self.sampled_stems:
            self.image_label.setText("没有可显示的图像")
            self.index_label.setText("")
            self.name_label.setText("")
            return

        self.index_label.setText(f"{self.current_index + 1} / {len(self.sampled_stems)}")
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.sampled_stems) - 1)

        stem = self.sampled_stems[self.current_index]
        self.name_label.setText(stem)
        img_path = str(self.image_paths[stem])
        ann = self.annotations[stem]

        pm = QPixmap(img_path)
        if pm.isNull():
            self.image_label.setText("加载失败")
            return
        pm = draw_pose_on_pixmap(pm, ann)
        self.image_label.setPixmap(pm)
        self.image_label.adjustSize()


# ── 主窗口 ──────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("标注质量检查工具")
        self.resize(1300, 900)
        self._tmp_dir = None  # 保持临时目录引用，防止被回收

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 项目选择
        top = QHBoxLayout()
        self.path_label = QLabel("未选择项目")
        top.addWidget(self.path_label)
        self.open_folder_btn = QPushButton("打开项目文件夹")
        self.open_folder_btn.clicked.connect(self._open_project)
        top.addWidget(self.open_folder_btn)
        self.open_archive_btn = QPushButton("从压缩包导入")
        self.open_archive_btn.setToolTip("选择图片压缩包(tar/zip) + 标注压缩包(zip)，快速加载检查")
        self.open_archive_btn.clicked.connect(self._open_from_archives)
        top.addWidget(self.open_archive_btn)
        top.addStretch()
        main_layout.addLayout(top)

        # 标签页
        self.tabs = QTabWidget()
        self.tab_skip = SkipReasonTab()
        self.tab_dist = ScoreDistributionTab()
        self.tab_matrix = ScoreMatrixTab()
        self.tab_correction = CorrectionCheckTab()
        self.tabs.addTab(self.tab_skip, "1. 跳过原因分类")
        self.tabs.addTab(self.tab_dist, "2. 分数分布")
        self.tabs.addTab(self.tab_matrix, "3. 评分图像矩阵")
        self.tabs.addTab(self.tab_correction, "4. 修正抽查")
        main_layout.addWidget(self.tabs)

        self.statusBar().showMessage("请打开一个项目文件夹")

    def _load_into_tabs(self, annotations, image_paths):
        """加载数据到所有标签页。"""
        self.tab_skip.load_data(annotations, image_paths)
        self.tab_dist.load_data(annotations)
        self.tab_matrix.load_data(annotations, image_paths)
        self.tab_correction.load_data(annotations, image_paths)

    def _open_project(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择项目文件夹")
        if not dir_path:
            return
        project_dir = Path(dir_path)
        self.path_label.setText(str(project_dir))
        self.statusBar().showMessage("加载中...")
        QApplication.processEvents()

        annotations, image_paths = load_project(project_dir)
        self.statusBar().showMessage(
            f"已加载 {len(annotations)} 个标注, {len(image_paths)} 张图像"
        )
        self._load_into_tabs(annotations, image_paths)

        # 锁定另一个按钮
        self.open_archive_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)

    def _open_from_archives(self):
        archive_filter = "压缩包 (*.tar *.tar.gz *.tgz *.zip)"
        image_archive, _ = QFileDialog.getOpenFileName(
            self, "选择图片压缩包", "", archive_filter
        )
        if not image_archive:
            return

        annotation_archive, _ = QFileDialog.getOpenFileName(
            self, "选择标注压缩包（从PoseEditor导出）", "", "ZIP 文件 (*.zip)"
        )
        if not annotation_archive:
            return

        self.path_label.setText(
            f"图片: {Path(image_archive).name}  |  标注: {Path(annotation_archive).name}"
        )
        self.statusBar().showMessage("正在解压并加载...")
        QApplication.processEvents()

        try:
            self._tmp_dir = tempfile.TemporaryDirectory()
            tmp_path = Path(self._tmp_dir.name)

            annotations, image_paths = load_from_archives(
                Path(image_archive), Path(annotation_archive), tmp_path
            )
            self.statusBar().showMessage(
                f"已加载 {len(annotations)} 个标注, {len(image_paths)} 张图像"
            )
            self._load_into_tabs(annotations, image_paths)

            # 锁定另一个按钮
            self.open_folder_btn.setEnabled(False)
            self.open_archive_btn.setEnabled(False)
        except Exception as e:
            self.statusBar().showMessage(f"加载失败: {e}")


def main():
    matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "SimHei", "Arial Unicode MS"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 暗色主题
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(45, 45, 45))
    palette.setColor(palette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(palette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(palette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(palette.ColorRole.Button, QColor(55, 55, 55))
    palette.setColor(palette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(palette.ColorRole.Highlight, QColor(70, 130, 200))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
