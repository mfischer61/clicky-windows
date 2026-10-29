"""
Mini chat bar — small, mostly-transparent box at the bottom-center of the
screen. Ctrl+Alt+Space shows it, Ctrl+Alt+Space again (or Esc) hides it
completely.

  Enter         send + screenshot of your screen (bar hides itself for the grab)
  Shift+Enter   send text only, no screenshot
  Esc           hide
  ↺ button      start a fresh conversation

Settings (.env, optional):
  CLICKY_BAR_WIDTH          default 560
  CLICKY_BAR_IDLE_OPACITY   opacity when not focused (default 0.35)
  CLICKY_BAR_ACTIVE_OPACITY opacity while typing / hovering (default 0.92)
"""

import os
import sys
import threading

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QEvent
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel,
    QComboBox, QPushButton, QScrollArea, QFrame,
)

from config import cfg


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


BAR_W          = int(_f("CLICKY_BAR_WIDTH", 560))
IDLE_OPACITY   = max(0.08, min(1.0, _f("CLICKY_BAR_IDLE_OPACITY", 0.35)))
ACTIVE_OPACITY = max(0.2, min(1.0, _f("CLICKY_BAR_ACTIVE_OPACITY", 0.92)))
INPUT_H        = 40
MAX_ANSWER_H   = 280
BOTTOM_GAP     = 14
GRAB_DELAY_MS  = 90         # time for the compositor to drop the bar from screen

_GEMINI_SKIP = ("tts", "embedding", "image", "live", "audio", "aqa", "gemma",
                "robotics", "computer-use", "native")


def _force_foreground(hwnd: int):
    """Windows won't let a background app take focus with a plain
    activateWindow(). Briefly attach to the foreground thread's input queue
    so SetForegroundWindow is honoured."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes
        u, k = ctypes.windll.user32, ctypes.windll.kernel32
        u.GetForegroundWindow.restype = wintypes.HWND
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.c_void_p]
        u.GetWindowThreadProcessId.restype = wintypes.DWORD
        u.SetForegroundWindow.argtypes = [wintypes.HWND]
        u.BringWindowToTop.argtypes = [wintypes.HWND]
        u.SetFocus.argtypes = [wintypes.HWND]
        u.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
        fg = u.GetForegroundWindow()
        fg_tid = u.GetWindowThreadProcessId(fg, None) if fg else 0
        me = k.GetCurrentThreadId()
        attached = bool(fg_tid and fg_tid != me and u.AttachThreadInput(fg_tid, me, True))
        try:
            u.BringWindowToTop(hwnd)
            u.SetForegroundWindow(hwnd)
            u.SetFocus(hwnd)
        finally:
            if attached:
                u.AttachThreadInput(fg_tid, me, False)
    except Exception:
        pass


class _Input(QLineEdit):
    """Line edit that reports Enter vs Shift+Enter and Esc."""
    submitted = pyqtSignal(str, bool)     # (text, with_screenshot)
    escape    = pyqtSignal()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            shot = not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.submitted.emit(self.text(), shot)
            return
        if e.key() == Qt.Key.Key_Escape:
            self.escape.emit()
            return
        super().keyPressEvent(e)


class MiniBar(QWidget):
    # (text, [jpeg_b64, ...]) → manager.ask_check
    on_check          = pyqtSignal(str, list)
    # (provider, model) — provider switch handled by main
    on_model_selected = pyqtSignal(str, str)
    on_new_chat       = pyqtSignal()
    on_course_selected = pyqtSignal(str)   # course notes file stem, '' = none
    on_shown          = pyqtSignal()
    on_stop           = pyqtSignal()
    sig_toggle        = pyqtSignal()       # thread-safe: emit from hotkey thread
    _sig_ollama       = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self._answer = ""
        self._busy = False
        self._hover = False
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool                 # no taskbar button
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
        self.setFixedWidth(BAR_W)
        self._build()
        self.sig_toggle.connect(self.toggle)
        self._sig_ollama.connect(self._add_ollama_models)
        self._populate_models()
        self._resize_to_content()
        self.setWindowOpacity(IDLE_OPACITY)

    # ── layout ────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 10, 8)
        root.setSpacing(6)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }"
            "QScrollBar:vertical { width: 6px; background: transparent; }"
            "QScrollBar::handle:vertical { background: rgba(255,255,255,60); border-radius: 3px; }"
            "QScrollBar::add-line, QScrollBar::sub-line { height: 0; }"
        )
        self._answer_lbl = QLabel()
        self._answer_lbl.setWordWrap(True)
        self._answer_lbl.setTextFormat(Qt.TextFormat.PlainText)
        self._answer_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._answer_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._answer_lbl.setFont(QFont("Segoe UI", 10))
        self._answer_lbl.setStyleSheet("color: rgb(236,236,242); background: transparent;")
        self._scroll.setWidget(self._answer_lbl)
        self._scroll.setVisible(False)
        root.addWidget(self._scroll)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._input = _Input()
        self._input.setPlaceholderText("Ask, or Enter to check the screen…")
        self._input.setFixedHeight(INPUT_H - 12)
        self._input.setFont(QFont("Segoe UI", 10))
        self._input.setStyleSheet(
            "QLineEdit { background: transparent; border: none; color: rgb(240,240,245);"
            " selection-background-color: rgba(120,140,255,120); }"
        )
        self._input.submitted.connect(self._submit)
        self._input.escape.connect(self._escape)
        row.addWidget(self._input, stretch=1)

        self._model = QComboBox()
        self._model.setFixedHeight(INPUT_H - 16)
        self._model.setMaximumWidth(170)
        self._model.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._model.setStyleSheet(
            "QComboBox { background: rgba(255,255,255,18); border: none; border-radius: 6px;"
            " color: rgb(200,200,212); padding: 0 6px; font-size: 11px; }"
            "QComboBox::drop-down { width: 0; border: none; }"
            "QComboBox QAbstractItemView { background: rgb(28,28,34); color: rgb(225,225,235);"
            " selection-background-color: rgb(60,70,120); border: 1px solid rgb(60,60,72); }"
        )
        self._model.currentIndexChanged.connect(self._model_changed)
        row.addWidget(self._model)

        # Course picker — course_notes/<NAME>.md is sent with every question
        self._course = QComboBox()
        self._course.setFixedHeight(INPUT_H - 16)
        self._course.setMaximumWidth(110)
        self._course.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._course.setToolTip("Course notes sent with each question")
        self._course.setStyleSheet(self._model.styleSheet())
        self._populate_courses()
        self._course.currentIndexChanged.connect(
            lambda _i: self.on_course_selected.emit(self._course.currentData() or ""))
        row.insertWidget(row.indexOf(self._model), self._course)

        self._new_btn = QPushButton("↺")
        self._new_btn.setFixedSize(INPUT_H - 16, INPUT_H - 16)
        self._new_btn.setToolTip("New conversation")
        self._new_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._new_btn.setStyleSheet(
            "QPushButton { background: transparent; color: rgb(170,170,185); border: none; font-size: 14px; }"
            "QPushButton:hover { color: white; }"
        )
        self._new_btn.clicked.connect(self._new_chat)
        row.addWidget(self._new_btn)
        root.addLayout(row)

    # ── courses ───────────────────────────────────────────────────────────

    def _populate_courses(self):
        from pathlib import Path
        folder = cfg.env_path().parent / "course_notes"
        names = sorted(p.stem for p in folder.glob("*.md")) if folder.exists() else []
        self._course.blockSignals(True)
        self._course.clear()
        self._course.addItem("No course", userData="")
        for n in names:
            self._course.addItem(n, userData=n)
        cur = os.getenv("CLICKY_COURSE", "").strip()
        i = self._course.findData(cur)
        self._course.setCurrentIndex(i if i >= 0 else 0)
        self._course.blockSignals(False)

    # ── models ────────────────────────────────────────────────────────────

    def _populate_models(self):
        self._model.blockSignals(True)
        self._model.clear()
        avail = cfg.available_llm_providers()
        if "claude_cli" in avail:
            from ai.claude_cli_provider import MODELS
            for mid, label in MODELS:
                self._model.addItem(label, userData=("claude_cli", mid))
        if "agy" in avail:
            from ai.agy_provider import models as _agy_models
            for mid in _agy_models():
                label = "Gemini (Pro plan)" if mid == "default" else f"{mid} (Pro plan)"
                self._model.addItem(label, userData=("agy", mid))
        if "gemini" in avail:
            for mid in self._gemini_ids():
                self._model.addItem(mid, userData=("gemini", mid))
        vm = cfg.get_ollama_model("vision")
        if vm:
            self._model.addItem(f"Local · {vm}", userData=("ollama", vm))
        # Select whatever provider/model is active now
        active = cfg.llm_provider()
        for i in range(self._model.count()):
            if self._model.itemData(i)[0] == active:
                self._model.setCurrentIndex(i)
                break
        self._model.blockSignals(False)
        # Discover other installed local models without blocking the UI
        threading.Thread(target=self._probe_ollama, daemon=True).start()

    @staticmethod
    def _gemini_ids(limit: int = 4) -> list:
        """Gemini models shown in the menu. Override with
        CLICKY_GEMINI_MODELS=id1,id2 in .env. Default list = models that work
        on a free-tier key (2.5 models are retired for new users; Pro models
        have no free quota)."""
        override = [m.strip() for m in os.getenv("CLICKY_GEMINI_MODELS", "").split(",") if m.strip()]
        if override:
            return override
        return ["gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-flash-latest"][:limit]

    def _probe_ollama(self):
        try:
            from ai.ollama_bootstrap import list_installed_models
            names = list_installed_models()
        except Exception:
            names = []
        if names:
            self._sig_ollama.emit(names)

    def _add_ollama_models(self, names: list):
        have = {self._model.itemData(i) for i in range(self._model.count())}
        hints = ("vl", "llava", "vision", "gemma3", "minicpm-v", "moondream", "llama3.2-vision")
        self._model.blockSignals(True)
        for n in names:
            if ("ollama", n) in have or not any(h in n.lower() for h in hints):
                continue                          # screenshots need a vision model
            self._model.addItem(f"Local · {n}", userData=("ollama", n))
        self._model.blockSignals(False)

    def _model_changed(self, _idx: int):
        data = self._model.currentData()
        if data:
            self.on_model_selected.emit(data[0], data[1])

    def current_selection(self):
        return self._model.currentData()

    # ── show / hide ───────────────────────────────────────────────────────

    def toggle(self):
        if self.isVisible():
            self.hide_bar()
        else:
            self.show_bar()

    def show_bar(self):
        self._place()
        self.setWindowOpacity(ACTIVE_OPACITY)
        self.show()
        self.raise_()
        self.activateWindow()
        _force_foreground(int(self.winId()))
        self._input.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._input.selectAll()
        self.on_shown.emit()

    def hide_bar(self):
        self.hide()

    def _escape(self):
        if self._busy:
            self.on_stop.emit()
        self.hide_bar()

    def _place(self):
        scr = QApplication.screenAt(self.cursor().pos()) or QApplication.primaryScreen()
        g = scr.availableGeometry()            # excludes the taskbar
        x = g.x() + (g.width() - self.width()) // 2
        y = g.y() + g.height() - self.height() - BOTTOM_GAP
        self.move(x, y)

    def _resize_to_content(self):
        bottom = self.geometry().bottom() if self.isVisible() else None
        if self._answer:
            self._scroll.setVisible(True)
            self._answer_lbl.setFixedWidth(BAR_W - 34)
            h = self._answer_lbl.heightForWidth(BAR_W - 34)
            self._scroll.setFixedHeight(max(20, min(MAX_ANSWER_H, h + 4)))
        else:
            self._scroll.setVisible(False)
        self.adjustSize()
        self.setFixedHeight(self.sizeHint().height())
        if bottom is not None:                 # grow upward, keep the bottom edge
            self.move(self.x(), bottom - self.height() + 1)
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── opacity: faint when you're back on your test, solid when using it ──

    def _update_opacity(self):
        active = self.isActiveWindow() or self._hover
        self.setWindowOpacity(ACTIVE_OPACITY if active else IDLE_OPACITY)
        self.update()

    def changeEvent(self, e):
        if e.type() == QEvent.Type.ActivationChange:
            self._update_opacity()
        super().changeEvent(e)

    def enterEvent(self, e):
        self._hover = True
        self._update_opacity()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self._update_opacity()
        super().leaveEvent(e)

    # ── sending ───────────────────────────────────────────────────────────

    def _submit(self, text: str, with_shot: bool):
        if self._busy:
            return
        text = text.strip()
        if not text and not with_shot:
            return
        self._input.clear()
        self._answer = ""
        self._answer_lbl.setText("")
        if not with_shot:
            self._send(text, [])
            return
        # Drop the bar off-screen for one frame so it isn't in the screenshot
        self._restore_opacity = self.windowOpacity()
        self.setWindowOpacity(0.0)
        QTimer.singleShot(GRAB_DELAY_MS, lambda: self._grab_and_send(text))

    def _grab_and_send(self, text: str):
        images = []
        try:
            from screen.capture import capture_all_screens
            images = [s.base64_jpeg for s in capture_all_screens()]
        except Exception as e:
            self.append_chunk(f"[screenshot failed: {e}]\n")
        finally:
            self.setWindowOpacity(getattr(self, "_restore_opacity", ACTIVE_OPACITY))
        self._send(text, images)

    def _send(self, text: str, images: list):
        self.set_busy(True)
        self.on_check.emit(text, images)

    def _new_chat(self):
        self._answer = ""
        self._answer_lbl.setText("")
        self._resize_to_content()
        self.on_new_chat.emit()
        self._input.setFocus()

    # ── manager → bar ─────────────────────────────────────────────────────

    def set_busy(self, busy: bool):
        self._busy = busy
        self._input.setPlaceholderText(
            "Thinking…  (Esc to stop)" if busy else "Ask, or Enter to check the screen…")
        self.update()

    def set_state_name(self, name: str):
        self.set_busy(name in ("THINKING", "LISTENING", "SPEAKING"))

    def append_chunk(self, chunk: str):
        self._answer += chunk
        self._answer_lbl.setText(self._answer.strip())
        self._resize_to_content()

    # ── paint: dark frosted pill ──────────────────────────────────────────

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(16, 16, 20, 200)))
        edge = QColor(130, 150, 255, 110) if self._busy else QColor(255, 255, 255, 34)
        p.setPen(QPen(edge, 1))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 14, 14)
        p.end()
