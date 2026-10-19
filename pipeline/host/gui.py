"""Tkinter control panel for the live pipeline (P2.2 M2-lite).

Mirrors the FirstLight interface's basics — deck presets, model selection,
one-button start/stop with auto-reload between matches:
  - deck presets are read straight from FirstLight's own saved decks
    (%LOCALAPPDATA%/FirstLight_CR/custom_decks.json) and passed to the
    orchestrator in memory; my_deck.json stays the fallback default
  - models come from FirstLight's checkpoints/manifest.json and are passed
    to PolicyClient(checkpoint=...)
  - continuous mode keeps re-arming matches until 停止 is pressed

Run (FirstLight venv): python -I -m pipeline.host.gui
"""
from __future__ import annotations

import json
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from pipeline.act.orchestrator import run_live
from pipeline.adapt.firstlight_adapter import FIRSTLIGHT_ROOT

CUSTOM_DECKS = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "FirstLight_CR" / "custom_decks.json"
CHECKPOINT_MANIFEST = FIRSTLIGHT_ROOT / "checkpoints" / "manifest.json"
DEFAULT_MODEL = "General/checkpoint-step-00000460.pt"
DECK_FILE_LABEL = "my_deck.json(默认)"


def load_saved_decks() -> dict[str, tuple[int, ...]]:
    """FirstLight interface presets: name -> 8 card ids."""
    try:
        data = json.loads(CUSTOM_DECKS.read_text(encoding="utf-8"))
        return {item["name"]: tuple(int(cid) for cid in item["deck"])
                for item in data.get("decks", [])
                if isinstance(item.get("deck"), list) and len(item["deck"]) == 8}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def discover_models() -> tuple[str, ...]:
    try:
        manifest = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
        return tuple(item["path"] for item in manifest.get("models", []) if "path" in item)
    except (OSError, ValueError, KeyError, TypeError):
        return ()


class PipelineGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("crai pipeline")
        self._log_q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._saved_decks: dict[str, tuple[int, ...]] = {}
        self._build()
        self._reload_presets()
        self._poll_log()

    # -- layout ------------------------------------------------------------

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(5, weight=1)

        ttk.Label(main, text="卡组预设:").grid(row=0, column=0, sticky="w")
        self.deck_box = ttk.Combobox(main, state="readonly", width=26)
        self.deck_box.grid(row=0, column=1, sticky="we")
        self.deck_box.bind("<<ComboboxSelected>>", self._on_deck_change)
        ttk.Button(main, text="刷新", width=6, command=self._reload_presets).grid(row=0, column=2, padx=(6, 0))
        self.deck_info = ttk.Label(main, text="")
        self.deck_info.grid(row=1, column=0, columnspan=3, sticky="w")

        ttk.Label(main, text="推理模型:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.model_box = ttk.Combobox(main, state="readonly", width=42)
        self.model_box.grid(row=2, column=1, columnspan=2, sticky="we", pady=(6, 0))

        self.continuous_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main, text="连续多局(自动重载)", variable=self.continuous_var)\
            .grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.start_btn = ttk.Button(main, text="启动", command=self._start)
        self.start_btn.grid(row=3, column=2, sticky="e", pady=(6, 0), padx=(6, 0))
        self.stop_btn = ttk.Button(main, text="停止", command=self._request_stop, state="disabled")
        self.stop_btn.grid(row=3, column=3, sticky="e", pady=(6, 0), padx=(6, 0))

        self.status = ttk.Label(main, text="空闲")
        self.status.grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))

        self.log_text = tk.Text(main, height=18, width=120, state="disabled",
                                font=("Consolas", 9), wrap="none")
        scroll = ttk.Scrollbar(main, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.grid(row=5, column=0, columnspan=4, sticky="nsew", pady=(6, 0))
        scroll.grid(row=5, column=4, sticky="ns")

    def _reload_presets(self) -> None:
        self._saved_decks = load_saved_decks()
        names = [DECK_FILE_LABEL, *self._saved_decks]
        self.deck_box["values"] = names
        self.deck_box.set("2.6 hog" if "2.6 hog" in self._saved_decks else names[0])
        self._on_deck_change()
        models = discover_models()
        self.model_box["values"] = models
        self.model_box.set(DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else ""))

    def _on_deck_change(self, event=None) -> None:
        deck = self._saved_decks.get(self.deck_box.get())
        self.deck_info["text"] = " ".join(str(cid) for cid in deck) if deck else "使用校准目录 my_deck.json"

    # -- worker ------------------------------------------------------------

    def _start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        selection = self.deck_box.get()
        deck = self._saved_decks.get(selection)
        model = self.model_box.get()
        checkpoint = (FIRSTLIGHT_ROOT / "checkpoints" / model) if model else None
        self._stop.clear()
        self.start_btn["state"] = "disabled"
        self.stop_btn["state"] = "normal"
        self.status["text"] = "启动中…"
        continuous = self.continuous_var.get()  # read on the main thread: a
        # tk.BooleanVar is not safe to touch from the worker thread
        self._worker = threading.Thread(target=self._run, args=(deck, checkpoint, continuous), daemon=True)
        self._worker.start()

    def _run(self, deck: tuple[int, ...] | None, checkpoint: Path | None,
             continuous: bool) -> None:
        def log(msg) -> None:
            self._log_q.put(str(msg))

        try:
            run_live(deck=deck, checkpoint=checkpoint,
                     continuous=continuous,
                     should_continue=lambda: not self._stop.is_set(),
                     log=log)
        except SystemExit as error:
            log(f"已退出: {error}")
        except Exception as error:  # keep the GUI alive no matter what
            log(f"异常退出: {error!r}")
        finally:
            self._log_q.put("__DONE__")

    def _request_stop(self) -> None:
        self._stop.set()
        self.status["text"] = "停止中(等待会话取消)…"

    # -- log pump ------------------------------------------------------------

    def _poll_log(self) -> None:
        try:
            while True:
                msg = self._log_q.get_nowait()
                if msg == "__DONE__":
                    self.start_btn["state"] = "normal"
                    self.stop_btn["state"] = "disabled"
                    self.status["text"] = "空闲"
                    continue
                self.log_text["state"] = "normal"
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text["state"] = "disabled"
                if msg.startswith("summary:"):
                    self.status["text"] = msg[9:109]
                elif not msg.startswith(("cal probe", "play:", "[stream]")):
                    self.status["text"] = msg[:100]
        except queue.Empty:
            pass
        self.root.after(200, self._poll_log)


def main() -> int:
    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
