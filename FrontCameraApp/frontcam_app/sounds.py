"""Short cues for capturing from 2 m away, where the screen text is hard to read."""

import sys
import threading


def _beep(frequency, milliseconds):
    if sys.platform == "win32":
        import winsound
        try:
            winsound.Beep(frequency, milliseconds)
        except RuntimeError:
            winsound.MessageBeep()
    else:
        sys.stdout.write("\a")
        sys.stdout.flush()


def play(kind):
    """kind: "capture", "error" or "done". Never blocks the caller (winsound.Beep does)."""
    frequency, milliseconds = {"capture": (1200, 90), "error": (300, 220), "done": (1600, 250)}[kind]
    threading.Thread(target=_beep, args=(frequency, milliseconds), daemon=True).start()
