"""Required to make multiprocessing find the source directory by setting
   sys.path if necessary."""

import os
import sys
from pathlib import Path

_SRC_PATH = str(Path(__file__).resolve().parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

# Disable PySDL3 version check
os.environ["SDL_CHECK_VERSION"] = '0'


def _patch_imgui_key_aliases() -> None:
    """Map imgui-bundle 1.5.x Key names to the newer mod_* aliases."""
    try:
        from imgui_bundle import imgui
    except ImportError:
        return
    aliases = {
        "mod_ctrl": "im_gui_mod_ctrl",
        "mod_shift": "im_gui_mod_shift",
        "mod_alt": "im_gui_mod_alt",
        "mod_none": "im_gui_mod_none",
        "mod_super": "im_gui_mod_super",
    }
    for new_name, old_name in aliases.items():
        if not hasattr(imgui.Key, new_name) and hasattr(imgui.Key, old_name):
            setattr(imgui.Key, new_name, getattr(imgui.Key, old_name))


_patch_imgui_key_aliases()

# Don't expose temporary module imports
__all__ = []
