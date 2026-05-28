"""Components/ConfigSections/DebugInfoSection.py: Shows debug info."""

from dataclasses import dataclass

from imgui_bundle import imgui, icons_fontawesome_6 as fa

from ICGui.State.Volatile import CameraState
from .Section import Section


@dataclass
class DebugInfoSection(Section):
    name: str = f'{fa.ICON_FA_CODE} Debug Info'
    always_open: bool = False
    default_open: bool = False

    def _render(self):
        camera_state = CameraState()
        current_view = camera_state.current_view

        imgui.text('Current View:')
        if current_view is None:
            imgui.text('  None')
        else:
            imgui.text(f'  Frame Index: {current_view.frame_idx}')
            imgui.text(f'  Global Frame Index: {current_view.global_frame_idx}')
            imgui.text(f'  Camera Index: {current_view.camera_index}')
            if current_view.exif is not None and len(current_view.exif) > 0:
                imgui.text('  Exif: {')
                for key, value in current_view.exif.items():
                    imgui.text(f'    {key}: {value}')
                imgui.text('  }')
