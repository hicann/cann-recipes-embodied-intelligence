from dataclasses import dataclass
from typing import Tuple
from enum import Enum, auto


class FrontFace(Enum):
    CCW = auto()
    CW = auto()


@dataclass
class RenderSettings:
    image_size: Tuple[int, int] = (256, 256)
    max_blend_depth: int = 8
    bin_size: int = 64
    blur_radius_ndc: float = 0.0
    clip_barycentric_coords: bool = False
    cull_backfaces: bool = False
    front_face: FrontFace = FrontFace.CCW
    persp_correct: bool = False

    def front_face_direction(self) -> int:
        return 1 if self.front_face == FrontFace.CCW else -1
    
    def ndc_to_screen_scale(self) -> float:
        return min(self.image_size[0], self.image_size[1] * 0.5)
    
    def screen_to_ndc_scale(self) -> float:
        return 1 / self.ndc_to_screen_scale()
    
    def blur_radius_scr(self) -> float:
        return self.blur_radius_ndc * self.ndc_to_screen_scale()