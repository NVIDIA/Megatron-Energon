import filetype
from bitstring.bits import BitsType

from .containers.matroska import parse_matroska
from .containers.mpeg import parse_mpeg
from .containers.probe import parse_probe
from .keyframeinfo import KeyframeInfo


class Fastseek:
    """

    Gathers information from the video container file (e.g. metadata which requires minimal decoding)
    to find keyframes in the video for fast seeking.

    Information is returned in the form of KeyframeInfo structures which can be used by a decoding loop
    to make informed decisions about the best seeking behavior

    Currently supports:
    - MP4/MOV: frames are indexed by number and frame counting can be used to get the exact frame
    - Matroska/WebM: frames are indexed by time and inter-frame duration must be accounted for to get to the right frame

    If your container is not list above, pass "probe=True" to the constructor, this will use ffmpeg to parse the stream
    without decoding it. Frames will be indexed by number. This is not as fast as using a supported container but is still
    significantly faster than sequential decoding.
    """

    def __init__(self, file: BitsType, probe: bool = False) -> None:
        if probe:
            self.keyframes = parse_probe(file)
            self.unit = "count"
        else:
            ftype = filetype.guess(file)

            if ftype is None:
                raise ValueError("Unable to determine file type")

            self.mime = ftype.mime

            if ftype.mime in ["video/mp4", "video/quicktime"]:
                self.keyframes = parse_mpeg(file)
                self.unit = "count"
            elif ftype.mime in ["video/x-matroska", "video/webm"]:
                self.keyframes = parse_matroska(file)
                self.unit = "time"
            else:
                raise ValueError(
                    f"Unsupported container: {ftype.mime} (hint: try passing probe=True to the Fastseek constructor)"
                )

    def should_seek(
        self, current: int, target: int, stream: int = 0
    ) -> KeyframeInfo | None:
        nearest_iframe: int = self.nearest_keyframe(target, stream)
        return nearest_iframe if current < nearest_iframe.index <= target else None

    def nearest_keyframe(self, target: int, stream: int = 0) -> KeyframeInfo:
        # HACK some videos don't report track ID correctly, so just use a list for now
        nearest_iframe_to_target_index: int = (
            list(self.keyframes.values())[stream].bisect_left(target) - 1
        )
        return list(self.keyframes.values())[stream][
            max(0, nearest_iframe_to_target_index)
        ]
