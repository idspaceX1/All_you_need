import argparse
import sys
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np


BANNER = r"""
BLACKCAT
"""

README = """
README: Video to ASCII Converter

This tool converts videos or live webcam feeds into ASCII art videos.

Features:
- Supports video files (MP4, AVI, etc.) and live webcam.
- Customizable ASCII characters, colors, and output settings.
- Progress tracking during conversion.

Examples:
- Convert a video file: python video_to_ascii_ui.py --source input.mp4 --output ascii.mp4
- Use webcam:          python video_to_ascii_ui.py --webcam --output webcam_ascii.mp4 --display
"""


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """Parse a color string in the form "R,G,B" or "#RRGGBB" into BGR tuple for OpenCV."""
    value = color_str.strip()
    if value.startswith("#"):
        hex_value = value[1:]
        if len(hex_value) != 6:
            raise argparse.ArgumentTypeError("Hex color must be in the form #RRGGBB")
        r = int(hex_value[0:2], 16)
        g = int(hex_value[2:4], 16)
        b = int(hex_value[4:6], 16)
        return (b, g, r)
    if "," in value:
        parts = value.split(",")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("RGB color must have exactly 3 comma-separated numbers")
        r, g, b = [int(p) for p in parts]
        for c in (r, g, b):
            if not (0 <= c <= 255):
                raise argparse.ArgumentTypeError("RGB color components must be between 0 and 255")
        return (b, g, r)
    raise argparse.ArgumentTypeError("Color must be 'R,G,B' or '#RRGGBB'")


class VideoToAscii:
    def __init__(
        self,
        video_source: Union[str, int],
        output_file: str = "ascii_output.mp4",
        width: int = 120,
        fps: int = 30,
        ascii_chars: str = " .:-=+*#%@",
        font_scale: float = 0.4,
        font_color: Tuple[int, int, int] = (255, 255, 255),
        background_color: Tuple[int, int, int] = (0, 0, 0),
        max_frames: Optional[int] = None,
        display: bool = False,
    ) -> None:
        self.video_capture = cv2.VideoCapture(video_source)
        if not self.video_capture.isOpened():
            raise ValueError(f"Unable to open video source: {video_source}")

        self.ascii_chars = ascii_chars
        self.width = width
        self.fps = fps
        self.output_file = output_file
        self.font_scale = float(font_scale)
        self.font_color = font_color
        self.background_color = background_color
        self.max_frames = max_frames
        self.display = display

        # Read a probe frame to determine output size
        ret, frame = self.video_capture.read()
        if not ret:
            raise ValueError("Unable to read from video source")
        gray_probe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        probe_h, probe_w = gray_probe.shape
        aspect_ratio = probe_h / probe_w if probe_w != 0 else 1.0
        self.height = max(1, int(self.width * aspect_ratio * 0.55))

        # Determine approximate character cell size based on current font settings
        (char_w, char_h), _ = cv2.getTextSize(
            text="M",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=self.font_scale,
            thickness=1,
        )
        self.char_w = max(1, char_w)
        self.char_h = max(1, char_h + 4)  # add small line spacing for readability

        self.out_w = int(self.width * self.char_w + 10)
        self.out_h = int(self.height * self.char_h + 10)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.output_file, fourcc, self.fps, (self.out_w, self.out_h))

        # Reset capture to the beginning when applicable (has no effect on webcams)
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _pixel_to_ascii(self, pixel_value: int) -> str:
        index = int((pixel_value / 255.0) * (len(self.ascii_chars) - 1))
        return self.ascii_chars[index]

    def _frame_to_ascii(self, gray_frame: np.ndarray) -> list[str]:
        resized = cv2.resize(gray_frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # Build ASCII lines
        lines: list[str] = []
        for row in resized:
            line_chars = (self._pixel_to_ascii(int(p)) for p in row)
            lines.append("".join(line_chars))
        return lines

    def convert(self) -> None:
        start_time = time.time()
        frame_count_written = 0
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        is_webcam_like = total_frames <= 0

        if self.display:
            cv2.namedWindow("ASCII", cv2.WINDOW_NORMAL)

        try:
            while True:
                if self.max_frames is not None and frame_count_written >= self.max_frames:
                    break

                ret, frame = self.video_capture.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ascii_frame = self._frame_to_ascii(gray)

                # Draw ASCII on a canvas
                img = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
                img[:] = self.background_color

                baseline_y = self.char_h  # top padding already included
                for i, line in enumerate(ascii_frame):
                    y = baseline_y + i * self.char_h
                    cv2.putText(
                        img,
                        line,
                        (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        self.font_color,
                        1,
                        cv2.LINE_AA,
                    )

                self.video_writer.write(img)
                frame_count_written += 1

                if self.display:
                    cv2.imshow("ASCII", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if is_webcam_like:
                    sys.stdout.write(f"\rConverting (webcam/live): {frame_count_written} frames")
                else:
                    sys.stdout.write(
                        f"\rConverting: {frame_count_written}/{total_frames} frames"
                    )
                sys.stdout.flush()
        finally:
            self.video_capture.release()
            self.video_writer.release()
            if self.display:
                cv2.destroyAllWindows()

        duration_s = time.time() - start_time
        print(f"\nASCII video saved to: {self.output_file}")
        print(f"Time taken: {duration_s:.2f}s")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a video file or webcam stream to an ASCII art video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )
    parser.add_argument("--webcam", action="store_true", help="Use the default webcam as input")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to the input video file. Ignored if --webcam is used.",
    )
    parser.add_argument("--output", type=str, default="ascii_output.mp4", help="Output video file path")
    parser.add_argument("--width", type=int, default=120, help="ASCII character width")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--font-scale", type=float, default=0.4, help="Font scale for ASCII characters")
    parser.add_argument(
        "--font-color",
        type=parse_color,
        default="255,255,255",
        help="Font color as 'R,G,B' or '#RRGGBB' (interpreted as RGB)",
    )
    parser.add_argument(
        "--bg-color",
        type=parse_color,
        default="0,0,0",
        help="Background color as 'R,G,B' or '#RRGGBB' (interpreted as RGB)",
    )
    parser.add_argument(
        "--chars",
        type=str,
        default=" .:-=+*#%@",
        help="ASCII characters from lightest to darkest",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (useful for webcam)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display a live preview window (press 'q' to stop)",
    )
    parser.add_argument(
        "--readme",
        action="store_true",
        help="Show README and exit",
    )
    return parser


def interactive_prompt() -> tuple[Union[str, int], str, int, int]:
    print(BANNER)
    print("=" * 50)
    print("Video to ASCII Converter")
    print("=" * 50)

    print("\nSelect Mode:")
    print("1. Convert a video file")
    print("2. Use live webcam")
    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        video_source = input("Enter video file path: ").strip()
        output_file = input("Output file name (e.g., output.mp4): ").strip() or "ascii_output.mp4"
        width = int(input("ASCII width (default: 120): ") or 120)
        fps = int(input("Output FPS (default: 30): ") or 30)
        return video_source, output_file, width, fps
    elif choice == "2":
        print("Using webcam (press 'q' to stop).")
        return 0, "webcam_ascii.mp4", 80, 15
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.readme:
        print(README)
        return

    # If no CLI inputs, fall back to interactive prompts
    use_interactive = not (args.webcam or args.source)

    if use_interactive:
        video_source, output_file, width, fps = interactive_prompt()
        ascii_chars = " .:-=+*#%@"
        font_scale = 0.4
        font_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        display = True if video_source == 0 else False
        max_frames = None
    else:
        video_source = 0 if args.webcam else args.source
        if video_source is None:
            parser.error("--source is required unless --webcam is provided")
        output_file = args.output
        width = args.width
        fps = args.fps if not args.webcam else max(1, args.fps)
        ascii_chars = args.chars
        font_scale = args.font_scale
        font_color = args.font_color
        bg_color = args.bg_color
        display = args.display
        max_frames = args.max_frames

    try:
        converter = VideoToAscii(
            video_source=video_source,
            output_file=output_file,
            width=width,
            fps=fps,
            ascii_chars=ascii_chars,
            font_scale=font_scale,
            font_color=font_color,
            background_color=bg_color,
            max_frames=max_frames,
            display=display,
        )
        converter.convert()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

