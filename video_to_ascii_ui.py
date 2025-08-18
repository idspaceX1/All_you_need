import argparse
import cv2
import numpy as np
import time
import sys
from typing import Union, Optional, Tuple, List

# ASCII Banner
BANNER = r"""
BLACKCAT
"""

# Help/README
README = """
📖 **README: Video to ASCII Converter**

This tool converts videos or live webcam feeds into ASCII art videos.

🔹 **Features:**
- Supports video files (MP4, AVI, etc.) and live webcam.
- Customizable ASCII characters, colors, and output settings.
- Progress tracking during conversion.

🔹 **Usage:**
1. Choose an option (video file or webcam) via CLI flags.
2. Provide the input source (file path or webcam ID).
3. Specify output settings (optional).
4. Wait for the conversion to complete.

🔹 **Example Commands:**
- For video file: `python video_to_ascii_ui.py -i input.mp4 -o output.mp4`
- For webcam: `python video_to_ascii_ui.py --webcam 0 --duration 10 --output webcam.mp4`
"""


def parse_color(color_csv: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in color_csv.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Color must be in 'R,G,B' format")
    try:
        r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color components must be integers") from exc
    for v in (r, g, b):
        if v < 0 or v > 255:
            raise argparse.ArgumentTypeError("Color components must be between 0 and 255")
    return (r, g, b)


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
        is_webcam: bool = False,
    ) -> None:
        self.video_capture = cv2.VideoCapture(video_source)
        if not self.video_capture.isOpened():
            raise ValueError(f"❌ Unable to open video source: {video_source}")

        self.ascii_chars = ascii_chars
        self.width = width
        self.fps = fps
        self.output_file = output_file
        self.font_scale = font_scale
        self.font_color = font_color
        self.background_color = background_color
        self.is_webcam = is_webcam

        # Read initial frame to compute output height
        ret, frame = self.video_capture.read()
        if not ret:
            raise ValueError("❌ Unable to read from video source")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = gray.shape
        aspect_ratio = frame_h / frame_w if frame_w > 0 else 1.0
        # ASCII characters tend to be taller than wide; scale height accordingly
        self.height = max(1, int(self.width * aspect_ratio * 0.55))

        # Estimate character cell size (pixels) for drawing
        self.char_w, self.char_h = 10, 18
        self.out_w = self.width * self.char_w
        self.out_h = self.height * self.char_h
        # Some encoders require even dimensions
        if self.out_w % 2 != 0:
            self.out_w += 1
        if self.out_h % 2 != 0:
            self.out_h += 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.output_file, fourcc, self.fps, (self.out_w, self.out_h)
        )
        if not self.video_writer.isOpened():
            raise RuntimeError("❌ Unable to open video writer. Try a different codec or filename.")

        # Reset capture to beginning for file input
        if not self.is_webcam:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _pixel_to_ascii(self, pixel_value: int) -> str:
        index = int((pixel_value / 255) * (len(self.ascii_chars) - 1))
        return self.ascii_chars[index]

    def _frame_to_ascii(self, frame_gray: np.ndarray) -> List[str]:
        resized = cv2.resize(frame_gray, (self.width, self.height))
        return ["".join(self._pixel_to_ascii(p) for p in row) for row in resized]

    def _ascii_to_image(self, ascii_frame: List[str]) -> np.ndarray:
        img = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
        img[:] = self.background_color
        y0 = 15
        for i, line in enumerate(ascii_frame):
            y = y0 + i * self.char_h
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
        return img

    def convert_file(self) -> None:
        start_time = time.time()
        frame_num = 0
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ascii_frame = self._frame_to_ascii(gray)
            img = self._ascii_to_image(ascii_frame)
            self.video_writer.write(img)

            frame_num += 1
            if total_frames > 0:
                sys.stdout.write(f"\r🔄 Converting: {frame_num}/{total_frames} frames")
            else:
                sys.stdout.write(f"\r🔄 Converting: {frame_num} frames")
            sys.stdout.flush()

        self._cleanup()
        print(f"\n✅ ASCII video saved to: {self.output_file}")
        print(f"⏱️  Time taken: {time.time() - start_time:.2f}s")

    def convert_webcam(self, duration_seconds: Optional[float] = None, preview: bool = False) -> None:
        start_time = time.time()
        frame_num = 0

        try:
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ascii_frame = self._frame_to_ascii(gray)
                img = self._ascii_to_image(ascii_frame)
                self.video_writer.write(img)
                frame_num += 1

                if preview:
                    try:
                        cv2.imshow("ASCII Webcam (press 'q' to quit)", img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except cv2.error:
                        # Likely running headless; disable preview gracefully
                        preview = False

                if duration_seconds is not None and (time.time() - start_time) >= duration_seconds:
                    break

                if frame_num % 10 == 0:
                    elapsed = time.time() - start_time
                    sys.stdout.write(f"\r📸 Recording webcam: {frame_num} frames, {elapsed:.1f}s")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            pass

        self._cleanup()
        print(f"\n✅ ASCII video saved to: {self.output_file}")
        print(f"⏱️  Time taken: {time.time() - start_time:.2f}s, frames: {frame_num}")

    def _cleanup(self) -> None:
        try:
            self.video_capture.release()
        finally:
            try:
                self.video_writer.release()
            finally:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert video files or webcam feed to ASCII art video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to input video file",
    )
    source_group.add_argument(
        "--webcam",
        type=int,
        metavar="CAM_ID",
        help="Use webcam with given camera index (e.g., 0)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="ascii_output.mp4",
        help="Output video filename",
    )
    parser.add_argument("--width", type=int, default=120, help="ASCII width in characters")
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second")
    parser.add_argument("--ascii-chars", type=str, default=" .:-=+*#%@", help="Characters used for ASCII mapping, from light to dark")
    parser.add_argument("--font-scale", type=float, default=0.4, help="Font scale for drawing ASCII")
    parser.add_argument("--font-color", type=parse_color, default=(255, 255, 255), help="Font color as R,G,B")
    parser.add_argument("--bg-color", type=parse_color, default=(0, 0, 0), help="Background color as R,G,B")
    parser.add_argument("--preview", action="store_true", help="Preview output in a window (requires display)")
    parser.add_argument("--duration", type=float, default=None, help="For webcam mode, record for N seconds and stop")
    parser.add_argument("--readme", action="store_true", help="Show README and exit")
    return parser


def main() -> None:
    print(BANNER)
    print("=" * 50)
    print("🎥 Video to ASCII Converter")
    print("=" * 50)

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.readme:
        print(README)
        return

    is_webcam = args.webcam is not None
    video_source: Union[str, int] = args.input if not is_webcam else args.webcam

    # Adjust defaults for webcam if user did not override
    output_file = args.output
    if is_webcam and output_file == "ascii_output.mp4":
        output_file = "webcam_ascii.mp4"

    fps = args.fps
    if is_webcam and args.fps == 30:
        # Use a gentler default for webcams unless explicitly set
        fps = 15

    try:
        converter = VideoToAscii(
            video_source=video_source,
            output_file=output_file,
            width=args.width,
            fps=fps,
            ascii_chars=args["ascii_chars"] if isinstance(args, dict) else args.ascii_chars,
            font_scale=args.font_scale,
            font_color=args.font_color,
            background_color=args.bg_color,
            is_webcam=is_webcam,
        )

        if is_webcam:
            converter.convert_webcam(duration_seconds=args.duration, preview=args.preview)
        else:
            converter.convert_file()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

