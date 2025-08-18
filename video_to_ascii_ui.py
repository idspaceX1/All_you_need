import cv2
import numpy as np
import time
import sys
from typing import Union, Optional

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
1. Choose an option (video file or webcam).
2. Provide the input source (file path or webcam ID).
3. Specify output settings (optional).
4. Wait for the conversion to complete.

🔹 **Example Commands:**
- For video file: `python video_to_ascii_ui.py`
- For webcam: `python video_to_ascii_ui.py --webcam`
"""

class VideoToAscii:
    def __init__(
        self,
        video_source: Union[str, int],
        output_file: str = "ascii_output.mp4",
        width: int = 120,
        fps: int = 30,
        ascii_chars: str = " .:-=+*#%@",
        font_scale: float = 0.4,
        font_color: tuple = (255, 255, 255),
        background_color: tuple = (0, 0, 0),
    ):
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

        # Calculate ASCII height
        ret, frame = self.video_capture.read()
        if not ret:
            raise ValueError("❌ Unable to read video")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        aspect_ratio = h / w
        self.height = int(self.width * aspect_ratio * 0.55)

        # Initialize video writer
        self.char_w, self.char_h = 10, 18
        self.out_w = self.width * self.char_w
        self.out_h = self.height * self.char_h
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.output_file, fourcc, fps, (self.out_w, self.out_h)
        )

        # Reset capture to beginning
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _pixel_to_ascii(self, pixel_value: int) -> str:
        index = int((pixel_value / 255) * (len(self.ascii_chars) - 1))
        return self.ascii_chars[index]

    def _frame_to_ascii(self, frame: np.ndarray) -> list:
        resized = cv2.resize(frame, (self.width, self.height))
        return ["".join(self._pixel_to_ascii(p) for p in row) for row in resized]

    def convert(self) -> None:
        start_time = time.time()
        frame_num = 0
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ascii_frame = self._frame_to_ascii(gray)

            # Draw ASCII on canvas
            img = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
            img[:] = self.background_color
            y0 = 15
            for i, line in enumerate(ascii_frame):
                y = y0 + i * self.char_h
                cv2.putText(
                    img, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale, self.font_color, 1, cv2.LINE_AA
                )

            self.video_writer.write(img)
            frame_num += 1
            sys.stdout.write(f"\r🔄 Converting: {frame_num}/{total_frames} frames")
            sys.stdout.flush()

        self.video_capture.release()
        self.video_writer.release()
        print(f"\n✅ ASCII video saved to: {self.output_file}")
        print(f"⏱️  Time taken: {time.time() - start_time:.2f}s")

def main():
    print(BANNER)
    print("=" * 50)
    print("🎥 Video to ASCII Converter")
    print("=" * 50)

    # Show README if requested
    if "--help" in sys.argv or "-h" in sys.argv:
        print(README)
        return

    # Choose mode
    print("\n🔘 Select Mode:")
    print("1. Convert a video file")
    print("2. Use live webcam")
    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        video_source = input("📁 Enter video file path: ").strip()
        output_file = input("💾 Output file name (e.g., output.mp4): ").strip() or "ascii_output.mp4"
        width = int(input("📏 ASCII width (default: 120): ") or 120)
        fps = int(input("🎞️  Output FPS (default: 30): ") or 30)
    elif choice == "2":
        video_source = 0
        output_file = "webcam_ascii.mp4"
        width = 80
        fps = 15
        print("📸 Using webcam (press 'q' to stop).")
    else:
        print("❌ Invalid choice. Exiting.")
        return

    # Run converter
    try:
        converter = VideoToAscii(
            video_source=video_source,
            output_file=output_file,
            width=width,
            fps=fps,
        )
        converter.convert()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()