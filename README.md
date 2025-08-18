# Video to ASCII Converter 🎥➡️📺

A powerful Python tool that converts videos and live webcam feeds into ASCII art videos with customizable settings.

## Features ✨

- 🎬 **Video File Support**: Convert MP4, AVI, and other common video formats
- 📹 **Live Webcam**: Real-time ASCII conversion from webcam feed
- 🎨 **Customizable ASCII Characters**: Choose your own character set for different visual styles
- 🌈 **Color Options**: Customize font and background colors
- 📏 **Adjustable Resolution**: Control ASCII art width and output quality
- ⚡ **Progress Tracking**: Real-time conversion progress display
- 🎯 **Easy to Use**: Interactive command-line interface

## Installation 🔧

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage 🚀

### Basic Usage
```bash
python video_to_ascii_ui.py
```

### Help
```bash
python video_to_ascii_ui.py --help
```

### Interactive Mode
The script will guide you through:
1. **Mode Selection**: Choose between video file or webcam
2. **Input Source**: Specify video file path or use default webcam
3. **Output Settings**: Configure output filename, width, and FPS
4. **Conversion**: Watch the progress as your video is converted

### Example Workflow

1. **Video File Conversion**:
   - Choose option 1
   - Enter your video file path (e.g., `my_video.mp4`)
   - Set output filename (e.g., `ascii_video.mp4`)
   - Configure width (default: 120 characters)
   - Set FPS (default: 30)

2. **Webcam Conversion**:
   - Choose option 2
   - The script automatically uses your default webcam
   - Output saved as `webcam_ascii.mp4`

## Customization Options 🎛️

The `VideoToAscii` class supports various parameters:

- `video_source`: Video file path or webcam ID (0 for default)
- `output_file`: Output video filename
- `width`: ASCII art width in characters (default: 120)
- `fps`: Output video frame rate (default: 30)
- `ascii_chars`: Character set for ASCII conversion (default: " .:-=+*#%@")
- `font_scale`: Text size scaling (default: 0.4)
- `font_color`: RGB color tuple for text (default: white)
- `background_color`: RGB color tuple for background (default: black)

## Requirements 📋

- Python 3.6+
- OpenCV (cv2)
- NumPy

## How It Works 🔍

1. **Frame Extraction**: Reads video frames using OpenCV
2. **Grayscale Conversion**: Converts color frames to grayscale
3. **Resize & Map**: Resizes frames and maps pixel intensities to ASCII characters
4. **Text Rendering**: Renders ASCII text onto video frames
5. **Video Output**: Saves the result as an MP4 video file

## Tips 💡

- **Performance**: Lower width values process faster but reduce detail
- **Quality**: Higher width values provide more detail but take longer to process
- **ASCII Characters**: Experiment with different character sets for unique visual styles
- **File Formats**: Most common video formats are supported (MP4, AVI, MOV, etc.)

## Troubleshooting 🔧

- **"Unable to open video source"**: Check file path and format
- **Webcam not working**: Ensure camera permissions and try different camera IDs
- **Slow processing**: Reduce width or FPS for faster conversion
- **Large output files**: Lower FPS or use video compression tools afterward

## Examples 📸

### Custom ASCII Characters
```python
# Minimal characters for clean look
ascii_chars = " .-+*@"

# Dense characters for detailed output
ascii_chars = " .':;coxkXN"
```

### Different Styles
- **Minimalist**: `" .:-=+"`
- **Detailed**: `" .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"`
- **Blocks**: `" ░▒▓█"`

## License 📄

This project is open source and available under the MIT License.

## Contributing 🤝

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

---

**Enjoy creating ASCII art videos! 🎨**