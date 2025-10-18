import os
import cv2
import imageio
import numpy as np
import html
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Basic ASCII HTML renderer
# -----------------------------
def basic_ascii_html(
    gif_path,
    output_html="ascii_animation.html",
    font_size_y=8,
    framerate=12,
    alphanumerics=False,
    color=True,
    specific_color=None,
    block_height=8,
    block_width=4
):
    if not os.path.exists(gif_path):
        raise FileNotFoundError(f"GIF not found: {gif_path}")

    print(f"Loading GIF: {gif_path}")
    im = imageio.mimread(gif_path)
    print(f"Loaded {len(im)} frames.")

    asciiChars = " .'`^\",:;Il!i~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    if alphanumerics:
        asciiChars += "?#$%&@0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()[]{}<>"

    frames_html = []
    for idx, frame in enumerate(im):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

        h, w = gray.shape
        frame_html = "<pre class='ascii-frame'>\n"

        for y in range(0, h, block_height):
            line = ""
            for x in range(0, w, block_width):
                avg = np.mean(gray[y:y+block_height, x:x+block_width])
                idx_char = int((avg / 255) * (len(asciiChars) - 1))
                char = asciiChars[idx_char]

                if color:
                    region = frame_rgb[y:y+block_height, x:x+block_width]
                    avg_color = np.mean(region.reshape(-1,3), axis=0)
                    r,g,b = map(int, avg_color)
                    fill_color = f"rgb({r},{g},{b})"
                elif specific_color:
                    fill_color = specific_color
                else:
                    fill_color = "white"

                if color or specific_color:
                    line += f'<span style="color:{fill_color}">{html.escape(char)}</span>'
                else:
                    line += html.escape(char)

            frame_html += line + "\n"

        frame_html += "</pre>\n"
        frames_html.append(frame_html)
        print(f"Frame {idx+1}/{len(im)} processed")

    # Construct HTML
    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ASCII GIF Animation</title>
<style>
body {{
    background: black;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
}}
pre.ascii-frame {{
    font-family: monospace;
    font-size: {font_size_y}px;
    line-height: 0.8;
    white-space: pre;
    user-select: text;
}}
</style>
</head>
<body>
<div id="ascii-container"></div>
<script>
const frames = {frames_html};
const container = document.getElementById('ascii-container');
let current = 0;
setInterval(() => {{
    container.innerHTML = frames[current];
    current = (current + 1) % frames.length;
}}, 1000 / {framerate});
</script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_code)
    print(f"✅ ASCII animation saved to {output_html}")


# -----------------------------
# Laplacian ASCII HTML renderer (Creates outline effect)
# -----------------------------
def laplacian_ascii_html(
    gif_path,
    output_html="ascii_laplacian.html",
    font_size_y=12,
    framerate=12,
    alphanumerics=False,
    density=25,
    color=True,
    specific_color=None
):
    if not os.path.exists(gif_path):
        raise FileNotFoundError(f"GIF not found: {gif_path}")

    print(f"Loading GIF: {gif_path}")
    im = imageio.mimread(gif_path)
    print(f"Loaded {len(im)} frames.")

    asciiChars = " !\"'*+,-./:;=\^_`|~"
    if alphanumerics:
        asciiChars += "?#$%&@0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()[]{}<>"

    font_size_x = int(font_size_y * 0.45)
    consolas = ImageFont.truetype("./Consolas.ttf", int(font_size_y * 0.8))

    # Create character atlas and Laplacians
    asciiAtlas = np.zeros((len(asciiChars), font_size_y, font_size_x), dtype=np.uint8)
    laplacianAtlas = np.zeros(asciiAtlas.shape, dtype=np.uint8)
    for i, char in enumerate(asciiChars):
        im_p = Image.fromarray(asciiAtlas[i])
        draw = ImageDraw.Draw(im_p)
        draw.text((0, 0), char, (255), font=consolas)
        asciiAtlas[i] = np.array(im_p)
        laplacianAtlas[i] = cv2.Laplacian(asciiAtlas[i], cv2.CV_8U)
        cv2.normalize(laplacianAtlas[i].astype(np.float32), laplacianAtlas[i], 0, 255, cv2.NORM_MINMAX)
        laplacianAtlas[i] = laplacianAtlas[i].astype(np.uint8)

    frames_html = []

    for idx, frame in enumerate(im):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.Laplacian(gray, cv2.CV_8U)
        gray = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX) * (density / 255)

        h, w = gray.shape
        frame_html = "<pre class='ascii-frame'>\n"

        for y in range(0, h, font_size_y):
            line = ""
            for x in range(0, w, font_size_x):
                img_slice = gray[y:y+font_size_y, x:x+font_size_x]
                similarities = np.sum(np.abs((laplacianAtlas/255) - img_slice), axis=(1,2))
                min_index = np.argmin(similarities)
                char = asciiChars[min_index]

                if color:
                    region = frame_rgb[y:y+font_size_y, x:x+font_size_x]
                    avg_color = np.mean(region.reshape(-1,3), axis=0)
                    r,g,b = map(int, avg_color)
                    fill_color = f"rgb({r},{g},{b})"
                elif specific_color:
                    fill_color = specific_color
                else:
                    fill_color = "white"

                line += f'<span style="color:{fill_color}">{html.escape(char)}</span>'
            frame_html += line + "\n"
        frame_html += "</pre>\n"
        frames_html.append(frame_html)
        print(f"Frame {idx+1}/{len(im)} processed")

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Laplacian ASCII GIF</title>
<style>
body {{
    background: black;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
}}
pre.ascii-frame {{
    font-family: monospace;
    font-size: {font_size_y}px;
    line-height: 0.8;
    white-space: pre;
    user-select: text;
}}
</style>
</head>
<body>
<div id="ascii-container"></div>
<script>
const frames = {frames_html};
const container = document.getElementById('ascii-container');
let current = 0;
setInterval(() => {{
    container.innerHTML = frames[current];
    current = (current + 1) % frames.length;
}}, 1000 / {framerate});
</script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_code)
    print(f"✅ Laplacian ASCII animation saved to {output_html}")


# -----------------------------
# Main CLI
# -----------------------------
if __name__ == "__main__":
    # --------------------------
    # CONFIGURATION
    # --------------------------
    gif_file = "bisonrun.gif"        # local GIF path
    output_html = "ascii_animation.html"

    # Choose render mode:
    # 1 = full ASCII render
    # 2 = Laplacian edge-based render
    render_mode = 1

    # Common parameters
    font_size_y = 8
    framerate = 12
    color = True
    specific_color = "lime"           # only used if color=False
    block_height = 8
    block_width = 4
    alphanumerics = False
    density = 25                      # only relevant for Laplacian render

    # --------------------------
    # RUN THE CHOSEN RENDER
    # --------------------------
    if render_mode == 1:
        # Standard ASCII rendering
        basic_ascii_html(
            gif_path=gif_file,
            output_html=output_html,
            font_size_y=font_size_y,
            framerate=framerate,
            alphanumerics=alphanumerics,
            color=color,
            specific_color=specific_color,
            block_height=block_height,
            block_width=block_width
        )
    elif render_mode == 2:
        # Laplacian edge-based rendering
        laplacian_ascii_html(
            gif_path=gif_file,
            output_html=output_html,
            font_size_y=font_size_y,
            framerate=framerate,
            alphanumerics=alphanumerics,
            density=density,
            color=color,
            specific_color=specific_color,
            block_height=block_height,
            block_width=block_width
        )
    else:
        raise ValueError("Invalid render_mode. Use 1 for ASCII or 2 for Laplacian.")

    print(f"✅ Done! ASCII animation saved to {output_html}")
