#!/usr/bin/env python3
"""
REIMAGINED MATERIALS SCIENCE EDA ICONS
Dark, Crystallographic, Quantum-inspired Aesthetic
Publication-grade | 1024×1024 RGBA
"""
import os, math, random
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SIZE = 1024
OUT = "icons_new"
os.makedirs(OUT, exist_ok=True)

FONT_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_M = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"

# ── Colour palette (dark science) ──────────────────────────────────────
C = {
    'bg':        (8,   12,  22,  255),   # near-black navy
    'grid':      (30,  45,  70,  120),   # dim lattice lines
    'accent1':   (0,   200, 255, 255),   # electric cyan
    'accent2':   (255, 120, 0,   255),   # hot amber
    'accent3':   (120, 255, 140, 255),   # cool green
    'accent4':   (200, 80,  255, 255),   # plasma violet
    'hot':       (255, 60,  30,  255),   # furnace red-orange
    'cold':      (50,  100, 240, 255),   # quench blue
    'gold':      (255, 200, 60,  255),   # diffraction gold
    'white':     (240, 248, 255, 255),
    'dim':       (100, 130, 160, 200),
    'subtle':    (60,  80,  110, 160),
}

def fimg():
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)

def font(size, bold=True):
    try: return ImageFont.truetype(FONT_B if bold else FONT_R, size)
    except: return ImageFont.load_default()

def mfont(size):
    try: return ImageFont.truetype(FONT_M, size)
    except: return ImageFont.load_default()

def bg_panel(draw, x, y, w, h, alpha=220):
    """Draw a dark panel with subtle border"""
    r = 18
    draw.rounded_rectangle([x, y, x+w, y+h], radius=r,
                            fill=(8, 14, 30, alpha),
                            outline=(40, 70, 110, 180), width=2)

def glow_circle(img, cx, cy, r, color, layers=5):
    """Soft glow blob"""
    d = ImageDraw.Draw(img)
    for i in range(layers, 0, -1):
        alpha = int(color[3] * (i / layers) * 0.4) if len(color) > 3 else int(80 * i / layers)
        col = color[:3] + (alpha,)
        rr = r + (layers - i) * r // 2
        d.ellipse([cx-rr, cy-rr, cx+rr, cy+rr], fill=col)

def draw_hex_grid(draw, cx, cy, radius, spacing=60, color=None):
    """Isometric hex lattice background"""
    if color is None: color = C['grid']
    dx = spacing
    dy = spacing * 0.866
    for row in range(-8, 9):
        for col in range(-8, 9):
            x = cx + col * dx + (row % 2) * dx / 2
            y = cy + row * dy
            dist = math.hypot(x - cx, y - cy)
            if dist > radius * 1.1: continue
            alpha = int(color[3] * max(0, 1 - dist / (radius * 1.05)))
            c = color[:3] + (alpha,)
            draw.ellipse([x-2, y-2, x+2, y+2], fill=c)

def draw_axis(draw, x0, y0, x1, y1, color, width=3):
    draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
    # arrowhead
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    ux, uy = dx/L, dy/L
    px, py = -uy, ux
    sz = 14
    draw.polygon([
        (x1, y1),
        (int(x1 - sz*ux + sz*0.45*px), int(y1 - sz*uy + sz*0.45*py)),
        (int(x1 - sz*ux - sz*0.45*px), int(y1 - sz*uy - sz*0.45*py)),
    ], fill=color)

def centered_text(draw, cx, cy, text, f, color):
    bb = draw.textbbox((0, 0), text, font=f)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    draw.text((cx - tw//2, cy - th//2), text, font=f, fill=color)

def label(draw, cx, cy, text, size=28, bold=True, color=None):
    if color is None: color = C['white']
    f = font(size, bold)
    bb = draw.textbbox((0,0), text, font=f)
    tw = bb[2]-bb[0]
    draw.text((cx - tw//2, cy), text, font=f, fill=color)

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD FILE  →  Powder Sintering reimagined as particle-flow ingestion
# ──────────────────────────────────────────────────────────────────────────────
def icon_loadfile():
    img, draw = fimg()
    cx, cy = SIZE//2, SIZE//2

    # Background
    bg_panel(draw, 30, 30, SIZE-60, SIZE-60)
    draw_hex_grid(draw, cx, cy, 460)

    # Three stages connected by flow pipes
    stages = [
        (cx - 290, cy, C['accent1'], "RAW\nDATA"),
        (cx,       cy, C['accent2'], "PARSE"),
        (cx + 290, cy, C['accent3'], "STORE"),
    ]

    # Draw connecting conduits first
    for i in range(len(stages)-1):
        x1, y1, c1, _ = stages[i]
        x2, y2, c2, _ = stages[i+1]
        # gradient pipe
        steps = 40
        for s in range(steps):
            t = s / steps
            x = int(x1 + (x2 - x1) * t)
            r1, g1, b1 = c1[:3]
            r2, g2, b2 = c2[:3]
            rc = int(r1 + (r2-r1)*t)
            gc = int(g1 + (g2-g1)*t)
            bc = int(b1 + (b2-b1)*t)
            draw.ellipse([x-8, cy-8, x+8, cy+8], fill=(rc, gc, bc, 180))
        # flow particles
        for p in range(6):
            t = (p+0.5)/6
            x = int(x1 + 70 + (x2-x1-140) * t)
            draw.ellipse([x-5, cy-5, x+5, cy+5], fill=C['white'])

    # Stage nodes
    for sx, sy, col, lbl in stages:
        r = 70
        # glow
        for g in range(5, 0, -1):
            a = int(80 * g / 5)
            rr = r + g * 14
            draw.ellipse([sx-rr, sy-rr, sx+rr, sy+rr], fill=col[:3]+(a,))
        # circle
        draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=(12, 20, 40, 240))
        draw.ellipse([sx-r, sy-r, sx+r, sy+r], outline=col, width=4)
        # icon symbol
        if "RAW" in lbl:
            # stacked lines = file
            for i, yw in enumerate([(-20,60), (-5,50), (10,40), (25,55)]):
                y, w = yw
                draw.rounded_rectangle([sx-w//2, sy+y, sx+w//2, sy+y+10],
                                       radius=4, fill=col[:3]+(200,))
        elif "PARSE" in lbl:
            # gear teeth
            for a in range(0, 360, 45):
                rad = math.radians(a)
                tx = sx + int(45 * math.cos(rad))
                ty = sy + int(45 * math.sin(rad))
                draw.ellipse([tx-8, ty-8, tx+8, ty+8], fill=col[:3]+(220,))
            draw.ellipse([sx-22, sy-22, sx+22, sy+22], fill=(12,20,40,240))
            draw.ellipse([sx-22, sy-22, sx+22, sy+22], outline=col, width=3)
        else:
            # cylinder = storage
            draw.rounded_rectangle([sx-28, sy-32, sx+28, sy+32],
                                    radius=8, fill=col[:3]+(180,), outline=col, width=2)
            for yy in [-12, 6]:
                draw.line([(sx-24, sy+yy), (sx+24, sy+yy)], fill=(12,20,40,255), width=2)

        # label
        lines = lbl.split('\n')
        for i, ln in enumerate(lines):
            f = font(24)
            bb = draw.textbbox((0,0), ln, font=f)
            tw = bb[2]-bb[0]
            draw.text((sx-tw//2, sy+85+i*28), ln, font=f, fill=col)

    # Title
    label(draw, cx, 50, "DATA INGESTION", 32, color=C['white'])
    f2 = font(20, False)
    label(draw, cx, 92, "Powder Sintering Analogy", 20, bold=False, color=C['dim'])

    img.save(os.path.join(OUT, "loadfile.png"), "PNG")
    print("✓ loadfile.png")

# ──────────────────────────────────────────────────────────────────────────────
# 2. MISSING VALUES  →  Zone-refining reimagined as signal denoising waveform
# ──────────────────────────────────────────────────────────────────────────────
def icon_missingvalues():
    img, draw = fimg()
    cx, cy = SIZE//2, SIZE//2

    bg_panel(draw, 30, 30, SIZE-60, SIZE-60)

    # Draw two waveforms: noisy top, clean bottom
    W = 820
    x0, y_noisy, y_clean = cx - W//2, cy - 160, cy + 80

    # Noisy signal
    pts_noisy = []
    random.seed(42)
    for i in range(200):
        t = i / 199
        x = x0 + int(t * W)
        base = math.sin(t * math.pi * 6) * 80
        noise = random.gauss(0, 35)
        y = int(y_noisy + base + noise)
        pts_noisy.append((x, y))

    # draw noisy with hot color + gaps (missing values)
    missing_zones = [(40, 55), (110, 125), (160, 175)]
    for i in range(len(pts_noisy)-1):
        idx_pct = i / 199 * 200
        skip = any(lo <= idx_pct <= hi for lo, hi in missing_zones)
        if skip:
            # draw gap indicator
            x = pts_noisy[i][0]
            draw.line([(x, y_noisy-90), (x, y_noisy+90)],
                      fill=C['hot'][:3]+(60,), width=1)
        else:
            alpha = 220
            draw.line([pts_noisy[i], pts_noisy[i+1]],
                      fill=C['hot'][:3]+(alpha,), width=3)

    # Missing value markers
    for lo, hi in missing_zones:
        xl = pts_noisy[int(lo/199*len(pts_noisy))][0]
        xr = pts_noisy[min(int(hi/199*len(pts_noisy)), len(pts_noisy)-1)][0]
        draw.rectangle([xl, y_noisy-85, xr, y_noisy+85],
                       fill=C['hot'][:3]+(25,), outline=C['hot'][:3]+(140,), width=2)
        centered_text(draw, (xl+xr)//2, y_noisy,
                      "NaN", font(22), C['hot'])

    # Clean / imputed signal
    pts_clean = []
    for i in range(200):
        t = i / 199
        x = x0 + int(t * W)
        y = int(y_clean + math.sin(t * math.pi * 6) * 80)
        pts_clean.append((x, y))

    # draw as glowing clean line
    for i in range(len(pts_clean)-1):
        for w, a in [(9, 30), (5, 80), (3, 200)]:
            draw.line([pts_clean[i], pts_clean[i+1]],
                      fill=C['accent3'][:3]+(a,), width=w)

    # Axis lines
    draw.line([(x0-10, y_noisy), (x0+W+10, y_noisy)],
               fill=C['subtle'], width=1)
    draw.line([(x0-10, y_clean), (x0+W+10, y_clean)],
               fill=C['subtle'], width=1)

    # Labels
    label(draw, cx, 50, "MISSING VALUE IMPUTATION", 32, color=C['white'])
    label(draw, cx, 92, "Zone Refining Analogy", 20, bold=False, color=C['dim'])

    f = font(24, False)
    draw.text((x0-15, y_noisy - 30), "Raw", font=f, fill=C['hot'])
    draw.text((x0-15, y_clean - 30), "Clean", font=f, fill=C['accent3'])

    # Arrow connecting the two with a "purification" label
    ax = cx
    draw.line([(ax, y_noisy+100), (ax, y_clean-100)],
               fill=C['accent1'][:3]+(180,), width=3)
    draw.polygon([(ax, y_clean-105), (ax-12, y_clean-125), (ax+12, y_clean-125)],
                  fill=C['accent1'])
    centered_text(draw, ax+80, (y_noisy+y_clean)//2,
                  "PURIFY", font(22), C['accent1'])

    img.save(os.path.join(OUT, "missingvalues.png"), "PNG")
    print("✓ missingvalues.png")

# ──────────────────────────────────────────────────────────────────────────────
# 3. TRANSFORM  →  Phase transformation as crystallographic unit cell morph
# ──────────────────────────────────────────────────────────────────────────────
def icon_transform():
    img, draw = fimg()
    cx, cy = SIZE//2, SIZE//2

    bg_panel(draw, 30, 30, SIZE-60, SIZE-60)
    draw_hex_grid(draw, cx, cy, 460, spacing=50)

    def draw_cell(ox, oy, a, b, c, color, label_str):
        """Draw projected unit cell"""
        scale = 110
        # isometric projection vectors
        ax_  = ( 0.866, 0.5)
        ay_  = (-0.866, 0.5)
        az_  = (0.0,   -1.0)

        def proj(x, y, z):
            px = ox + int((ax_[0]*x + ay_[0]*y + az_[0]*z) * scale * a)
            py = oy + int((ax_[1]*x + ay_[1]*y + az_[1]*z) * scale * c)
            return (px, py)

        corners = {
            (0,0,0): proj(0,0,0), (1,0,0): proj(1,0,0),
            (0,1,0): proj(0,1,0), (1,1,0): proj(1,1,0),
            (0,0,1): proj(0,0,1), (1,0,1): proj(1,0,1),
            (0,1,1): proj(0,1,1), (1,1,1): proj(1,1,1),
        }
        edges = [
            ((0,0,0),(1,0,0)),((0,0,0),(0,1,0)),((0,0,0),(0,0,1)),
            ((1,1,0),(1,0,0)),((1,1,0),(0,1,0)),((1,1,0),(1,1,1)),
            ((1,0,1),(1,0,0)),((1,0,1),(0,0,1)),((1,0,1),(1,1,1)),
            ((0,1,1),(0,1,0)),((0,1,1),(0,0,1)),((0,1,1),(1,1,1)),
        ]
        for e1, e2 in edges:
            p1, p2 = corners[e1], corners[e2]
            for w, alpha in [(5, 40), (2, 180)]:
                draw.line([p1, p2], fill=color[:3]+(alpha,), width=w)

        # atoms at corners + body center
        atom_positions = list(corners.values())
        if b > 0.5:  # BCC body center
            atom_positions.append(proj(0.5, 0.5, 0.5))
        if abs(a-1.0) < 0.05:  # FCC face centers
            for fc in [(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5),
                       (0.5,0.5,1),(0.5,1,0.5),(1,0.5,0.5)]:
                atom_positions.append(proj(*fc))

        for ap in atom_positions:
            r = 14
            for g in range(4, 0, -1):
                aa = int(120 * g / 4)
                rr = r + g*5
                draw.ellipse([ap[0]-rr, ap[1]-rr, ap[0]+rr, ap[1]+rr],
                             fill=color[:3]+(aa,))
            draw.ellipse([ap[0]-r, ap[1]-r, ap[0]+r, ap[1]+r],
                         fill=(12,20,40,240), outline=color, width=3)

        # label
        f = font(28)
        bb = draw.textbbox((0,0), label_str, font=f)
        draw.text((ox - (bb[2]-bb[0])//2, oy+145), label_str, font=f, fill=color)

    # FCC left
    draw_cell(cx - 260, cy - 30, 1.0, 0.0, 1.0, C['hot'], "FCC Austenite")
    # BCT right (c/a = 1.1)
    draw_cell(cx + 260, cy - 30, 1.0, 1.0, 1.12, C['cold'], "BCT Martensite")

    # Central transformation arrow
    aw = 60
    for y_off in range(-aw//2, aw//2, 4):
        t = abs(y_off) / (aw//2)
        a = int(220 * (1 - t * 0.5))
        draw.line([(cx-85, cy+y_off), (cx+85, cy+y_off)],
                   fill=C['accent1'][:3]+(a,), width=2)
    draw.polygon([(cx+90, cy), (cx+60, cy-22), (cx+60, cy+22)],
                  fill=C['accent1'])
    centered_text(draw, cx, cy - 70, "QUENCH", font(24), C['accent1'])
    centered_text(draw, cx, cy - 42, "1000°C/s", font(20, False), C['dim'])

    label(draw, cx, 50, "PHASE TRANSFORMATION", 32, color=C['white'])
    label(draw, cx, 92, "Martensitic Transition", 20, bold=False, color=C['dim'])

    img.save(os.path.join(OUT, "Transform.png"), "PNG")
    print("✓ Transform.png")

# ──────────────────────────────────────────────────────────────────────────────
# 4. VISUALIZATION  →  XRD pattern as elegant diffractogram
# ──────────────────────────────────────────────────────────────────────────────
def icon_visualization():
    img, draw = fimg()
    cx, cy = SIZE//2, SIZE//2

    bg_panel(draw, 30, 30, SIZE-60, SIZE-60)

    # Axes
    ax_l, ax_r = cx - 380, cx + 380
    ax_b, ax_t = cy + 300, cy - 280

    draw_axis(draw, ax_l, ax_b, ax_r, ax_b, C['dim'], 3)
    draw_axis(draw, ax_l, ax_b, ax_l, ax_t, C['dim'], 3)

    # Axis labels
    f_ax = font(26, False)
    draw.text((cx - 30, ax_b + 30), "2θ  (°)", font=f_ax, fill=C['dim'])
    draw.text((ax_l - 60, ax_t - 10), "I", font=f_ax, fill=C['dim'])

    # 2θ ticks
    for theta, lbl in [(20,20),(40,40),(60,60),(80,80),(100,100)]:
        tx = ax_l + int((theta/110) * (ax_r - ax_l))
        draw.line([(tx, ax_b), (tx, ax_b+12)], fill=C['dim'], width=2)
        fb = font(20, False)
        bb = draw.textbbox((0,0), str(lbl), font=fb)
        draw.text((tx-(bb[2]-bb[0])//2, ax_b+16), str(lbl), font=fb, fill=C['dim'])

    H = ax_b - ax_t
    W = ax_r - ax_l

    # Baseline noise
    random.seed(7)
    noise_pts = []
    for i in range(300):
        t = i/299
        x = ax_l + int(t * W)
        y = ax_b - int((0.02 + random.gauss(0, 0.008)) * H)
        noise_pts.append((x, y))
    for i in range(len(noise_pts)-1):
        draw.line([noise_pts[i], noise_pts[i+1]], fill=C['subtle'], width=1)

    # Bragg peaks
    peaks = [
        (28, 0.42, "(110)", C['accent1']),
        (40, 1.00, "(200)", C['gold']),
        (58, 0.55, "(211)", C['accent3']),
        (73, 0.78, "(220)", C['accent4']),
        (87, 0.32, "(310)", C['accent1']),
    ]

    for theta, rel_int, hkl, col in peaks:
        px = ax_l + int((theta/110) * W)
        ph = int(rel_int * H * 0.82)

        # Gaussian peak
        sigma = 8
        pts = []
        for dx in range(-60, 61):
            gx = px + dx
            if gx < ax_l or gx > ax_r: continue
            gy = ax_b - int(ph * math.exp(-(dx**2)/(2*sigma**2)))
            pts.append((gx, gy))

        # Fill under peak
        if pts:
            poly = [(pts[0][0], ax_b)] + pts + [(pts[-1][0], ax_b)]
            # gradient fill
            for py_step in range(ph):
                frac = py_step / ph
                alpha = int(80 * frac)
                y_line = ax_b - py_step
                # find x extents
                xs = [p[0] for p in pts if abs(p[1] - y_line) < 3]
                if len(xs) >= 2:
                    draw.line([(min(xs), y_line), (max(xs), y_line)],
                               fill=col[:3]+(alpha,), width=1)
            # outline
            for i in range(len(pts)-1):
                for w, a in [(6, 40), (2, 255)]:
                    draw.line([pts[i], pts[i+1]], fill=col[:3]+(a,), width=w)

        # Miller index label
        top = (px, ax_b - ph - 15)
        f_hkl = font(22)
        bb = draw.textbbox((0,0), hkl, font=f_hkl)
        tw = bb[2]-bb[0]
        draw.text((top[0]-tw//2, top[1]-bb[3]+bb[1]-8), hkl, font=f_hkl, fill=col)

        # vertical guide
        draw.line([(px, ax_b), (px, ax_b - ph - 5)],
                   fill=col[:3]+(60,), width=1)

    #label(draw, cx, 50, "X-RAY DIFFRACTOGRAM", 32, color=C['white'])
    #label(draw, cx, 92, "Crystal Structure Fingerprint", 20, bold=False, color=C['dim'])

    img.save(os.path.join(OUT, "visualization.png"), "PNG")
    print("✓ visualization.png")

# ──────────────────────────────────────────────────────────────────────────────
# 5. CORRELATION  →  Stress-strain curve, dark + glowing zones
# ──────────────────────────────────────────────────────────────────────────────
def icon_correlation():
    img, draw = fimg()
    cx, cy = SIZE//2, SIZE//2

    bg_panel(draw, 30, 30, SIZE-60, SIZE-60)

    ax_l, ax_r = cx-360, cx+340
    ax_b, ax_t = cy+290, cy-280
    W, H = ax_r - ax_l, ax_b - ax_t

    draw_axis(draw, ax_l, ax_b, ax_r, ax_b, C['dim'], 3)
    draw_axis(draw, ax_l, ax_b, ax_l, ax_t, C['dim'], 3)

    f_ax = font(24, False)
    draw.text((cx-40, ax_b+32), "Strain  ε", font=f_ax, fill=C['dim'])
    draw.text((ax_l-50, ax_t-8), "σ", font=font(32), fill=C['dim'])

    # Curve knots (strain 0→1, stress 0→1 normalised)
    knots = [
        (0.00, 0.00), (0.04, 0.18), (0.08, 0.36),
        (0.13, 0.52), (0.18, 0.61), (0.28, 0.72),
        (0.42, 0.82), (0.58, 0.90), (0.65, 0.92),   # UTS
        (0.74, 0.88), (0.85, 0.81), (0.94, 0.72),   # fracture
    ]

    def to_px(strain, stress):
        return (ax_l + int(strain * W * 0.92),
                ax_b - int(stress * H * 0.92))

    pixel_knots = [to_px(s, sg) for s, sg in knots]

    # Shade regions
    # Elastic region
    e_end = to_px(0.13, 0)
    e_pts = [to_px(s, sg) for s, sg in knots[:4]] + [(e_end[0], ax_b), (ax_l, ax_b)]
    draw.polygon(e_pts, fill=C['accent3'][:3]+(20,))

    # Plastic region
    p_start = to_px(0.13, 0)
    uts_pt = to_px(0.65, 0)
    p_pts  = [to_px(s, sg) for s, sg in knots[3:9]] + [(uts_pt[0], ax_b), (p_start[0], ax_b)]
    draw.polygon(p_pts, fill=C['gold'][:3]+(18,))

    # Necking region
    n_start = to_px(0.65, 0)
    frac_pt = to_px(0.94, 0)
    n_pts = [to_px(s, sg) for s, sg in knots[8:]] + [(frac_pt[0], ax_b), (n_start[0], ax_b)]
    draw.polygon(n_pts, fill=C['hot'][:3]+(22,))

    # Draw curve
    colors_for_segment = []
    thresholds = [0.13, 0.65]
    seg_colors = [C['accent3'], C['gold'], C['hot']]
    for i, (s, sg) in enumerate(knots):
        if s < thresholds[0]: colors_for_segment.append(seg_colors[0])
        elif s < thresholds[1]: colors_for_segment.append(seg_colors[1])
        else: colors_for_segment.append(seg_colors[2])

    for i in range(len(pixel_knots)-1):
        col = colors_for_segment[i]
        for w, a in [(10, 30), (5, 100), (3, 255)]:
            draw.line([pixel_knots[i], pixel_knots[i+1]], fill=col[:3]+(a,), width=w)

    # Key point markers
    keypts = [
        (3, "σ_y", "Yield", C['accent3']),
        (8, "UTS", "σ_u", C['gold']),
        (11, "×", "Fracture", C['hot']),
    ]
    for idx, sym, lbl, col in keypts:
        px_, py_ = pixel_knots[idx]
        r = 14
        for g in range(4,0,-1):
            draw.ellipse([px_-r-g*5, py_-r-g*5, px_+r+g*5, py_+r+g*5],
                          fill=col[:3]+(int(50*g/4),))
        draw.ellipse([px_-r, py_-r, px_+r, py_+r],
                      fill=(12,20,40,240), outline=col, width=3)
        fb = font(20)
        bb = draw.textbbox((0,0), sym, font=fb)
        draw.text((px_-(bb[2]-bb[0])//2, py_-(bb[3]-bb[1])//2-1), sym, font=fb, fill=col)
        draw.text((px_+20, py_-36), lbl, font=font(20, False), fill=col)

    # Region labels
    draw.text((ax_l+20, ax_t+20), "Elastic", font=font(24, False), fill=C['accent3'])
    draw.text((ax_l+W//3, ax_t+20), "Plastic", font=font(24, False), fill=C['gold'])
    draw.text((ax_l+int(W*0.72), ax_t+20), "Necking", font=font(24, False), fill=C['hot'])

    label(draw, cx, 50, "STRESS–STRAIN CORRELATION", 30, color=C['white'])
    label(draw, cx, 92, "Mechanical Property Mapping", 20, bold=False, color=C['dim'])

    img.save(os.path.join(OUT, "correlation.png"), "PNG")
    print("✓ correlation.png")

# ──────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE  →  Hardness bar chart, neon + depth
# ──────────────────────────────────────────────────────────────────────────────
def icon_featureimportance():
    img, draw = fimg()
    cx, cy = SIZE//2, SIZE//2

    bg_panel(draw, 30, 30, SIZE-60, SIZE-60)

    bars = [
        ("Hardness",    0.92, C['gold']),
        ("Yield Str.",  0.78, C['accent1']),
        ("UTS",         0.70, C['accent3']),
        ("Grain Size",  0.55, C['accent4']),
        ("Porosity",    0.38, C['hot']),
        ("Roughness",   0.22, C['cold']),
    ]

    bw = 100
    gap = 36
    total_w = len(bars)*(bw+gap) - gap
    x0 = cx - total_w//2
    ax_b = cy + 310
    ax_t = cy - 200
    H = ax_b - ax_t

    # Axis
    draw_axis(draw, x0-40, ax_b, x0 + total_w + 60, ax_b, C['dim'], 3)
    draw_axis(draw, x0-40, ax_b, x0-40, ax_t, C['dim'], 3)

    # Grid lines
    for v in [0.25, 0.5, 0.75, 1.0]:
        gy = ax_b - int(v * H * 0.92)
        draw.line([(x0-40, gy), (x0+total_w+40, gy)],
                   fill=C['subtle'][:3]+(80,), width=1)
        fb = font(20, False)
        draw.text((x0-80, gy-10), f"{int(v*100)}%", font=fb, fill=C['dim'])

    for i, (name, val, col) in enumerate(bars):
        bx = x0 + i*(bw+gap)
        bh = int(val * H * 0.88)
        by = ax_b - bh

        # 3D depth shadow
        draw.rounded_rectangle([bx+6, by+6, bx+bw+6, ax_b+6],
                                 radius=8, fill=(0,0,0,80))
        # glow behind bar
        for g in range(6, 0, -1):
            aa = int(50 * g / 6)
            draw.rounded_rectangle([bx-g*2, by-g*2, bx+bw+g*2, ax_b],
                                     radius=10, fill=col[:3]+(aa,))
        # bar fill (gradient-like via segments)
        for y in range(bh):
            frac = y / bh
            r = int(col[0]*0.3 + col[0]*0.7*frac)
            g2 = int(col[1]*0.3 + col[1]*0.7*frac)
            b = int(col[2]*0.3 + col[2]*0.7*frac)
            draw.rectangle([bx, ax_b-y, bx+bw, ax_b-y+1], fill=(r,g2,b,220))
        # outline
        draw.rounded_rectangle([bx, by, bx+bw, ax_b], radius=8,
                                 outline=col, width=3)
        # top glow cap
        draw.rectangle([bx+4, by, bx+bw-4, by+6], fill=C['white'][:3]+(120,))

        # value label on top
        lbl = f"{int(val*100)}%"
        fb = font(24)
        bb = draw.textbbox((0,0), lbl, font=fb)
        draw.text((bx + (bw-(bb[2]-bb[0]))//2, by-32), lbl, font=fb, fill=col)

        # name label
        fn = font(20, False)
        bn = draw.textbbox((0,0), name, font=fn)
        draw.text((bx+(bw-(bn[2]-bn[0]))//2, ax_b+12), name, font=fn, fill=C['dim'])

    label(draw, cx, 50, "FEATURE IMPORTANCE", 32, color=C['white'])
    label(draw, cx, 92, "Vickers Hardness Analogy", 20, bold=False, color=C['dim'])

    img.save(os.path.join(OUT, "featureimportance.png"), "PNG")
    print("✓ featureimportance.png")

# ──────────────────────────────────────────────────────────────────────────────
# 7. DESCRIPTORS  →  Microstructure as radar / polar plot
# ──────────────────────────────────────────────────────────────────────────────
def icon_descriptors():
    img, draw = fimg()
    cx, cy = SIZE//2, SIZE//2 + 20

    bg_panel(draw, 30, 30, SIZE-60, SIZE-60)
    draw_hex_grid(draw, cx, cy, 430, spacing=55)

    r_max = 300
    descriptors = [
        ("Grain\nSize",   0.78, C['accent1']),
        ("Aspect\nRatio", 0.55, C['accent3']),
        ("Porosity",      0.30, C['hot']),
        ("Phase\nFrac.",  0.88, C['gold']),
        ("Boundary\nDen.",0.65, C['accent4']),
        ("Texture\nIndex",0.42, C['cold']),
    ]
    N = len(descriptors)

    # Web grid circles
    for frac in [0.25, 0.5, 0.75, 1.0]:
        r = int(frac * r_max)
        pts = []
        for i in range(N):
            ang = math.radians(i * 360/N - 90)
            pts.append((cx + int(r*math.cos(ang)), cy + int(r*math.sin(ang))))
        for i in range(N):
            draw.line([pts[i], pts[(i+1)%N]], fill=C['subtle'][:3]+(100,), width=1)
        # radial label
        if frac in [0.5, 1.0]:
            lbl = f"{int(frac*100)}%"
            draw.text((cx+4, cy - r - 16), lbl, font=font(18, False), fill=C['subtle'])

    # Spoke lines
    for i in range(N):
        ang = math.radians(i * 360/N - 90)
        ex = cx + int(r_max * math.cos(ang))
        ey = cy + int(r_max * math.sin(ang))
        draw.line([(cx, cy), (ex, ey)], fill=C['subtle'][:3]+(80,), width=1)

    # Filled radar polygon (glow)
    poly_pts = []
    for i, (name, val, col) in enumerate(descriptors):
        ang = math.radians(i * 360/N - 90)
        r = val * r_max
        poly_pts.append((cx + int(r*math.cos(ang)), cy + int(r*math.sin(ang))))

    # glow layers
    for g in range(5, 0, -1):
        offset_pts = []
        for px_, py_ in poly_pts:
            dx_, dy_ = px_ - cx, py_ - cy
            dist = math.hypot(dx_, dy_)
            if dist > 0:
                offset_pts.append((cx + int((dist+g*6)/dist*dx_),
                                    cy + int((dist+g*6)/dist*dy_)))
            else:
                offset_pts.append((px_, py_))
        draw.polygon(offset_pts, fill=C['accent1'][:3]+(int(30*g/5),))

    draw.polygon(poly_pts, fill=C['accent1'][:3]+(50,),
                 outline=C['accent1'][:3]+(220,))

    # Vertex dots and labels
    for i, (name, val, col) in enumerate(descriptors):
        ang = math.radians(i * 360/N - 90)
        vx = cx + int(val * r_max * math.cos(ang))
        vy = cy + int(val * r_max * math.sin(ang))

        # glow dot
        for g2 in range(5, 0, -1):
            draw.ellipse([vx-g2*4-8, vy-g2*4-8, vx+g2*4+8, vy+g2*4+8],
                          fill=col[:3]+(int(40*g2/5),))
        draw.ellipse([vx-10, vy-10, vx+10, vy+10],
                      fill=(12,20,40,240), outline=col, width=3)

        # axis tip label
        lx = cx + int((r_max + 60) * math.cos(ang))
        ly = cy + int((r_max + 60) * math.sin(ang))
        lines = name.split('\n')
        fnt = font(22, False)
        for j, ln in enumerate(lines):
            bb = draw.textbbox((0,0), ln, font=fnt)
            tw, th2 = bb[2]-bb[0], bb[3]-bb[1]
            draw.text((lx - tw//2, ly - th2//2 + j*(th2+3)), ln, font=fnt, fill=col)

    # Center dot
    draw.ellipse([cx-10, cy-10, cx+10, cy+10],
                  fill=C['white'][:3]+(200,))

    label(draw, cx, 50, "MICROSTRUCTURE DESCRIPTORS", 30, color=C['white'])
    label(draw, cx, 90, "Multi-Dimensional Feature Space", 20, bold=False, color=C['dim'])

    img.save(os.path.join(OUT, "descriptors.png"), "PNG")
    print("✓ descriptors.png")

# ──────────────────────────────────────────────────────────────────────────────
# 8. MACHINE LEARNING  →  Phase diagram with neural net overlay
# ──────────────────────────────────────────────────────────────────────────────
def icon_machinelearning():
    img, draw = fimg()
    cx, cy = SIZE//2, SIZE//2

    bg_panel(draw, 30, 30, SIZE-60, SIZE-60)

    ax_l, ax_r = cx-360, cx+220
    ax_b, ax_t = cy+290, cy-290
    W, H = ax_r - ax_l, ax_b - ax_t

    # Phase regions (filled)
    def px_coord(x_frac, y_frac):
        return (ax_l + int(x_frac*W), ax_b - int(y_frac*H))

    # Alpha region
    alpha_pts = [
        px_coord(0,0), px_coord(0.28,0), px_coord(0.35,0.42),
        px_coord(0.22,0.82), px_coord(0,1), px_coord(0,0)
    ]
    draw.polygon(alpha_pts, fill=C['cold'][:3]+(50,), outline=C['cold'][:3]+(150,), width=2)
    p = px_coord(0.1, 0.5)
    draw.text((p[0], p[1]), "α", font=font(64), fill=C['cold'][:3]+(200,))

    # Beta region
    beta_pts = [
        px_coord(0.7,0), px_coord(1,0), px_coord(1,1),
        px_coord(0.72,0.9), px_coord(0.65,0.42), px_coord(0.7,0)
    ]
    draw.polygon(beta_pts, fill=C['hot'][:3]+(50,), outline=C['hot'][:3]+(150,), width=2)
    p = px_coord(0.85, 0.5)
    draw.text((p[0], p[1]), "β", font=font(64), fill=C['hot'][:3]+(200,))

    # Liquid region (top)
    liq_pts = [
        px_coord(0,1), px_coord(0.22,0.82), px_coord(0.35,0.42),
        px_coord(0.65,0.42), px_coord(0.72,0.9), px_coord(1,1), px_coord(0,1)
    ]
    draw.polygon(liq_pts, fill=C['gold'][:3]+(40,), outline=C['gold'][:3]+(140,), width=2)
    p = px_coord(0.48, 0.82)
    draw.text((p[0], p[1]), "L", font=font(52), fill=C['gold'][:3]+(200,))

    # α+β region
    mix_pts = [
        px_coord(0.28,0), px_coord(0.7,0),
        px_coord(0.65,0.42), px_coord(0.35,0.42)
    ]
    draw.polygon(mix_pts, fill=C['accent3'][:3]+(40,),
                 outline=C['accent3'][:3]+(140,), width=2)
    p = px_coord(0.48, 0.22)
    draw.text((p[0], p[1]), "α+β", font=font(36), fill=C['accent3'][:3]+(200,))

    # Axes
    draw_axis(draw, ax_l, ax_b, ax_r, ax_b, C['dim'], 3)
    draw_axis(draw, ax_l, ax_b, ax_l, ax_t, C['dim'], 3)
    f_ax = font(22, False)
    draw.text((cx - 100, ax_b+32), "Composition →", font=f_ax, fill=C['dim'])
    draw.text((ax_l-55, ax_t-6), "T  (°C)", font=f_ax, fill=C['dim'])

    # ML prediction zone (confidence ellipses)
    ml_cx = ax_l + int(0.48 * W)
    ml_cy = ax_b - int(0.55 * H)
    for conf, a in [(0.95,40), (0.75,70), (0.50,110)]:
        ea = int(80 * conf)
        eb = int(55 * conf)
        draw.ellipse([ml_cx-ea, ml_cy-eb, ml_cx+ea, ml_cy+eb],
                      outline=C['accent4'][:3]+(a,), width=3)

    # NN panel (right side)
    nn_x = cx + 240
    nn_y = cy
    nn_layers = [3, 5, 4, 2]
    nn_colors = [C['accent1'], C['accent3'], C['accent4'], C['gold']]
    neuron_r = 14
    l_spacing = 80
    all_neuron_pos = []

    for li, n in enumerate(nn_layers):
        lx = nn_x + li * l_spacing
        v_spacing = 55
        ys = [nn_y + (k - (n-1)/2) * v_spacing for k in range(n)]
        all_neuron_pos.append([(lx, int(y)) for y in ys])

    # Connections
    for li in range(len(nn_layers)-1):
        for pos1 in all_neuron_pos[li]:
            for pos2 in all_neuron_pos[li+1]:
                w_val = 0.2 + random.random() * 0.6
                draw.line([pos1, pos2],
                           fill=C['accent1'][:3]+(int(40*w_val),), width=1)

    # Neurons
    for li, positions in enumerate(all_neuron_pos):
        col = nn_colors[li]
        for (nx, ny) in positions:
            for g in range(4,0,-1):
                draw.ellipse([nx-neuron_r-g*5, ny-neuron_r-g*5,
                               nx+neuron_r+g*5, ny+neuron_r+g*5],
                              fill=col[:3]+(int(30*g/4),))
            draw.ellipse([nx-neuron_r, ny-neuron_r, nx+neuron_r, ny+neuron_r],
                          fill=(12,20,40,240), outline=col, width=3)

    # NN label
    centered_text(draw, nn_x + l_spacing, nn_y + 175, "DEEP NN", font(24), C['accent4'])
    centered_text(draw, nn_x + l_spacing, nn_y + 205, "PREDICTOR", font(20, False), C['dim'])

    # Arrow from NN to phase diagram
    draw.line([(nn_x - 10, nn_y), (ml_cx + 95, ml_cy)],
               fill=C['accent4'][:3]+(180,), width=2)
    draw.polygon([(ml_cx+90, ml_cy),
                   (ml_cx+110, ml_cy-12),
                   (ml_cx+110, ml_cy+12)],
                  fill=C['accent4'])

    centered_text(draw, ml_cx, ml_cy+5, "γ' ?", font(28), C['accent4'])

    label(draw, cx - 70, 50, "MACHINE LEARNING", 32, color=C['white'])
    label(draw, cx - 70, 92, "Phase Diagram Prediction", 20, bold=False, color=C['dim'])

    img.save(os.path.join(OUT, "machinelearning.png"), "PNG")
    print("✓ machinelearning.png")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(0)
    print("\n=== Generating Reimagined EDA Icons ===\n")
    icon_loadfile()
    icon_missingvalues()
    icon_transform()
    icon_visualization()
    icon_correlation()
    icon_featureimportance()
    icon_descriptors()
    icon_machinelearning()
    print(f"\n✓ All icons saved to  '{OUT}/'")
