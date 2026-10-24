"""Procedural synthetic artwork benchmark generator for CanvasIQ.

Generates reproducible synthetic 2D canvas images and paired fine art auction
metadata adhering to empirical econometric distributions.
"""

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


class SyntheticArtDatasetGenerator:
    """Procedural benchmark generator for synthetic artwork images and paired metadata."""

    # Curated color palettes with RGB 3-tuples
    PALETTES: Dict[str, List[Tuple[int, int, int]]] = {
        "earthy_ochre": [
            (160, 82, 45),   # Burnt Sienna
            (115, 74, 18),   # Raw Umber
            (204, 119, 34),  # Yellow Ochre
            (226, 114, 91),  # Terracotta
            (107, 142, 35),  # Muted Olive
            (255, 248, 220), # Warm Cream
            (192, 64, 0),    # Burnt Orange
        ],
        "oceanic_cool": [
            (0, 49, 83),     # Prussian Blue
            (42, 82, 190),   # Cerulean
            (0, 31, 63),     # Deep Navy
            (115, 194, 251), # Seafoam Cyan
            (112, 128, 144), # Slate Grey
            (127, 255, 212), # Pale Aquamarine
            (240, 248, 255), # Cool White
        ],
        "monochrome": [
            (10, 10, 10),    # Stark Black
            (34, 34, 34),    # Deep Charcoal
            (85, 85, 85),    # Medium Slate
            (136, 136, 136), # Ash Grey
            (204, 204, 204), # Silver Mist
            (250, 249, 246), # Ivory White
        ],
        "neon_vibrant": [
            (255, 0, 127),   # Electric Magenta
            (57, 255, 20),   # Acid Lime
            (0, 240, 255),   # Hyper Cyan
            (255, 95, 31),   # Fluorescent Orange
            (121, 40, 202),  # Vivid Violet
            (255, 255, 0),   # Electric Yellow
        ],
        "pastel_wash": [
            (176, 224, 230), # Powder Blue
            (255, 182, 193), # Blush Rose
            (152, 255, 152), # Mint Green
            (230, 230, 250), # Soft Lavender
            (255, 253, 208), # Buttercup Cream
            (255, 229, 180), # Pale Peach
        ],
    }

    # Artist pedigree tiers and market base parameters
    ARTIST_TIERS: Dict[str, dict] = {
        "Tier 1 - Master": {
            "artists": [
                "Pablo Picasso", "Claude Monet", "Mark Rothko",
                "Jackson Pollock", "Georgia O'Keeffe", "Willem de Kooning",
                "Gerhard Richter", "Jean-Michel Basquiat"
            ],
            "weight": 0.15,
            "base_log": 13.5,
            "year_range": (1870, 1995),
        },
        "Tier 2 - Mid-Career": {
            "artists": [
                "Elena Vance", "Marcus Thorne", "David Sterling",
                "Sophia Laurent", "Julian Finch", "Ananya Rao",
                "Hiroshi Tanaka", "Chloe Dubois"
            ],
            "weight": 0.45,
            "base_log": 10.5,
            "year_range": (1970, 2020),
        },
        "Tier 3 - Emerging": {
            "artists": [
                "Leo Chen", "Maya Patel", "Liam O'Connor",
                "Amara Okafor", "Zara Novak", "Felix Meyer",
                "Noah Kim", "Suki Takahashi"
            ],
            "weight": 0.40,
            "base_log": 8.0,
            "year_range": (2000, 2024),
        },
    }

    # Medium weights and econometric coefficients
    MEDIA_EFFECTS: Dict[str, Tuple[float, float]] = {
        "Oil on Canvas": (0.40, 0.85),
        "Acrylic on Canvas": (0.25, 0.40),
        "Mixed Media": (0.15, 0.15),
        "Watercolor on Paper": (0.12, -0.35),
        "Lithograph / Print": (0.08, -1.10),
    }

    # Style market demand premiums
    STYLE_PREMIUMS: Dict[str, float] = {
        "color_field": 0.20,
        "abstract_splatter": 0.25,
        "geometric": 0.05,
        "impressionist": 0.15,
    }

    def __init__(self, seed: int = 42) -> None:
        """Initialize generator with deterministic seed.

        Args:
            seed: Integer random seed for full reproducibility.
        """
        self.seed = seed
        self.py_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def _render_color_field(self, colors: List[Tuple[int, int, int]], size: Tuple[int, int]) -> Image.Image:
        w, h = size
        bg = self.py_rng.choice(colors)
        img = Image.new("RGB", size, bg)
        draw = ImageDraw.Draw(img)
        num_blocks = self.py_rng.randint(2, 4)
        y_stride = max(1, h // (num_blocks + 1))
        for i in range(num_blocks):
            color_candidates = [c for c in colors if c != bg] or colors
            color = self.py_rng.choice(color_candidates)
            x0_min = int(w * 0.08)
            x0_max = max(x0_min, int(w * 0.18))
            x0 = self.py_rng.randint(x0_min, x0_max)
            x1_min = max(x0 + 1, int(w * 0.82))
            x1_max = max(x1_min, int(w * 0.92))
            x1 = self.py_rng.randint(x1_min, x1_max)
            y0 = int((i + 0.3) * y_stride)
            y1 = max(y0 + 1, int((i + 1.2) * y_stride))
            draw.rectangle([x0, y0, x1, y1], fill=color)
        blur_radius = max(1, min(self.py_rng.randint(6, 12), max(1, min(w, h) // 4)))
        return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def _render_abstract_splatter(self, colors: List[Tuple[int, int, int]], size: Tuple[int, int]) -> Image.Image:
        w, h = size
        bg = self.py_rng.choice(colors)
        img = Image.new("RGB", size, bg)
        draw = ImageDraw.Draw(img)
        # Curved lines
        for _ in range(self.py_rng.randint(14, 24)):
            color = self.py_rng.choice(colors)
            points = [(self.py_rng.randint(0, max(1, w - 1)), self.py_rng.randint(0, max(1, h - 1))) for _ in range(self.py_rng.randint(4, 7))]
            draw.line(points, fill=color, width=self.py_rng.randint(1, max(1, min(4, min(w, h) // 10))))
        # Splatter droplets
        for _ in range(self.py_rng.randint(40, 90)):
            color = self.py_rng.choice(colors)
            cx = self.py_rng.randint(0, max(1, w - 1))
            cy = self.py_rng.randint(0, max(1, h - 1))
            r = self.py_rng.randint(1, max(1, min(5, min(w, h) // 10)))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        return img

    def _render_geometric(self, colors: List[Tuple[int, int, int]], size: Tuple[int, int]) -> Image.Image:
        w, h = size
        bg = self.py_rng.choice(colors)
        img = Image.new("RGB", size, bg)
        draw = ImageDraw.Draw(img)
        mode = self.py_rng.choice(["grid", "concentric", "polygons"])
        if mode == "grid":
            rows, cols = self.py_rng.randint(2, 4), self.py_rng.randint(2, 4)
            r_step, c_step = max(1, h // rows), max(1, w // cols)
            for r in range(rows):
                for c in range(cols):
                    draw.rectangle([c * c_step, r * r_step, (c + 1) * c_step, (r + 1) * r_step], fill=self.py_rng.choice(colors))
            line_col = (20, 20, 20)
            line_w = max(1, min(3, min(w, h) // 10))
            for r in range(rows + 1):
                draw.line([(0, r * r_step), (w, r * r_step)], fill=line_col, width=line_w)
            for c in range(cols + 1):
                draw.line([(c * c_step, 0), (c * c_step, h)], fill=line_col, width=line_w)
        elif mode == "concentric":
            max_r = max(5, min(w, h) // 2)
            center = (w // 2, h // 2)
            radii = sorted([self.py_rng.randint(max(1, max_r // 5), max_r) for _ in range(5)], reverse=True)
            for r in radii:
                draw.ellipse([center[0] - r, center[1] - r, center[0] + r, center[1] + r], fill=self.py_rng.choice(colors))
        else:
            for _ in range(self.py_rng.randint(4, 8)):
                pts = [(self.py_rng.randint(0, max(1, w - 1)), self.py_rng.randint(0, max(1, h - 1))) for _ in range(self.py_rng.randint(3, 5))]
                draw.polygon(pts, fill=self.py_rng.choice(colors))
        return img

    def _render_impressionist(self, colors: List[Tuple[int, int, int]], size: Tuple[int, int]) -> Image.Image:
        w, h = size
        bg = self.py_rng.choice(colors)
        img = Image.new("RGB", size, bg)
        draw = ImageDraw.Draw(img)
        num_dabs = self.py_rng.randint(350, 650)
        for _ in range(num_dabs):
            c = self.py_rng.choice(colors)
            jitter = (self.py_rng.randint(-15, 15), self.py_rng.randint(-15, 15), self.py_rng.randint(-15, 15))
            c_jitter = tuple(max(0, min(255, c[j] + jitter[j])) for j in range(3))
            x, y = self.py_rng.randint(0, max(1, w - 1)), self.py_rng.randint(0, max(1, h - 1))
            rx = self.py_rng.randint(1, max(2, min(6, w // 10)))
            ry = self.py_rng.randint(1, max(1, min(4, h // 15)))
            draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=c_jitter)
        return img

    def generate_image(
        self,
        style: Optional[str] = None,
        palette: Optional[str] = None,
        size: Tuple[int, int] = (224, 224),
    ) -> Image.Image:
        """Generate a procedural artwork image in RGB mode."""
        style_aliases = {
            "rothko": "color_field",
            "colorfield": "color_field",
            "color_field": "color_field",
            "pollock": "abstract_splatter",
            "abstract": "abstract_splatter",
            "splatter": "abstract_splatter",
            "abstract_splatter": "abstract_splatter",
            "minimalist": "geometric",
            "geometric": "geometric",
            "pointillist": "impressionist",
            "impressionist": "impressionist",
        }
        palette_aliases = {
            "earthy": "earthy_ochre",
            "ochre": "earthy_ochre",
            "earthy_ochre": "earthy_ochre",
            "oceanic": "oceanic_cool",
            "cool": "oceanic_cool",
            "oceanic_cool": "oceanic_cool",
            "monochrome": "monochrome",
            "neon": "neon_vibrant",
            "vibrant": "neon_vibrant",
            "neon_vibrant": "neon_vibrant",
            "pastel": "pastel_wash",
            "wash": "pastel_wash",
            "pastel_wash": "pastel_wash",
        }

        if style is None or style.lower() == "random":
            style = self.py_rng.choice(list(self.STYLE_PREMIUMS.keys()))
        else:
            style = style_aliases.get(style.lower().strip(), style.lower().strip())

        if style not in self.STYLE_PREMIUMS:
            raise ValueError(f"Unknown style '{style}'. Supported: {list(self.STYLE_PREMIUMS.keys())}")

        if palette is None or palette.lower() == "random":
            palette = self.py_rng.choice(list(self.PALETTES.keys()))
        else:
            palette = palette_aliases.get(palette.lower().strip(), palette.lower().strip())

        if palette not in self.PALETTES:
            raise ValueError(f"Unknown palette '{palette}'. Supported: {list(self.PALETTES.keys())}")

        colors = self.PALETTES[palette]

        if style == "color_field":
            return self._render_color_field(colors, size)
        elif style == "abstract_splatter":
            return self._render_abstract_splatter(colors, size)
        elif style == "geometric":
            return self._render_geometric(colors, size)
        elif style == "impressionist":
            return self._render_impressionist(colors, size)
        else:
            raise ValueError(f"Unsupported style handler for '{style}'")

    def generate_benchmark(
        self,
        n_samples: int = 200,
        output_dir: Union[str, Path] = "data",
    ) -> Tuple[pd.DataFrame, str]:
        """Generate paired benchmark dataset containing metadata and saved procedural images.

        Args:
            n_samples: Number of synthetic artwork records to generate. Default: 200.
            output_dir: Base directory where images and CSV metadata will be saved.

        Returns:
            Tuple of (pd.DataFrame, str) with metadata and CSV file path.
        """
        base_dir = Path(output_dir)
        img_dir = base_dir / "images" / "synthetic"
        raw_dir = base_dir / "raw"
        img_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for i in range(n_samples):
            art_id = f"art_{i+1:05d}"
            img_rel_path = f"images/synthetic/{art_id}.jpg"
            img_abs_path = base_dir / img_rel_path

            # 1. Tier selection
            r_tier = self.py_rng.random()
            if r_tier < 0.15:
                tier_name = "Tier 1 - Master"
            elif r_tier < 0.60:
                tier_name = "Tier 2 - Mid-Career"
            else:
                tier_name = "Tier 3 - Emerging"

            tier_info = self.ARTIST_TIERS[tier_name]
            artist = self.py_rng.choice(tier_info["artists"])
            year = self.py_rng.randint(*tier_info["year_range"])

            # 2. Medium selection
            r_med = self.py_rng.random()
            cum = 0.0
            medium = "Oil on Canvas"
            med_effect = 0.85
            for med_name, (w_med, beta_med) in self.MEDIA_EFFECTS.items():
                cum += w_med
                if r_med <= cum:
                    medium = med_name
                    med_effect = beta_med
                    break

            # 3. Style and palette selection
            style = self.py_rng.choice(list(self.STYLE_PREMIUMS.keys()))
            palette = self.py_rng.choice(list(self.PALETTES.keys()))
            style_effect = self.STYLE_PREMIUMS[style]

            # 4. Physical dimensions (log-normal in cm)
            height = round(max(15.0, min(350.0, float(np.exp(self.np_rng.normal(4.2, 0.45))))), 1)
            width = round(max(15.0, min(450.0, float(np.exp(self.np_rng.normal(4.4, 0.45))))), 1)
            area = height * width

            # 5. Hedonic price calculation
            area_effect = 0.45 * math.log(max(100.0, area) / 1000.0)
            age_effect = 0.50 * max(0.0, (2026 - year) / 100.0)
            eps = float(self.np_rng.normal(0.0, 0.35))
            log_price = tier_info["base_log"] + med_effect + area_effect + age_effect + style_effect + eps
            price = round(max(100.0, math.exp(log_price) - 1.0), 2)

            # 6. Procedural image rendering and disk persistence
            image = self.generate_image(style=style, palette=palette, size=(224, 224))
            image.save(img_abs_path, format="JPEG", quality=90)

            title_style = style.replace("_", " ").title()
            title_palette = palette.replace("_", " ").title()
            title = f"{title_style} in {title_palette} #{i+1}"

            records.append({
                "artwork_id": art_id,
                "title": title,
                "artist": artist,
                "artist_tier": tier_name,
                "year": year,
                "medium": medium,
                "height": height,
                "width": width,
                "style": style,
                "palette": palette,
                "price": price,
                "image_path": str(img_abs_path),
            })

        columns = [
            "artwork_id",
            "title",
            "artist",
            "artist_tier",
            "year",
            "medium",
            "height",
            "width",
            "style",
            "palette",
            "price",
            "image_path",
        ]
        df = pd.DataFrame(records, columns=columns)
        csv_path = raw_dir / "synthetic_artworks.csv"
        df.to_csv(csv_path, index=False)

        return df, str(csv_path)
