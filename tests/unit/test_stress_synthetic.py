"""Empirical Stress Tests and Challenge Harness for Synthetic Art Generator (Milestone 1).

Attacks src/data/synthetic.py (SyntheticArtDatasetGenerator):
1. Seed determinism across multiple runs and distinct Python processes.
2. Procedural image rendering across extreme resolutions (8x8 miniature, 1024x1024 poster, asymmetric 8x1024).
3. Benchmark throughput and memory consumption for 500+ generated images.
4. Statistical verification of price distributions (Tier 1 > Tier 2 > Tier 3 medians with p < 0.001).
5. Downstream pipeline compatibility with SchemaValidator and ArtDataCleaner.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import scipy.stats as stats

from src.data.cleaner import ArtDataCleaner
from src.data.synthetic import SyntheticArtDatasetGenerator
from src.data.validator import validate_artwork_schema


# ============================================================================
# 1. Seed Determinism Tests
# ============================================================================

class TestSyntheticSeedDeterminism:
    """Attacks reproducibility and determinism across generator instances and processes."""

    def test_in_process_identical_seed_reproducibility(self, tmp_path):
        """Two generators initialized with identical seed must produce identical images and metadata."""
        dir1 = tmp_path / "run_a"
        dir2 = tmp_path / "run_b"

        gen1 = SyntheticArtDatasetGenerator(seed=2026)
        df1, _ = gen1.generate_benchmark(n_samples=25, output_dir=dir1)

        gen2 = SyntheticArtDatasetGenerator(seed=2026)
        df2, _ = gen2.generate_benchmark(n_samples=25, output_dir=dir2)

        # Drop machine-local path for DataFrame equality check
        pd.testing.assert_frame_equal(
            df1.drop(columns=["image_path"]),
            df2.drop(columns=["image_path"]),
        )

        # Bit-for-bit file equality across all 25 generated images
        for i in range(1, 26):
            img1_bytes = (dir1 / f"images/synthetic/art_{i:05d}.jpg").read_bytes()
            img2_bytes = (dir2 / f"images/synthetic/art_{i:05d}.jpg").read_bytes()
            assert img1_bytes == img2_bytes, f"Image {i} binary mismatch between runs"

    def test_cross_process_seed_determinism(self, tmp_path):
        """Distinct spawned Python processes must produce identical outputs given identical seed."""
        worker_code = """
import hashlib, json, sys
from pathlib import Path
from src.data.synthetic import SyntheticArtDatasetGenerator

out_dir, seed_str, n_str = sys.argv[1], sys.argv[2], sys.argv[3]
gen = SyntheticArtDatasetGenerator(seed=int(seed_str))
df, csv_path = gen.generate_benchmark(n_samples=int(n_str), output_dir=out_dir)

hashes = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in df["image_path"]]
summary = {
    "records": df.drop(columns=["image_path"]).to_dict(orient="records"),
    "hashes": hashes,
}
print(json.dumps(summary))
"""
        dir_p1 = tmp_path / "proc1"
        dir_p2 = tmp_path / "proc2"

        proc1 = subprocess.run(
            [sys.executable, "-c", worker_code, str(dir_p1), "42", "20"],
            capture_output=True,
            text=True,
            check=True,
        )
        proc2 = subprocess.run(
            [sys.executable, "-c", worker_code, str(dir_p2), "42", "20"],
            capture_output=True,
            text=True,
            check=True,
        )

        res1 = json.loads(proc1.stdout.strip().split("\n")[-1])
        res2 = json.loads(proc2.stdout.strip().split("\n")[-1])

        assert res1["records"] == res2["records"], "Metadata discrepancy across processes"
        assert res1["hashes"] == res2["hashes"], "Binary image hash discrepancy across processes"

    def test_seed_entropy_variation(self, tmp_path):
        """Different seeds must produce diverging distributions and distinct values."""
        gen_a = SyntheticArtDatasetGenerator(seed=101)
        df_a, _ = gen_a.generate_benchmark(n_samples=30, output_dir=tmp_path / "ent_a")

        gen_b = SyntheticArtDatasetGenerator(seed=999)
        df_b, _ = gen_b.generate_benchmark(n_samples=30, output_dir=tmp_path / "ent_b")

        matching_prices = (df_a["price"] == df_b["price"]).sum()
        assert matching_prices <= 2, f"Too many matching prices across different seeds: {matching_prices}"


# ============================================================================
# 2. Extreme Resolution & Rendering Stress Tests
# ============================================================================

class TestProceduralRenderingExtremes:
    """Stress tests procedural rendering against extreme resolutions, aspects, and boundary sizes."""

    @pytest.fixture
    def generator(self):
        return SyntheticArtDatasetGenerator(seed=777)

    @pytest.mark.parametrize("size", [
        (8, 8),          # Miniature
        (1024, 1024),    # Poster
        (8, 1024),       # Extremely tall aspect
        (1024, 8),       # Extremely wide aspect
        (16, 16),        # Small square
        (17, 31),        # Non-power-of-two primes
    ])
    def test_all_styles_render_across_extreme_resolutions(self, generator, size):
        """Verify that every procedural style renders without crashes or coordinate overflow."""
        styles = list(generator.STYLE_PREMIUMS.keys())
        palettes = list(generator.PALETTES.keys())

        for style in styles:
            palette = palettes[0]
            img = generator.generate_image(style=style, palette=palette, size=size)
            assert isinstance(img, Image.Image)
            assert img.size == size
            assert img.mode == "RGB"

            arr = np.array(img)
            assert arr.shape == (size[1], size[0], 3)
            assert arr.dtype == np.uint8
            assert arr.min() >= 0 and arr.max() <= 255

            if min(size) >= 8:
                assert arr.std() > 0.0, f"Image rendered flat for style={style}, size={size}"

    def test_all_palettes_valid_and_distinct(self, generator):
        """Verify all 5 palettes render distinct color distributions."""
        mean_colors = {}
        for palette_name in generator.PALETTES.keys():
            img = generator.generate_image(style="color_field", palette=palette_name, size=(64, 64))
            arr = np.array(img)
            mean_colors[palette_name] = arr.mean(axis=(0, 1))

        # Check that palettes differ from each other
        palettes = list(mean_colors.keys())
        for i in range(len(palettes)):
            for j in range(i + 1, len(palettes)):
                dist = np.linalg.norm(mean_colors[palettes[i]] - mean_colors[palettes[j]])
                assert dist > 5.0, f"Palettes {palettes[i]} and {palettes[j]} too similar"

    def test_rendering_style_and_palette_aliases(self, generator):
        """Verify case-insensitive string parsing, alias resolution, and whitespace trimming."""
        img1 = generator.generate_image(style="  ROTHKO  ", palette="  EARTHY  ", size=(32, 32))
        img2 = generator.generate_image(style="POLLOCK", palette="MONOCHROME", size=(32, 32))
        img3 = generator.generate_image(style="Minimalist", palette="Neon", size=(32, 32))
        img4 = generator.generate_image(style="pointillist", palette="pastel", size=(32, 32))

        for img in [img1, img2, img3, img4]:
            assert img.size == (32, 32)
            assert img.mode == "RGB"

    def test_invalid_style_and_palette_rejection(self, generator):
        """Unsupported styles or palettes must raise ValueError with informative message."""
        with pytest.raises(ValueError, match="Unknown style"):
            generator.generate_image(style="renaissance_chiaroscuro")

        with pytest.raises(ValueError, match="Unknown palette"):
            generator.generate_image(palette="holographic_chrome")


# ============================================================================
# 3. Throughput & Memory Stress Tests (500+ Images)
# ============================================================================

class TestBenchmarkThroughputAndMemory:
    """Stress tests throughput and peak memory consumption for 500+ generated images."""

    def test_benchmark_throughput_500_samples(self, tmp_path):
        """Benchmark generation of 500 images: measure images/sec and execution latency."""
        gen = SyntheticArtDatasetGenerator(seed=42)
        n_samples = 500

        t0 = time.perf_counter()
        df, csv_path = gen.generate_benchmark(n_samples=n_samples, output_dir=tmp_path / "bench500")
        t1 = time.perf_counter()

        elapsed = t1 - t0
        throughput = n_samples / elapsed

        assert len(df) == n_samples
        assert os.path.exists(csv_path)
        # Require throughput of at least 100 images per second
        assert throughput >= 100.0, f"Throughput {throughput:.2f} img/s below 100 img/s requirement"

    def test_memory_stability_across_successive_batches(self, tmp_path):
        """Verify memory stays tightly bounded across 5 successive batches of 100 artworks."""
        tracemalloc.start()
        gen = SyntheticArtDatasetGenerator(seed=123)

        peak_memories: List[float] = []
        for batch in range(5):
            df, _ = gen.generate_benchmark(n_samples=100, output_dir=tmp_path / f"batch_{batch}")
            _, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024 * 1024)
            peak_memories.append(peak_mb)
            assert len(df) == 100

        tracemalloc.stop()

        # Peak memory must not exceed 25 MB throughout the run
        max_peak = max(peak_memories)
        assert max_peak < 25.0, f"Peak memory {max_peak:.2f} MB exceeded 25 MB ceiling"


# ============================================================================
# 4. Statistical Verification of Hedonic Price Distributions
# ============================================================================

class TestStatisticalHedonicDistributions:
    """Statistical hypothesis testing verifying price distribution properties."""

    @pytest.mark.parametrize("seed", [42, 100, 999])
    def test_tier_price_monotonicity_and_p_values(self, tmp_path, seed):
        """Verify Median(Tier 1) > Median(Tier 2) > Median(Tier 3) with Mann-Whitney p < 0.001."""
        gen = SyntheticArtDatasetGenerator(seed=seed)
        df, _ = gen.generate_benchmark(n_samples=250, output_dir=tmp_path / f"stat_{seed}")

        tier1_prices = df[df["artist_tier"] == "Tier 1 - Master"]["price"]
        tier2_prices = df[df["artist_tier"] == "Tier 2 - Mid-Career"]["price"]
        tier3_prices = df[df["artist_tier"] == "Tier 3 - Emerging"]["price"]

        med1 = tier1_prices.median()
        med2 = tier2_prices.median()
        med3 = tier3_prices.median()

        # Strict median ordering
        assert med1 > med2, f"Tier 1 median (${med1:,.0f}) did not exceed Tier 2 (${med2:,.0f})"
        assert med2 > med3, f"Tier 2 median (${med2:,.0f}) did not exceed Tier 3 (${med3:,.0f})"

        # Mann-Whitney U test (one-sided greater)
        _, p_12 = stats.mannwhitneyu(tier1_prices, tier2_prices, alternative="greater")
        _, p_23 = stats.mannwhitneyu(tier2_prices, tier3_prices, alternative="greater")

        assert p_12 < 0.001, f"p-value for Tier 1 > Tier 2 ({p_12:.2e}) exceeded 0.001"
        assert p_23 < 0.001, f"p-value for Tier 2 > Tier 3 ({p_23:.2e}) exceeded 0.001"

        # Kruskal-Wallis across all 3 tiers
        _, kw_p = stats.kruskal(tier1_prices, tier2_prices, tier3_prices)
        assert kw_p < 0.001, f"Kruskal-Wallis p-value ({kw_p:.2e}) exceeded 0.001"

    def test_hedonic_parameters_bounds_and_validity(self, tmp_path):
        """Verify all generated values obey physical and econometric boundary limits."""
        gen = SyntheticArtDatasetGenerator(seed=42)
        df, _ = gen.generate_benchmark(n_samples=300, output_dir=tmp_path / "bounds")

        # Strictly positive dimensions and prices
        assert (df["price"] >= 100.0).all(), "Prices must be >= 100.0"
        assert (df["height"] >= 15.0).all(), "Height must be >= 15.0 cm"
        assert (df["width"] >= 15.0).all(), "Width must be >= 15.0 cm"
        assert (df["height"] <= 450.0).all(), "Height exceeds 450.0 cm"
        assert (df["width"] <= 450.0).all(), "Width exceeds 450.0 cm"

        # Historical year bounds
        assert (df["year"] >= 1870).all(), "Year is earlier than 1870"
        assert (df["year"] <= 2026).all(), "Year is in the future"

        # No nulls or NaNs in generated records
        assert not df.isna().any().any(), "Synthetic benchmark contains NaN values"


# ============================================================================
# 5. Downstream Pipeline Compatibility
# ============================================================================

class TestDownstreamPipelineIntegration:
    """Verify synthetic benchmark outputs pass SchemaValidator and ArtDataCleaner seamlessly."""

    def test_benchmark_to_validator_and_cleaner(self, tmp_path):
        """Synthetic benchmark data must pass validator and cleaner without drops or errors."""
        gen = SyntheticArtDatasetGenerator(seed=42)
        df, _ = gen.generate_benchmark(n_samples=50, output_dir=tmp_path / "integration")

        # 1. Validate Schema
        validated_df = validate_artwork_schema(df, is_training=True)
        assert len(validated_df) == 50

        # 2. Clean and engineer features
        cleaner = ArtDataCleaner()
        cleaned_df = cleaner.clean(validated_df, is_training=True)

        assert len(cleaned_df) == 50
        for expected_col in ["surface_area", "aspect_ratio", "age", "log_price"]:
            assert expected_col in cleaned_df.columns
            assert not cleaned_df[expected_col].isna().any()
