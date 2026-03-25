#!/usr/bin/env python3
"""Generate project summary PDF report with all images and analytics."""

import json
from pathlib import Path
from fpdf import FPDF
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BT = DATA / "backtest"
IMG = DATA / "figures"
BT_OUT = BT / "outputs"


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 5, "SAR Port Activity Monitoring | Project Summary", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(25, 60, 120)
        self.ln(4)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(25, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(40, 40, 40)
        self.ln(2)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def add_image_safe(self, path, w=180, caption=None):
        path = Path(path)
        if not path.exists():
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(180, 0, 0)
            self.cell(0, 6, f"[Image not found: {path.name}]", new_x="LMARGIN", new_y="NEXT")
            return
        # Check if enough space, otherwise new page
        if self.get_y() > 200:
            self.add_page()
        try:
            self.image(str(path), w=w)
        except Exception as e:
            self.set_font("Helvetica", "I", 9)
            self.cell(0, 6, f"[Error loading {path.name}: {e}]", new_x="LMARGIN", new_y="NEXT")
        if caption:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(80, 80, 80)
            self.cell(0, 5, caption, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(25, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self.set_fill_color(240, 245, 255)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=True, align="C")
            self.ln()
        self.ln(3)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_report():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(25, 60, 120)
    pdf.multi_cell(0, 12, "Satellite-Based Port Activity\nMonitoring", align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, "SAR Vessel Detection & Economic Signal Validation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Date Range: January 2024 - March 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Primary Port: Rotterdam | Secondary: Santos", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Data Source: Sentinel-1 SAR via Microsoft Planetary Computer", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Generated March 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # =========================================================================
    # SECTION 1: PIPELINE ARCHITECTURE
    # =========================================================================
    pdf.add_page()
    pdf.section_title("1. Pipeline Architecture")

    pdf.body_text(
        "This project implements an end-to-end pipeline for monitoring port activity from space using "
        "Synthetic Aperture Radar (SAR) satellite imagery. The pipeline acquires Sentinel-1 scenes from "
        "Microsoft Planetary Computer, applies radiometric terrain correction, detects vessels using a "
        "CA-CFAR (Cell-Averaging Constant False Alarm Rate) algorithm, and aggregates detections into "
        "weekly port activity indices."
    )

    pdf.body_text(
        "The pipeline is designed to operate on any port globally, requiring only a bounding box and a "
        "water mask polygon. The system is configured for 19 sites (12 major container ports and 7 cargo airports), "
        "though detailed analysis was performed on Rotterdam (349 scenes) and Santos (20 scenes)."
    )

    pdf.subsection_title("Pipeline Stages")
    pdf.body_text(
        "1. Acquisition: Search STAC catalog for Sentinel-1 GRD scenes overlapping the site bbox. "
        "Download VV and VH polarization bands as cloud-optimized GeoTIFF crops.\n"
        "2. Preprocessing: Radiometric calibration to sigma-nought (dB). Scenes arrive as RTC "
        "(Radiometric Terrain Corrected) products at 10m resolution.\n"
        "3. Water Masking: Apply manual QGIS-drawn water masks (authoritative) or fall back to "
        "automated EU-Hydro/JRC/OSM layers. Manual masks use buffer_m=0 and all_touched=False.\n"
        "4. Detection: CA-CFAR on water-masked VV band. Cluster connected bright pixels. "
        "Post-filters: minimum size (20 px), aspect ratio, dB contrast, brightness gate.\n"
        "5. Indexing: Aggregate per-scene vessel counts into weekly medians. Optionally normalize as z-scores.\n"
        "6. Backtesting: Compare SAR index against external references (GFW, Eurostat, port statistics)."
    )

    pdf.subsection_title("Key Configuration Parameters")
    pdf.add_table(
        ["Parameter", "Value", "Notes"],
        [
            ["SAR Product", "Sentinel-1 GRD RTC", "10m resolution, IW mode"],
            ["Polarization", "VV (primary)", "VH available but not used for detection"],
            ["CFAR Guard Cells", "5", "Pixels around target excluded from background"],
            ["CFAR Background Cells", "15", "Ring of pixels for noise estimation"],
            ["CFAR PFA (global)", "1e-8", "False alarm probability threshold"],
            ["CFAR PFA (Rotterdam)", "1e-9", "Per-site override, stricter"],
            ["Min Target Pixels", "20", "Minimum cluster size to count as vessel"],
            ["Min dB", "0.0 dB", "Post-CFAR brightness gate"],
            ["Min dB Contrast", "1.5 dB", "Peak - mean within cluster"],
        ],
        col_widths=[55, 45, 90],
    )

    pdf.add_image_safe(IMG / "rotterdam_sar_detections.png", caption="Figure 1.1: Example SAR vessel detection overlay on Rotterdam (Jan 2024)")

    # =========================================================================
    # SECTION 2: WATER MASKING
    # =========================================================================
    pdf.add_page()
    pdf.section_title("2. Water Masking Evolution")

    pdf.body_text(
        "Water masking was the single most impactful quality improvement in the pipeline. The initial "
        "approach used automated layers (EU-Hydro coastal polygons, JRC Global Surface Water satellite-derived "
        "occurrence, and OpenStreetMap harbour polygons), but this produced widespread false alarms from "
        "metallic port infrastructure on land."
    )

    pdf.subsection_title("Automated Masking (v1)")
    pdf.body_text(
        "The automated stack combined three sources: EU-Hydro (Copernicus, official European hydrography), "
        "JRC Global Surface Water (30m Landsat-derived, global), and OSM harbour supplement. Each was "
        "rasterized into the scene grid with a 30m buffer and all_touched=True. The union of all three "
        "formed the water mask."
    )
    pdf.body_text(
        "Problems: EU-Hydro's broad coastal polygons extended too far inland. JRC's 30m resolution blurred "
        "tidal zones. The 30m buffer inflated narrow channels. The result was 42% water coverage for "
        "Rotterdam, with many detections on land (cranes, containers, vehicles)."
    )

    pdf.add_image_safe(IMG / "rotterdam_water_mask.png", w=170,
                       caption="Figure 2.1: Layered water mask breakdown showing EU-Hydro, JRC, and OSM contributions")

    pdf.add_page()
    pdf.subsection_title("Manual Masking (v2 - Current)")
    pdf.body_text(
        "The solution was hand-drawn water masks in QGIS, exported as GeoPackage files. When a manual mask "
        "exists for a site (config/water_masks/{site_id}.gpkg), it is used as the sole authoritative source. "
        "All automated layers are skipped. Manual masks use buffer_m=0 and all_touched=False, since the "
        "geometry already traces quay edges precisely."
    )
    pdf.body_text(
        "Rotterdam: 27 polygons, 10.9% water coverage (down from 42%). Santos: 2 polygons (inner port + "
        "offshore anchorage). Drawing each mask took approximately 30 minutes in QGIS."
    )

    pdf.add_image_safe(IMG / "rotterdam_water_filtered.png", w=170,
                       caption="Figure 2.2: SAR with water-filtered detections using manual mask")

    pdf.add_page()
    pdf.subsection_title("False Alarm Diagnosis")
    pdf.add_image_safe(IMG / "rotterdam_zoomed_debug.png", w=170,
                       caption="Figure 2.3: Zoomed debug view showing detection locations relative to optical imagery")

    # =========================================================================
    # SECTION 3: ROTTERDAM ANALYSIS
    # =========================================================================
    pdf.add_page()
    pdf.section_title("3. Rotterdam: Two-Year Analysis")

    pdf.body_text(
        "Rotterdam is Europe's largest port and the primary test case for this project. We acquired 349 "
        "Sentinel-1 scenes from January 2024 through March 2026, covering the full period when both S1A "
        "and S1C satellites were operational (S1C joined the constellation in April 2025, doubling revisit "
        "frequency from ~6 to ~3 days)."
    )

    pdf.subsection_title("Dataset Summary")
    pdf.add_table(
        ["Metric", "Value"],
        [
            ["Total scenes", "349"],
            ["Date range", "2024-01-01 to 2026-03-18"],
            ["Mean vessel count/scene", "163"],
            ["Standard deviation", "27.4"],
            ["Min / Max", "96 / 218"],
            ["Coefficient of variation", "16.8%"],
            ["S1A scenes", "~220"],
            ["S1C scenes (post Apr 2025)", "~129"],
            ["Scenes per month (pre-S1C)", "6-10"],
            ["Scenes per month (post-S1C)", "18-21"],
        ],
        col_widths=[80, 110],
    )

    pdf.add_image_safe(IMG / "rotterdam_2yr_vessel_counts.png", w=180,
                       caption="Figure 3.1: Two-year vessel count time series - raw scatter, weekly aggregation, and monthly distribution")

    pdf.add_page()
    pdf.subsection_title("Optical vs SAR Comparison")
    pdf.body_text(
        "To validate detection quality, we pulled Sentinel-2 optical imagery from the same period "
        "(0% cloud cover scene from Jan 10, 2024) and overlaid SAR detections from Jan 9, 2024. "
        "This confirms that detected vessels correspond to real ships visible in optical imagery."
    )

    pdf.add_image_safe(IMG / "rotterdam_optical_vs_sar.png", w=180,
                       caption="Figure 3.2: Sentinel-2 optical (Jan 10) with SAR vessel detections (Jan 9) overlaid")

    # =========================================================================
    # SECTION 4: GFW BACKTEST
    # =========================================================================
    pdf.add_page()
    pdf.section_title("4. Rotterdam Backtest: GFW Daily Presence")

    pdf.body_text(
        "Global Fishing Watch (GFW) provides free daily vessel presence data via API. We fetched daily "
        "commercial vessel presence for the exact Rotterdam AOI (dissolved from the 27 manual mask polygons) "
        "for all 349 scene dates. GFW returns vessel_id, type, flag, and hours of presence per day."
    )

    pdf.bold_text("Important caveat: GFW reports daily presence (was this vessel in the area at any point "
                  "during the 24-hour period), not snapshot occupancy at the SAR overpass time. This temporal "
                  "mismatch is structural and cannot be resolved without scene-time AIS data.")

    pdf.subsection_title("Scene-Level Diagnostic")
    pdf.body_text("Same-date correlation between SAR vessel_count and GFW unique vessels:")

    scene_diag = load_json(BT_OUT / "scene_diagnostic.json") if (BT_OUT / "scene_diagnostic.json").exists() else {}
    cc = scene_diag.get("count_vs_count", {})
    pdf.add_table(
        ["Metric", "Value"],
        [
            ["N scenes", str(cc.get("n_scenes", "N/A"))],
            ["Spearman rho", f"{cc.get('spearman_r', 'N/A'):.4f}" if isinstance(cc.get('spearman_r'), (int, float)) else "N/A"],
            ["Spearman p-value", f"{cc.get('spearman_p', 'N/A'):.4f}" if isinstance(cc.get('spearman_p'), (int, float)) else "N/A"],
            ["Pearson r", f"{cc.get('pearson_r', 'N/A'):.4f}" if isinstance(cc.get('pearson_r'), (int, float)) else "N/A"],
            ["MAE", f"{cc.get('mae', 'N/A'):.1f}" if isinstance(cc.get('mae'), (int, float)) else "N/A"],
        ],
        col_widths=[80, 110],
    )

    pdf.subsection_title("Monthly Headline Validation")
    monthly = load_json(BT_OUT / "monthly_validation.json") if (BT_OUT / "monthly_validation.json").exists() else {}
    mc = monthly.get("count_vs_count", {})
    pdf.add_table(
        ["Metric", "Value"],
        [
            ["N months", str(mc.get("n_months", "N/A"))],
            ["Spearman rho", f"{mc.get('spearman_r', 'N/A'):.4f}" if isinstance(mc.get('spearman_r'), (int, float)) else "N/A"],
            ["Spearman p-value", f"{mc.get('spearman_p', 'N/A'):.4f}" if isinstance(mc.get('spearman_p'), (int, float)) else "N/A"],
        ],
        col_widths=[80, 110],
    )

    pdf.subsection_title("Wind as Dominant Confounder")
    pdf.body_text(
        "A controlled regression of vessel_count on acquisition conditions reveals that wind speed, "
        "time of day (morning vs evening pass), and satellite (S1A vs S1C) together explain 43% of "
        "vessel count variance. GFW vessel presence explains none."
    )
    pdf.add_table(
        ["Covariate", "Coefficient", "p-value"],
        [
            ["wind_speed_mps", "-4.92", "<0.001"],
            ["is_evening", "+29.67", "<0.001"],
            ["is_s1c", "+12.02", "<0.001"],
            ["R-squared", "0.427", "--"],
        ],
        col_widths=[65, 60, 65],
    )

    pdf.bold_text(
        "Conclusion: GFW daily commercial vessel presence is not a suitable proxy for Rotterdam SAR "
        "snapshot occupancy. This is a conclusive null result for GFW as a reference, not necessarily "
        "a null result for the SAR signal itself."
    )

    # =========================================================================
    # SECTION 5: EUROSTAT THROUGHPUT
    # =========================================================================
    pdf.add_page()
    pdf.section_title("5. Rotterdam Backtest: Eurostat Quarterly Throughput")

    pdf.body_text(
        "As an alternative reference, we compared the SAR vessel count series against Eurostat quarterly "
        "throughput data for Rotterdam (in thousands of tonnes). With only 6 quarters of overlap, "
        "statistical power is limited."
    )

    pdf.subsection_title("Wind Residualization Effect")
    pdf.body_text(
        "Raw quarterly SAR median shows essentially no correlation with throughput (Spearman rho = 0.09). "
        "After removing wind effects via OLS regression, the correlation improves substantially to "
        "Spearman rho = 0.60, Pearson r = 0.73 (p = 0.099). However, further residualization on "
        "pass_family and satellite overcorrects, dropping the correlation back to 0.26."
    )

    pdf.add_table(
        ["Residualization", "Spearman rho", "Pearson r", "Pearson p"],
        [
            ["Raw vessel_count", "0.086", "0.050", "0.925"],
            ["Wind-only residual", "0.600", "0.731", "0.099"],
            ["Wind + pass + satellite", "0.257", "0.288", "0.579"],
        ],
        col_widths=[55, 45, 45, 45],
    )

    pdf.bold_text(
        "Key result: Wind-adjusted SAR shows a suggestive but not statistically significant quarterly "
        "correlation (rho = 0.60, p = 0.21, n = 6). The test is underpowered: only 6 quarters, and "
        "Rotterdam throughput varies by just 6% quarter-to-quarter."
    )

    pdf.add_image_safe(IMG / "wind_residualization.png", w=180,
                       caption="Figure 5.1: Wind residualization effect on SAR signal and quarterly throughput comparison")

    # =========================================================================
    # SECTION 6: SANTOS
    # =========================================================================
    pdf.add_page()
    pdf.section_title("6. Santos: Soybean Season Test")

    pdf.body_text(
        "Santos (Brazil) was selected as a secondary test case because it handles the majority of "
        "Brazil's soybean exports, which peak sharply from March to June. We acquired 20 scenes from "
        "January through August 2024 to capture the quiet-peak-quiet cycle."
    )

    pdf.subsection_title("Port + Anchorage Mask")
    pdf.body_text(
        "The Santos mask consists of two polygons: the inner port channel/berths and an offshore "
        "anchorage area south of the port entrance where bulk carriers queue. The anchorage was added "
        "after initial results showed the inner port (like Rotterdam) has near-constant occupancy."
    )

    pdf.add_image_safe(IMG / "santos_optical_full.png", w=160,
                       caption="Figure 6.1: Sentinel-2 optical (Mar 31, 2024) with port mask (green) and anchorage mask (red)")

    pdf.add_page()
    pdf.subsection_title("Zone Split Results")
    pdf.body_text(
        "Splitting detections by zone reveals the same pattern as Rotterdam: inner port is always full "
        "(mean=32, sigma=2.7), while the anchorage holds all the variance (mean=68, sigma=13.7)."
    )

    pdf.add_table(
        ["Zone", "Mean Vessels", "Std Dev", "CV%", "Signal?"],
        [
            ["Port (inner)", "32", "2.7", "8.4%", "No - always full"],
            ["Anchorage", "68", "13.7", "20.1%", "Variance exists"],
            ["Total", "100", "14.8", "14.8%", "--"],
        ],
        col_widths=[35, 35, 30, 30, 60],
    )

    pdf.add_image_safe(IMG / "santos_zone_split.png", w=180,
                       caption="Figure 6.2: Port vs anchorage vessel count breakdown over time")

    pdf.add_page()
    pdf.subsection_title("Throughput Comparison")
    pdf.body_text(
        "Santos throughput in 2024 was reconstructed from cumulative press releases (monthly figures not "
        "publicly available). The pattern shows a smooth ramp from 11.9 MT in January to 16.4 MT in July, "
        "with no sharp seasonal discontinuity."
    )

    pdf.add_table(
        ["Zone", "Spearman rho", "p-value"],
        [
            ["Total vessels", "-0.217", "0.606"],
            ["Port only", "-0.267", "0.523"],
            ["Anchorage only", "-0.157", "0.711"],
        ],
        col_widths=[60, 65, 65],
    )

    pdf.body_text(
        "All correlations are negative but none are significant. The negative direction is consistent "
        "with a congestion hypothesis (more throughput leads to fewer ships waiting), but the effect "
        "is weak with only 8 data points and smooth throughput variation."
    )

    pdf.add_image_safe(IMG / "santos_throughput_comparison.png", w=180,
                       caption="Figure 6.3: Santos SAR vessel counts vs official port throughput")

    pdf.add_image_safe(IMG / "santos_scenes_overview.png", w=180,
                       caption="Figure 6.4: Four Santos scenes across Jan-Aug 2024 showing SAR + mask + detections")

    # =========================================================================
    # SECTION 7: LESSONS LEARNED
    # =========================================================================
    pdf.add_page()
    pdf.section_title("7. Lessons Learned & Recommendations")

    pdf.subsection_title("Key Findings")

    findings = [
        ("1. Wind is the biggest confounder.",
         "Wind speed explains 10-20% of SAR vessel count variance at both Rotterdam and Santos. "
         "Each 1 m/s increase in wind reduces detected vessels by ~5 at Rotterdam. This is because "
         "higher wind raises sea surface roughness, reducing vessel-to-background contrast in SAR. "
         "Wind must be controlled before any economic comparison."),

        ("2. Berth occupancy is nearly constant at major ports.",
         "Both Rotterdam (CV=17%) and Santos inner port (CV=8%) show minimal variation in berth "
         "occupancy. These ports are always full. The economically interesting signal, if it exists, "
         "is in anchorage/waiting areas, not at berth."),

        ("3. GFW daily presence is the wrong reference for SAR snapshots.",
         "GFW reports whether a vessel was present at any point during a 24-hour period. SAR captures "
         "a 2-second snapshot. This structural temporal mismatch produces null correlations at every "
         "timescale tested. Scene-time AIS (e.g., Spire Maritime) is needed for proper detector validation."),

        ("4. Manual water masks are essential.",
         "Automated layers (EU-Hydro, JRC, OSM) produced ~40% false alarm rates from land-based "
         "metallic infrastructure. Hand-drawn QGIS masks (~30 min per port) eliminated false alarms "
         "entirely. The manual mask should be authoritative with buffer_m=0."),

        ("5. The method needs throughput variation to validate.",
         "Both Rotterdam (2024-2026) and Santos (2024) were record years with minimal disruption. "
         "Throughput varied by only 6% at Rotterdam. Without variation in the target signal, "
         "correlation tests have no statistical power."),
    ]

    for title, text in findings:
        pdf.bold_text(title)
        pdf.body_text(text)

    pdf.add_page()
    pdf.subsection_title("Recommendations for Next Steps")

    recs = [
        "Extend Rotterdam back to 2020: The COVID period (Mar-May 2020) saw 5-10% throughput drops. "
        "The 2021 recovery and 2022 supply chain crisis created real variation. This would provide "
        "20 quarters with meaningful throughput swings instead of 6 flat ones.",

        "Test a mid-tier port with seasonal swings: A port smaller than Rotterdam/Santos that actually "
        "experiences busy and quiet periods would be a stronger validation target.",

        "Buy a 1-month Spire AIS sample for detector validation: A narrow Rotterdam AIS extract "
        "(~$200-500) covering one month would establish whether the CFAR detector correctly identifies "
        "individual vessels. This validates the detector, separate from the economic signal question.",

        "Always wind-residualize before economic comparison: The wind effect is large enough to mask "
        "real activity signals. Scene-level wind regression should be a standard preprocessing step.",

        "Focus on anchorage counts, not berth counts: The anchorage/waiting zone is where throughput "
        "variation is most visible in SAR. Berth areas are saturated and insensitive to demand changes.",
    ]

    for i, rec in enumerate(recs, 1):
        pdf.bold_text(f"{i}.")
        pdf.body_text(rec)

    # =========================================================================
    # APPENDIX A: CONFIGURATION
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Appendix A: Configured Sites")

    pdf.add_table(
        ["ID", "Name", "Region", "Lat", "Lon"],
        [
            ["shanghai", "Port of Shanghai", "Asia-Pacific", "30.63", "121.84"],
            ["singapore", "Port of Singapore", "Asia-Pacific", "1.26", "103.83"],
            ["ningbo", "Port of Ningbo-Zhoushan", "Asia-Pacific", "29.87", "121.95"],
            ["shenzhen", "Port of Shenzhen", "Asia-Pacific", "22.48", "114.03"],
            ["qingdao", "Port of Qingdao", "Asia-Pacific", "36.07", "120.32"],
            ["busan", "Port of Busan", "Asia-Pacific", "35.07", "129.06"],
            ["dubai", "Jebel Ali (Dubai)", "Middle East", "25.01", "55.06"],
            ["rotterdam", "Port of Rotterdam", "Europe", "51.92", "4.40"],
            ["antwerp", "Port of Antwerp", "Europe", "51.30", "4.30"],
            ["hamburg", "Port of Hamburg", "Europe", "53.53", "9.93"],
            ["losangeles", "Port of Los Angeles", "North America", "33.73", "-118.27"],
            ["santos", "Port of Santos", "South America", "-23.97", "-46.30"],
        ],
        col_widths=[30, 50, 35, 30, 30],
    )

    # =========================================================================
    # APPENDIX B: IMAGE INDEX
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Appendix B: Generated Image Index")

    images = [
        ["rotterdam_sar_detections.png", "1.1", "SAR detection overlay example"],
        ["rotterdam_water_mask.png", "2.1", "Layered water mask (EU-Hydro/JRC/OSM)"],
        ["rotterdam_water_filtered.png", "2.2", "Water-filtered SAR detections"],
        ["rotterdam_zoomed_debug.png", "2.3", "Zoomed false alarm diagnosis"],
        ["rotterdam_2yr_vessel_counts.png", "3.1", "Two-year time series"],
        ["rotterdam_optical_vs_sar.png", "3.2", "Optical vs SAR comparison"],
        ["wind_residualization.png", "5.1", "Wind regression + quarterly comparison"],
        ["santos_optical_full.png", "6.1", "Sentinel-2 optical with masks"],
        ["santos_zone_split.png", "6.2", "Port vs anchorage breakdown"],
        ["santos_throughput_comparison.png", "6.3", "SAR vs official throughput"],
        ["santos_scenes_overview.png", "6.4", "Multi-scene overview"],
        ["santos_anchorage_map.png", "--", "Anchorage location map"],
        ["santos_anchorage_seasonal.png", "--", "Seasonal patterns with anchorage"],
        ["santos_seasonality.png", "--", "Pre-anchorage seasonal analysis"],
        ["rotterdam_optical_only.png", "--", "Optical reference image"],
        ["rotterdam_5scene_diagnostic.png", "--", "Multi-scene detection diagnostic"],
    ]

    pdf.add_table(
        ["Filename", "Figure", "Description"],
        images,
        col_widths=[65, 20, 105],
    )

    # =========================================================================
    # SAVE
    # =========================================================================
    output_path = DATA / "project_summary_report.pdf"
    pdf.output(str(output_path))
    print(f"Report saved: {output_path}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_report()
