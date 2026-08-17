import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
SOURCES = [
    ROOT / "results/qwen2.5-0.5b-service-recovery.analysis.json",
    ROOT / "results/qwen2.5-1.5b-service-recovery.analysis.json",
    ROOT / "results/qwen2.5-0.5b-queue-control.analysis.json",
    ROOT / "results/qwen2.5-1.5b-queue-control.analysis.json",
]

DECODERS = [
    ("greedy", "Greedy"),
    ("temperature_0.5", "Temperature 0.5"),
    ("temperature_1.5", "Temperature 1.5"),
    ("top_k_2", "Top-k 2"),
    ("top_p_0.6", "Top-p 0.6"),
    ("top_p_0.9", "Top-p 0.9"),
    ("min_p_0.1", "Min-p 0.1"),
    ("min_p_0.3", "Min-p 0.3"),
    ("temperature_0.5_then_top_p_0.8", "Temp. 0.5 -> top-p 0.8"),
    ("top_p_0.8_then_temperature_0.5", "Top-p 0.8 -> temp. 0.5"),
]

PANEL_ORDER = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "service_recovery"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "service_recovery"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "queue_control"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "queue_control"),
]

PANEL_TITLES = {
    (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "service_recovery",
    ): "Qwen2.5-0.5B | Service recovery",
    (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "service_recovery",
    ): "Qwen2.5-1.5B | Service recovery",
    ("Qwen/Qwen2.5-0.5B-Instruct", "queue_control"): "Qwen2.5-0.5B | Queue control",
    ("Qwen/Qwen2.5-1.5B-Instruct", "queue_control"): "Qwen2.5-1.5B | Queue control",
}


def load_rows():
    rows = []
    for source in SOURCES:
        data = json.loads(source.read_text())
        for result in data["results"]:
            if result["decoder"] == "raw":
                continue
            rows.append(
                {
                    "source_artifact": str(source.relative_to(ROOT)),
                    "model_id": data["model_id"],
                    "environment": data["environment"],
                    "decoder": result["decoder"],
                    "label_mapping": result["label_mapping"],
                    "decoder_value_gap": result["decoder_value_gap"],
                }
            )
    return rows


def write_csv(rows):
    path = FIGURES / "decoder-value-gaps.csv"
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def esc(value):
    return html.escape(str(value), quote=True)


def text(x, y, value, size=24, weight=400, anchor="start", fill="#17202a", extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}" {extra}>{esc(value)}</text>'
    )


def round_out(value, step, direction):
    scaled = value / step
    return (math.floor(scaled) if direction < 0 else math.ceil(scaled)) * step


def render_svg(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        panel = (row["model_id"], row["environment"])
        grouped[panel][row["decoder"]].append(row)

    values = [row["decoder_value_gap"] for row in rows]
    tick_step = 0.2
    domain_min = round_out(min(values) - 0.03, tick_step, -1)
    domain_max = round_out(max(values) + 0.03, tick_step, 1)
    ticks = []
    tick = domain_min
    while tick <= domain_max + 1e-9:
        ticks.append(round(tick, 10))
        tick += tick_step

    width = 2000
    height = 1460
    panel_width = 890
    panel_height = 535
    panel_positions = [(80, 245), (1030, 245), (80, 850), (1030, 850)]
    label_width = 260
    plot_left_inset = label_width
    plot_right_inset = 30
    plot_top_inset = 62
    plot_bottom_inset = 55
    plot_width = panel_width - plot_left_inset - plot_right_inset
    row_height = (panel_height - plot_top_inset - plot_bottom_inset) / len(DECODERS)

    navy = "#244d70"
    orange = "#b55a2a"
    ink = "#17202a"
    muted = "#53616d"
    grid = "#d9dee3"
    border = "#aab4bc"
    background = "#ffffff"

    def scale_x(value, panel_x):
        start = panel_x + plot_left_inset
        return start + (value - domain_min) / (domain_max - domain_min) * plot_width

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">'
        ),
        '<title id="title">Decoder rules shift exact policy value</title>',
        '<desc id="description">Four range-and-dot panels compare decoder value gaps across two Qwen2.5 models and two finite decision environments. Each row contains six exhaustive label mappings, their minimum-to-maximum range, and their mean.</desc>',
        f'<rect width="{width}" height="{height}" fill="{background}"/>',
        '<g font-family="Helvetica Neue, Arial, sans-serif">',
        text(80, 74, "Decoder rules shift exact policy value", size=42, weight=500),
        text(
            80,
            116,
            "Each row compares a decoded policy with raw softmax on the same cached logits.",
            size=24,
            fill=muted,
        ),
        text(
            80,
            151,
            "Open circles are all six A/B/C label mappings; the diamond is their mean; the line spans min to max.",
            size=24,
            fill=muted,
        ),
    ]

    # Keep the key close to the subtitle so the four panels can share it.
    legend_y = 201
    parts.extend(
        [
            f'<circle cx="92" cy="{legend_y}" r="6" fill="#ffffff" stroke="{navy}" stroke-width="2"/>',
            text(111, legend_y + 7, "Label mapping", size=21, fill=muted),
            f'<line x1="299" y1="{legend_y}" x2="359" y2="{legend_y}" stroke="{navy}" stroke-width="3"/>',
            text(374, legend_y + 7, "Min-max", size=21, fill=muted),
            f'<polygon points="525,{legend_y - 8} 533,{legend_y} 525,{legend_y + 8} 517,{legend_y}" fill="{orange}"/>',
            text(545, legend_y + 7, "Mean", size=21, fill=muted),
        ]
    )

    jitter = (-8, -5, -2, 2, 5, 8)
    for panel, (panel_x, panel_y) in zip(PANEL_ORDER, panel_positions):
        plot_top = panel_y + plot_top_inset
        plot_bottom = panel_y + panel_height - plot_bottom_inset
        plot_left = panel_x + plot_left_inset
        plot_right = plot_left + plot_width

        parts.append(
            text(panel_x, panel_y + 30, PANEL_TITLES[panel], size=27, weight=500)
        )
        parts.append(
            f'<rect x="{plot_left:.1f}" y="{plot_top:.1f}" width="{plot_width:.1f}" '
            f'height="{plot_bottom - plot_top:.1f}" fill="none" stroke="{border}" stroke-width="1.5"/>'
        )

        for tick in ticks:
            x = scale_x(tick, panel_x)
            is_zero = abs(tick) < 1e-9
            parts.append(
                f'<line x1="{x:.1f}" y1="{plot_top:.1f}" x2="{x:.1f}" y2="{plot_bottom:.1f}" '
                f'stroke="{ink if is_zero else grid}" stroke-width="{2.3 if is_zero else 1}"/>'
            )
            label = "0" if is_zero else f"{tick:+.1f}"
            parts.append(
                text(x, plot_bottom + 29, label, size=19, anchor="middle", fill=muted)
            )

        for index, (decoder, decoder_label) in enumerate(DECODERS):
            y = plot_top + (index + 0.5) * row_height
            if index > 0:
                parts.append(
                    f'<line x1="{plot_left:.1f}" y1="{y - row_height / 2:.1f}" '
                    f'x2="{plot_right:.1f}" y2="{y - row_height / 2:.1f}" stroke="{grid}" stroke-width="0.8"/>'
                )
            parts.append(
                text(
                    plot_left - 14,
                    y + 7,
                    decoder_label,
                    size=20,
                    anchor="end",
                    fill=ink,
                )
            )

            observations = sorted(
                grouped[panel][decoder], key=lambda item: item["label_mapping"]
            )
            if len(observations) != 6:
                raise ValueError(f"Expected six label mappings for {panel} {decoder}")
            gaps = [item["decoder_value_gap"] for item in observations]
            x_min = scale_x(min(gaps), panel_x)
            x_max = scale_x(max(gaps), panel_x)
            mean = sum(gaps) / len(gaps)
            x_mean = scale_x(mean, panel_x)
            parts.append(
                f'<line x1="{x_min:.1f}" y1="{y:.1f}" x2="{x_max:.1f}" y2="{y:.1f}" '
                f'stroke="{navy}" stroke-width="3" stroke-linecap="round"/>'
            )
            for offset, observation in zip(jitter, observations):
                x = scale_x(observation["decoder_value_gap"], panel_x)
                label = observation["label_mapping"]
                value = observation["decoder_value_gap"]
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y + offset:.1f}" r="5.2" fill="#ffffff" '
                    f'stroke="{navy}" stroke-width="2"><title>{esc(label)}: {value:+.4f}</title></circle>'
                )
            points = (
                f"{x_mean:.1f},{y - 8:.1f} {x_mean + 8:.1f},{y:.1f} "
                f"{x_mean:.1f},{y + 8:.1f} {x_mean - 8:.1f},{y:.1f}"
            )
            parts.append(f'<polygon points="{points}" fill="{orange}"/>')

        parts.append(
            text(
                (plot_left + plot_right) / 2,
                panel_y + panel_height - 4,
                "Decoder value gap (decoded return - raw return)",
                size=21,
                anchor="middle",
                fill=ink,
            )
        )

    parts.extend(
        [
            text(
                80,
                1417,
                "Exact finite-MDP values; the six mappings are exhaustive strata, not repeated samples. All panels use the same x-axis.",
                size=20,
                fill=muted,
            ),
            text(
                80,
                1447,
                "Data: results/qwen2.5-{0.5b,1.5b}-{service-recovery,queue-control}.analysis.json",
                size=18,
                fill=muted,
            ),
            "</g>",
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main():
    rows = load_rows()
    expected_rows = len(SOURCES) * len(DECODERS) * 6
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(rows)}")
    write_csv(rows)
    (FIGURES / "decoder-value-gaps.svg").write_text(render_svg(rows))


if __name__ == "__main__":
    main()
