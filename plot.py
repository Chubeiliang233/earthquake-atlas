# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///

"""Turn saved USGS earthquakes into a two-layer seismic archive."""

import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data" / "usgs-earthquakes-2.5-month.geojson"
OUTPUT = HERE / "out" / "earthquake-atlas.png"

BACKGROUND = "#090b09"
WHITE = "#edf0ea"
SECONDARY = "#899188"
LIME = "#b7ff2a"
BOUNDS = (98, 178, -20, 60)
DEPTH_LIMIT = 70


def read_events(path):
    """Return longitude, latitude, depth, magnitude and date for each event."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = []
    for feature in raw["features"]:
        coordinates = feature["geometry"]["coordinates"]
        magnitude = feature["properties"]["mag"]
        if magnitude is None or len(coordinates) < 3:
            continue
        date = dt.datetime.fromtimestamp(
            feature["properties"]["time"] / 1000, dt.timezone.utc
        ).date()
        events.append(
            (coordinates[0], coordinates[1], coordinates[2], magnitude, date)
        )
    return events


def gaussian_smooth(field, radius):
    """Blur a count grid with a one-dimensional Gaussian in both directions."""
    half = int(radius * 4)
    positions = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (positions / radius) ** 2)
    kernel /= kernel.sum()
    field = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), 1, field
    )
    return np.apply_along_axis(
        lambda column: np.convolve(column, kernel, mode="same"), 0, field
    )


def density_surface(events):
    """Count events on a geographic grid, then soften the count field."""
    west, east, south, north = BOUNDS
    longitudes = np.linspace(west - 20, east + 20, 240)
    latitudes = np.linspace(south - 20, north + 20, 240)
    counts = np.zeros((len(latitudes), len(longitudes)))
    for longitude, latitude, _depth, _magnitude, _date in events:
        if (longitudes[0] <= longitude <= longitudes[-1]
                and latitudes[0] <= latitude <= latitudes[-1]):
            column = int(np.searchsorted(longitudes, longitude).clip(
                0, len(longitudes) - 1
            ))
            row = int(np.searchsorted(latitudes, latitude).clip(
                0, len(latitudes) - 1
            ))
            counts[row, column] += 1
    return longitudes, latitudes, gaussian_smooth(counts, radius=14)


def sample_surface(events, size=43):
    """Sample a density field on the grid used by the poster."""
    longitudes, latitudes, field = density_surface(events)
    longitude_grid, latitude_grid = np.meshgrid(
        np.linspace(BOUNDS[0], BOUNDS[1], size),
        np.linspace(BOUNDS[2], BOUNDS[3], size),
    )
    columns = np.searchsorted(longitudes, longitude_grid).clip(
        0, len(longitudes) - 1
    )
    rows = np.searchsorted(latitudes, latitude_grid).clip(
        0, len(latitudes) - 1
    )
    sampled = field[rows, columns]
    scale = max(float(np.quantile(sampled, 0.995)), 0.0001)
    strength = np.clip(sampled / scale, 0, 1) ** 0.65
    return longitude_grid, latitude_grid, strength


def project(longitude, latitude, height=0, layer="upper"):
    """Project geographic positions into an enlarged isometric field."""
    west, east, south, north = BOUNDS
    x = 2 * (np.asarray(longitude) - west) / (east - west) - 1
    y = 2 * (np.asarray(latitude) - south) / (north - south) - 1
    poster_x = 0.50 + 0.22 * (x + y)
    base = 0.50 if layer == "upper" else 0.30
    poster_y = base + 0.11 * (y - x) + 0.29 * height
    return poster_x, poster_y


def add_grain(axes):
    """Add a fixed, subtle grain so every run makes the same image."""
    random = np.random.default_rng(5913)
    axes.scatter(
        random.random(4200), random.random(4200),
        s=random.uniform(0.08, 0.55, 4200), color=WHITE,
        alpha=0.035, linewidths=0, zorder=0,
    )


def main():
    events = read_events(DATA)
    visible = [
        event for event in events
        if BOUNDS[0] <= event[0] <= BOUNDS[1]
        and BOUNDS[2] <= event[1] <= BOUNDS[3]
    ]
    shallow = [event for event in events if event[2] < DEPTH_LIMIT]
    deep = [event for event in events if event[2] >= DEPTH_LIMIT]
    visible_shallow = [event for event in visible if event[2] < DEPTH_LIMIT]
    visible_deep = [event for event in visible if event[2] >= DEPTH_LIMIT]

    longitude_grid, latitude_grid, shallow_strength = sample_surface(shallow)
    _, _, deep_strength = sample_surface(deep)

    figure = plt.figure(figsize=(10, 12), facecolor=BACKGROUND)
    axes = figure.add_axes([0, 0, 1, 1], facecolor=BACKGROUND)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_axis_off()
    add_grain(axes)

    # Lower layer: deep earthquakes become a field of sampled dots.
    for row in range(0, 43, 2):
        for column in range(0, 43, 2):
            strength = float(deep_strength[row, column])
            x, y = project(
                longitude_grid[row, column], latitude_grid[row, column],
                layer="lower",
            )
            axes.plot(
                x, y, ".", color=WHITE,
                markersize=0.35 + 4.8 * strength ** 1.5,
                alpha=0.12 + 0.76 * strength, zorder=1,
            )

    # Thin guide lines connect matching corners without pretending to be depth.
    for longitude, latitude in [(98, -20), (178, -20), (178, 60), (98, 60)]:
        lower_x, lower_y = project(longitude, latitude, layer="lower")
        upper_x, upper_y = project(longitude, latitude, layer="upper")
        axes.plot(
            [lower_x, upper_x], [lower_y, upper_y],
            color=SECONDARY, linewidth=0.35, alpha=0.22, zorder=2,
        )

    # Upper layer: shallow-earthquake density lifts the horizontal contour lines.
    for row in range(43):
        x, y = project(
            longitude_grid[row], latitude_grid[row], shallow_strength[row],
            layer="upper",
        )
        major = row % 6 == 0
        axes.plot(
            x, y, color=WHITE,
            linewidth=0.92 if major else 0.40,
            alpha=0.92 if major else 0.54, zorder=3,
        )
    for column in (0, 14, 28, 42):
        x, y = project(
            longitude_grid[:, column], latitude_grid[:, column],
            shallow_strength[:, column], layer="upper",
        )
        axes.plot(x, y, color=WHITE, linewidth=0.30, alpha=0.18, zorder=3)

    # The lime locator marks the strongest earthquake inside the shown region.
    strongest = max(visible, key=lambda event: event[3])
    mark_layer = "upper" if strongest[2] < DEPTH_LIMIT else "lower"
    if mark_layer == "upper":
        column = int(np.abs(longitude_grid[0] - strongest[0]).argmin())
        row = int(np.abs(latitude_grid[:, 0] - strongest[1]).argmin())
        mark_height = shallow_strength[row, column]
    else:
        mark_height = 0
    mark_x, mark_y = project(
        strongest[0], strongest[1], mark_height, layer=mark_layer
    )
    axes.scatter(
        [mark_x], [mark_y], s=52, facecolors=BACKGROUND,
        edgecolors=LIME, linewidths=1.0, zorder=6,
    )

    # A filled point projects the deep event onto the upper field.
    if mark_layer == "lower":
        column = int(np.abs(longitude_grid[0] - strongest[0]).argmin())
        row = int(np.abs(latitude_grid[:, 0] - strongest[1]).argmin())
        upper_x, upper_y = project(
            strongest[0], strongest[1], shallow_strength[row, column],
            layer="upper",
        )
        axes.scatter(
            [upper_x], [upper_y], s=24, color=LIME,
            edgecolors=BACKGROUND, linewidths=0.45, zorder=6,
        )

    axes.plot(
        [mark_x, mark_x], [mark_y + 0.012, min(mark_y + 0.082, 0.86)],
        color=LIME, linewidth=0.55, zorder=5,
    )
    locator_top = min(mark_y + 0.082, 0.86)
    axes.plot(
        [mark_x, min(mark_x + 0.075, 0.91)], [locator_top, locator_top],
        color=LIME, linewidth=0.55, zorder=5,
    )
    axes.text(
        mark_x + 0.006, min(mark_y + 0.090, 0.868),
        f"M {strongest[3]:.1f} / {strongest[2]:.0f} KM",
        color=LIME, fontsize=6.5, zorder=6,
    )

    first = min(event[4] for event in events)
    last = max(event[4] for event in events)
    axes.text(0.075, 0.943, "SEISMIC ARCHIVE", color=WHITE,
              fontsize=27, weight="bold")
    axes.text(0.075, 0.916, "WESTERN PACIFIC / M2.5+ / LAST 30 DAYS",
              color=SECONDARY, fontsize=7.5)
    axes.text(0.925, 0.943, "02 / DEPTH FIELD", color=SECONDARY,
              fontsize=7, ha="right")
    axes.text(0.925, 0.916, f"{first} — {last} UTC", color=SECONDARY,
              fontsize=7, ha="right")
    axes.plot([0.075, 0.925], [0.897, 0.897], color=SECONDARY,
              linewidth=0.35, alpha=0.45)

    axes.text(0.075, 0.850, "A / SHALLOW", color=WHITE,
              fontsize=8, weight="bold")
    axes.text(0.075, 0.830, f"DEPTH < {DEPTH_LIMIT} KM  /  "
              f"{len(visible_shallow):03} EVENTS", color=SECONDARY, fontsize=7)
    axes.text(0.925, 0.188, "B / DEEP", color=WHITE,
              fontsize=8, weight="bold", ha="right")
    axes.text(0.925, 0.168, f"DEPTH >= {DEPTH_LIMIT} KM  /  "
              f"{len(visible_deep):03} EVENTS", color=SECONDARY,
              fontsize=7, ha="right")

    axes.plot([0.075, 0.925], [0.125, 0.125], color=SECONDARY,
              linewidth=0.35, alpha=0.45)
    axes.text(0.075, 0.097, "USGS / EARTHQUAKE OBSERVATION FIELD",
              color=WHITE, fontsize=7.5, weight="bold")
    axes.text(0.075, 0.072,
              "WIRE HEIGHT + DOT SIZE = SMOOTHED LOCAL EVENT COUNT",
              color=SECONDARY, fontsize=6.5)
    axes.text(0.925, 0.097, f"{len(visible):03} / {len(events):,}",
              color=WHITE, fontsize=18, weight="bold", ha="right")
    axes.text(0.925, 0.072,
              "IN VIEW / IN SOURCE     LAYER GAP IS SCHEMATIC",
              color=SECONDARY, fontsize=6.5, ha="right")
    axes.add_patch(plt.Rectangle((0.075, 0.042), 0.040, 0.006,
                                 color=LIME, linewidth=0))
    axes.text(0.125, 0.042, "STRONGEST EVENT LOCATOR", color=LIME,
              fontsize=6.2, va="bottom")

    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=150, facecolor=BACKGROUND)
    print(f"saved {OUTPUT.relative_to(HERE)} from {len(events)} earthquakes")
    print(f"in view: {len(visible_shallow)} shallow / {len(visible_deep)} deep")
    plt.show()


if __name__ == "__main__":
    main()
