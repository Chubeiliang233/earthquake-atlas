# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///

"""Draw a two-layer seismic-density poster from the saved USGS data."""

import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data" / "usgs-earthquakes-2.5-month.geojson"
OUTPUT = HERE / "out" / "earthquake-atlas.png"

BACKGROUND = "#0b0d0b"
WHITE = "#ebeee8"
SECONDARY = "#9da69d"
LIME = "#b9ff27"
BOUNDS = (98, 178, -20, 60)


def read_events(path):
    """Return longitude, latitude, magnitude and UTC date for each event."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = []
    for feature in raw["features"]:
        coordinates = feature["geometry"]["coordinates"]
        magnitude = feature["properties"]["mag"]
        if magnitude is None or not coordinates:
            continue
        date = dt.datetime.fromtimestamp(
            feature["properties"]["time"] / 1000, dt.timezone.utc
        ).date()
        events.append((coordinates[0], coordinates[1], magnitude, date))
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
    """Count events and smooth them over roughly nine geographic degrees."""
    west, east, south, north = BOUNDS
    longitudes = np.linspace(west - 20, east + 20, 240)
    latitudes = np.linspace(south - 20, north + 20, 240)
    counts = np.zeros((len(latitudes), len(longitudes)))
    for longitude, latitude, _magnitude, _date in events:
        if (longitudes[0] <= longitude <= longitudes[-1]
                and latitudes[0] <= latitude <= latitudes[-1]):
            column = int(np.searchsorted(longitudes, longitude).clip(
                0, len(longitudes) - 1
            ))
            row = int(np.searchsorted(latitudes, latitude).clip(
                0, len(latitudes) - 1
            ))
            counts[row, column] += 1
    return longitudes, latitudes, gaussian_smooth(counts, radius=18)


def project(longitude, latitude, height):
    """Project longitude, latitude and density height onto the poster."""
    west, east, south, north = BOUNDS
    x = 2 * (np.asarray(longitude) - west) / (east - west) - 1
    y = 2 * (np.asarray(latitude) - south) / (north - south) - 1
    return 0.5 + 0.22 * (x + y), 0.52 + 0.11 * (y - x) + 0.27 * height


def main():
    events = read_events(DATA)
    visible = [
        event for event in events
        if BOUNDS[0] <= event[0] <= BOUNDS[1]
        and BOUNDS[2] <= event[1] <= BOUNDS[3]
    ]
    longitudes, latitudes, field = density_surface(events)

    longitude_grid, latitude_grid = np.meshgrid(
        np.linspace(BOUNDS[0], BOUNDS[1], 39),
        np.linspace(BOUNDS[2], BOUNDS[3], 39),
    )
    column_indices = np.searchsorted(longitudes, longitude_grid).clip(
        0, len(longitudes) - 1
    )
    row_indices = np.searchsorted(latitudes, latitude_grid).clip(
        0, len(latitudes) - 1
    )
    sampled = field[row_indices, column_indices]
    scale = float(np.quantile(sampled, 0.995))
    height = np.sqrt(np.clip(sampled / scale, 0, 1))

    figure = plt.figure(figsize=(10, 12), facecolor=BACKGROUND)
    axes = figure.add_axes([0, 0, 1, 1], facecolor=BACKGROUND)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_axis_off()

    # Lower layer: regular samples of the density field.
    for row in range(0, 39, 2):
        for column in range(0, 39, 2):
            strength = float(height[row, column])
            x, y = project(
                longitude_grid[row, column], latitude_grid[row, column], -0.35
            )
            axes.plot(
                x, y, ".", color=WHITE,
                markersize=0.5 + 3.0 * strength,
                alpha=0.17 + 0.58 * strength, zorder=1,
            )

    # Corner lines show that the dot matrix and wire surface share one grid.
    for longitude, latitude in [(98, -20), (178, -20), (178, 60), (98, 60)]:
        lower_x, lower_y = project(longitude, latitude, -0.35)
        upper_x, upper_y = project(longitude, latitude, 0)
        axes.plot(
            [lower_x, upper_x], [lower_y, upper_y],
            color=SECONDARY, linewidth=0.35, alpha=0.26, zorder=2,
        )

    # Upper layer: density becomes height. Thicker lines provide visual rhythm.
    for row in range(39):
        x, y = project(longitude_grid[row], latitude_grid[row], height[row])
        major = row % 5 == 0
        axes.plot(
            x, y, color=WHITE,
            linewidth=0.85 if major else 0.40,
            alpha=0.92 if major else 0.57, zorder=3,
        )
    for column in (0, 10, 20, 30, 38):
        x, y = project(
            longitude_grid[:, column], latitude_grid[:, column], height[:, column]
        )
        axes.plot(x, y, color=WHITE, linewidth=0.33, alpha=0.22, zorder=3)

    peak_row, peak_column = np.unravel_index(np.argmax(sampled), sampled.shape)
    peak_x, peak_y = project(
        longitude_grid[peak_row, peak_column],
        latitude_grid[peak_row, peak_column],
        height[peak_row, peak_column],
    )
    axes.scatter(
        [peak_x], [peak_y], s=32, facecolors=BACKGROUND,
        edgecolors=LIME, linewidths=0.9, zorder=5,
    )

    first = min(event[3] for event in events)
    last = max(event[3] for event in events)
    axes.text(0.075, 0.943, "SEISMIC / RELIEF", color=WHITE,
              fontsize=26, weight="bold")
    axes.text(0.075, 0.916, "WESTERN PACIFIC  ·  EARTHQUAKE DENSITY",
              color=SECONDARY, fontsize=8)
    axes.text(0.925, 0.943, "01 / DATA FIELD", color=SECONDARY,
              fontsize=7, ha="right")
    axes.text(0.925, 0.916, f"{first} — {last} UTC", color=SECONDARY,
              fontsize=7, ha="right")
    axes.plot([0.075, 0.925], [0.897, 0.897], color=SECONDARY,
              linewidth=0.4, alpha=0.4)

    axes.text(0.075, 0.105, "UPPER LAYER", color=WHITE,
              fontsize=8, weight="bold")
    axes.text(0.075, 0.084, "WIRE HEIGHT = LOCAL EVENT DENSITY",
              color=SECONDARY, fontsize=7)
    axes.text(0.075, 0.063, "LOWER LAYER", color=WHITE,
              fontsize=8, weight="bold")
    axes.text(0.075, 0.042, "DOT SIZE = SAME FIELD, SAMPLED ON A GRID",
              color=SECONDARY, fontsize=7)
    axes.text(0.925, 0.105, f"{len(visible):03}", color=WHITE,
              fontsize=26, weight="bold", ha="right")
    axes.text(0.925, 0.077, f"EVENTS IN VIEW  /  {len(events):,} IN SOURCE",
              color=SECONDARY, fontsize=7, ha="right")
    axes.text(0.925, 0.047,
              "LAYER GAP IS SCHEMATIC  /  HEIGHT IS NOT TERRAIN",
              color=LIME, fontsize=7, ha="right")

    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=150, facecolor=BACKGROUND)
    print(f"saved {OUTPUT.relative_to(HERE)} from {len(events)} earthquakes")
    plt.show()


if __name__ == "__main__":
    main()
