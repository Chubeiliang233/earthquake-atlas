# Seismic Relief

![A two-layer wireframe view of earthquake density in the western Pacific](out/earthquake-atlas.png)

## The phenomenon

Earthquakes happen when accumulated stress is released along faults in the Earth's
crust. Their locations are not evenly distributed: many occur in long belts near
tectonic plate boundaries. I chose earthquakes because I wanted to see whether
their spatial pattern could become a field of lines rather than a conventional map
covered with markers. The western Pacific was especially useful because a large
number of events form a strong, curved structure across the region.

## The source

The raw file comes from the [USGS past-month feed for earthquakes of magnitude 2.5
or greater](https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.geojson).
It contains 2,015 GeoJSON features. Each feature represents one recorded earthquake
and includes longitude and latitude in degrees, depth in kilometres, magnitude, and
time in milliseconds since the Unix epoch. The downloaded reply is stored unchanged
in `data/usgs-earthquakes-2.5-month.geojson`, so the drawing runs without internet
access.

## What the picture shows

The picture focuses on 422 events within the western Pacific crop. Earthquakes are
counted on a geographic grid and smoothed into a density field. In the upper layer,
local density becomes the height of the white wire surface. The lower layer samples
the same field as a regular dot matrix, with larger dots in denser areas. The gap
between layers is only a visual device, and the raised surface is not real terrain.
The transformation hides exact coastlines, individual event times, depths and most
magnitudes. It also removes earthquakes outside the selected region and softens
small local differences through smoothing.

## Run it

```
uv run plot.py
```
