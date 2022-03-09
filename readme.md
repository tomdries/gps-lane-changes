Code and partial data repository of the [AHFE 2022 paper]( https://www.researchgate.net/publication/358621757_Identifying_lane_changes_automatically_using_the_GPS_sensors_of_portable_devices) *Identifying lane changes automatically using the GPS sensors of portable devices.* by Tom Driessen, Lokin Prasad, Pavlo Bazilinskyy, and Joost de Winter

All tabular data is stored in the current repository. Video recordings can be found in the [4TU research repository](https://doi.org/10.4121/19170302). 


# Data
See `/data`, which contains
1. The GPS recordings, with synchronized time index (`t` column), coordinates of the nearest matched road coordinate, and the lateral distance projected on the road segment
2. The manually annoted time stamps of lane changes. Time stamps are annotated for the start, middle, and end of a lane change. Start and end are the moments when the vehicle starts or stops sliding sideways, as visually identified from the video. The middle is the time the car is centered over the line between the two lanes. In case of a double lane change, the middle is when the car is approximately on the center of the middle lane.

3. The matched trajectories for each ride, representing the roads geometry. These were matched using [Mapbox Map Matching API](https://docs.mapbox.com/help/glossary/map-matching-api/). 

# Code
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomdries/gps-lane-changes/blob/main/analysis.ipynb)

The analysis can be found in analysis.ipynb and can be opened in google Colab.
