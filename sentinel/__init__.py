"""SENTINEL — cascaded small-VLM video anomaly detection for the AHC / FlytBase hackathon."""

__version__ = "0.1.0"

CLASSES: tuple[str, ...] = (
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "fire",
    "smoke",
    "waterlogging_or_flood",
    "wrong_way_driving",
    "road_spill_or_debris",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
)
NORMAL = "normal"

# Temporal shape families (drive the decoder; see PLAN.md §1.3)
SHAPE: dict[str, str] = {
    "traffic_accident": "impulse",
    "fighting_or_violence": "impulse",
    "traffic_congestion": "ramp",
    "waterlogging_or_flood": "ramp",
    "smoke": "ramp",
    "fire": "ramp",
    "stalled_or_broken_down_vehicle": "dwell",
    "vehicle_blocking_traffic": "dwell",
    "loitering_or_suspicious_presence": "dwell",
    "wrong_way_driving": "trajectory",
    "road_spill_or_debris": "static",
}
