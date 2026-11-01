from __future__ import annotations


def suppress_ambulance_conflicts(detections: list, threshold: float | None) -> tuple[list, int]:
    if threshold is None:
        return list(detections), 0
    vehicles = [item for item in detections if item.label == "emergency_vehicle"]
    retained = [
        item for item in detections
        if item.label != "ambulance"
        or not any(item.box.iou(vehicle.box) >= threshold for vehicle in vehicles)
    ]
    return retained, len(detections) - len(retained)
