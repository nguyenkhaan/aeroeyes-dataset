from __future__ import annotations


COLORS = {"ambulance": (0, 220, 255), "emergency_vehicle": (255, 190, 0), "ignore": (220, 100, 255)}


def _draw_box(draw, box, text: str, color: tuple, offset: tuple[int, int]) -> None:
    x, y = offset
    coords = (x + box[0], y + box[1], x + box[2], y + box[3])
    draw.rectangle(coords, outline=color, width=3)
    draw.text((coords[0] + 2, max(y, coords[1] - 13)), text, fill=color,
              stroke_width=2, stroke_fill=(0, 0, 0))


def _gt_boxes(draw, objects: list, offset: tuple[int, int], ignored: bool = False, lost_ids=()) -> None:
    for item in objects:
        label = item["class_name"]
        if label not in ("ambulance", "emergency_vehicle"):
            continue
        suffix = " IGNORE" if ignored else " LOST" if item["object_id"] in lost_ids else ""
        color = COLORS["ignore"] if ignored else COLORS[label]
        _draw_box(draw, item["bbox_xyxy"], f"{label}{suffix}", color, offset)


def _prediction_boxes(draw, report: dict, offset: tuple[int, int]) -> None:
    for item in report["matching"]["refined_rescue_objects"]:
        label = item["label"]
        if label not in ("ambulance", "emergency_vehicle"):
            continue
        box = item["box"]
        coords = [box[key] for key in ("x_min", "y_min", "x_max", "y_max")]
        _draw_box(draw, coords, f"{label} {item['confidence']:.2f}", COLORS[label], offset)


def render_comparison(case: dict, ground_truth: dict, reports: tuple[dict, dict], paths: tuple, output) -> None:
    from PIL import Image, ImageDraw

    with Image.open(paths[0]) as source, Image.open(paths[1]) as generated:
        before, after = source.convert("RGB"), generated.convert("RGB")
    if before.size != after.size:
        raise ValueError("Before and After image sizes differ")
    width, height = before.size
    top = 76
    canvas = Image.new("RGB", (width * 4, height + top + 16), (18, 25, 36))
    for index, image in enumerate((before, after, after, after)):
        canvas.paste(image, (index * width, top))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f"Pair {case['source_pair_id']} | GT ambulances lost: {len(case['lost_ambulance_gt_ids'])} | baseline overlapping FP: {case['baseline_fp_overlap_emergency']}", fill="white")
    draw.text((8, 24), "Cyan: ambulance | Orange: emergency_vehicle | Purple: uncertain GT (ignore)", fill="white")
    for index, title in enumerate(("Before: GT", "After: GT", "After: baseline predictions", "After: candidate predictions")):
        draw.text((index * width + 8, 54), title, fill="white")
    _gt_boxes(draw, ground_truth["before_objects"], (0, top))
    _gt_boxes(draw, ground_truth["after_objects"], (width, top), lost_ids=case["lost_ambulance_gt_ids"])
    _gt_boxes(draw, ground_truth["ignore_regions"], (width, top), ignored=True)
    _prediction_boxes(draw, reports[0], (width * 2, top))
    _prediction_boxes(draw, reports[1], (width * 3, top))
    canvas.save(output, format="PNG")
