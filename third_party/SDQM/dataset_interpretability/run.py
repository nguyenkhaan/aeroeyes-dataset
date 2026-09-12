import argparse
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import entropy

MODEL_STORAGE_DIR = Path(
    os.getenv(
        "AEROEYES_MODEL_DIR",
        "/datastore/cndt_khanhnd/models/aeroeyes_model",
    )
)
ULTRALYTICS_DIR = MODEL_STORAGE_DIR / "ultralytics"
YOLO_CONFIG_DIR = ULTRALYTICS_DIR / "config"
YOLO_MODEL_PATH = Path(
    os.getenv(
        "SDQM_VINFO_MODEL_PATH",
        str(ULTRALYTICS_DIR / "weights" / "yolo11n.pt"),
    )
)
YOLO_RUNS_DIR = Path(
    os.getenv("YOLO_RUNS_DIR", str(ULTRALYTICS_DIR / "runs"))
)
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(YOLO_CONFIG_DIR)
os.environ["YOLO_WEIGHTS_DIR"] = str(YOLO_MODEL_PATH.parent)
os.environ["YOLO_RUNS_DIR"] = str(YOLO_RUNS_DIR)

from ultralytics import YOLO
from ultralytics.models.yolo.detect.rareplanes_val import RareplanesDetectionValidator
from ultralytics.models.yolo.detect.dimo_val import DIMODetectionValidator
from ultralytics.models.yolo.detect.wasabi_val import WASABIDetectionValidator


# Training should be synthetic images; Validation/Unseen set should be Real images

# Annotation file can be the same. Just two inputs incase they aren't

def get_v_info(train_annotation_file, validation_annotation_file, dataset="rareplanes", image_size=512):
    YOLO_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    YOLO_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the YOLO model
    model = YOLO(str(YOLO_MODEL_PATH))
    results = model.train(
        data=train_annotation_file,
        epochs=10,
        freeze=10,
        device=0,
        imgsz=image_size,
        project=str(YOLO_RUNS_DIR),
    )

    args = dict(
        model=str(YOLO_MODEL_PATH),
        data=validation_annotation_file,
        device=0,
        imgsz=image_size,
        project=str(YOLO_RUNS_DIR),
    )
    
    if dataset == "rareplanes":
        results2 = RareplanesDetectionValidator(args=args)
        results2()
    elif dataset == "dimo":
        results2 = DIMODetectionValidator(args=args)
        results2()
    elif dataset == "wasabi":
        results2 = WASABIDetectionValidator(args=args)
        results2()
    else:
        model2 = YOLO(str(YOLO_MODEL_PATH))
        results2 = model2.val(
            data=validation_annotation_file,
            device=0,
            imgsz=image_size,
            project=str(YOLO_RUNS_DIR),
        )
    
    # Calculate the entropy of the validation set
    conditional_iou = -1 * calculate_entropy(results.iou_stores)
    predictive_iou = -1 * calculate_entropy(results2.iou_stores)
    
    conditional_conf = -1 * calculate_entropy(results.conf_stores)
    predictive_conf = -1 * calculate_entropy(results2.conf_stores)
    
    conditional_fusion = -1 * calculate_entropy(results.iou_stores * results.conf_stores)
    predictive_fusion = -1 * calculate_entropy(results2.iou_stores * results2.conf_stores)
    
    v_info_iou = predictive_iou - conditional_iou
    v_info_conf = predictive_conf - conditional_conf
    v_info_fusion = predictive_fusion - conditional_fusion
    
    return conditional_iou, predictive_iou, v_info_iou, conditional_conf, predictive_conf, v_info_conf, conditional_fusion, predictive_fusion, v_info_fusion
    
    
def calculate_entropy(p_y_given_f_x):
    p_y_given_f_x[p_y_given_f_x == 0] = 10**-5
    entr = entropy(p_y_given_f_x, base=2)
    return entr

def main():
    parser = argparse.ArgumentParser(description="Calculate V-Info metric.")
    parser.add_argument(
        "--train_annotation_file", 
        type=str, 
        required=True, 
        help="Path to the training annotation file."
    )
    parser.add_argument(
        "--validation_annotation_file", 
        type=str, 
        required=True, 
        help="Path to the validation annotation file."
    )

    args = parser.parse_args()
    
    start_time = time.time()
    conditional_iou, predictive_iou, v_info_iou, conditional_conf, predictive_conf, v_info_conf, conditional_fusion, predictive_fusion, v_info_fusion = get_v_info(args.train_annotation_file, args.validation_annotation_file)
    end_time = time.time()
    
    print(f"conditional_iou: {conditional_iou}")
    print(f"predictive_iou: {predictive_iou}")
    print(f"v_info_iou: {v_info_iou}")
    
    print(f"conditional_conf: {conditional_conf}")
    print(f"predictive_conf: {predictive_conf}")
    print(f"v_info_conf: {v_info_conf}")
    
    print(f"conditional_fusion: {conditional_fusion}")
    print(f"predictive_fusion: {predictive_fusion}")
    print(f"v_info_fusion: {v_info_fusion}")
    
    print(f"Execution Time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
