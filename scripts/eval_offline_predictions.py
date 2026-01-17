#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Evaluate existing COCO prediction JSONs with SAM3 offline evaluator.

Applies score thresholding and max-dets-per-image filtering before evaluation.
"""

import json
import os
import csv
from collections import defaultdict
from typing import Dict, List

from sam3.eval.coco_eval_offline import CocoEvaluatorOfflineWithPredFileEvaluators


def load_predictions(pred_path: str) -> List[Dict]:
    with open(pred_path, "r") as file:
        return json.load(file)


def filter_predictions(
    predictions: List[Dict], score_thresh: float, max_dets: int
) -> List[Dict]:
    if not predictions:
        return []

    preds_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for pred in predictions:
        if pred.get("score", 0.0) >= score_thresh:
            preds_by_image[pred["image_id"]].append(pred)

    filtered = []
    for image_id, preds in preds_by_image.items():
        preds_sorted = sorted(preds, key=lambda p: p.get("score", 0.0), reverse=True)
        if max_dets > 0:
            preds_sorted = preds_sorted[:max_dets]
        filtered.extend(preds_sorted)

    return filtered


def evaluate_prediction_file(
    pred_path: str,
    gt_path: str,
    iou_type: str,
    score_thresh: float,
    max_dets: int,
) -> Dict[str, float]:
    raw_preds = load_predictions(pred_path)
    filtered_preds = filter_predictions(raw_preds, score_thresh, max_dets)

    tmp_pred_path = pred_path + ".filtered.tmp.json"
    with open(tmp_pred_path, "w") as file:
        json.dump(filtered_preds, file)

    try:
        evaluator = CocoEvaluatorOfflineWithPredFileEvaluators(
            gt_path=gt_path, tide=False, iou_type=iou_type
        )
        return evaluator.evaluate(tmp_pred_path)
    finally:
        if os.path.exists(tmp_pred_path):
            os.remove(tmp_pred_path)


def find_prediction_files(directory: str, suffix: str) -> List[str]:
    pred_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(suffix):
                pred_files.append(os.path.join(root, filename))
    return sorted(pred_files)

def main() -> None:

    pred_path = '/jhcnas1/houjunlin/fxx/sam3_results/15_01_26/pathocell_ft/dumps/roboflow/pathocell/coco_predictions_bbox.json'
    gt_path = '/jhcnas1/houjunlin/fxx/sam3_datasets/rf100-vl/pathocell/test/_annotations.coco.json'
    iou_type = "bbox" # ["bbox", "segm"],
    max_dets = 200
    score_thresh = 0.5

    results = evaluate_prediction_file(
        pred_path=pred_path,
        gt_path=gt_path,
        iou_type=iou_type,
        score_thresh=score_thresh,
        max_dets=max_dets,
    )

    with open('results.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(results.keys())
        writer.writerow(results.values())


if __name__ == "__main__":
    main()
