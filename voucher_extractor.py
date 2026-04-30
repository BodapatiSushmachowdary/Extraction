from __future__ import annotations
import argparse
import json
import os
import re
import sys
import cv2
from pathlib import Path
from typing import Optional


os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")


import numpy as np
from paddleocr import PaddleOCR


print("[ocr] Loading PaddleOCR...")
_PADDLE = PaddleOCR(
    lang="en",
    use_textline_orientation=False,
    device="cpu",
    enable_mkldnn=False,
)
print("[ocr] PaddleOCR loaded")

REGIONS = {
    "debit_account":   (0.16, 0.21, 0.44, 0.29),
    "rupees_amount":   (0.10, 0.56, 0.50, 0.69),
    "rs_box_amount":   (0.78, 0.39, 0.96, 0.50),
    "approved_by_sig": (0.04, 0.74, 0.22, 0.99),
    "main_signature":  (0.53, 0.74, 0.73, 0.99),
}

DEWARP_SIZE = (2000, 1000)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points as: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype="float32")


def detect_document(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Find the paper's four corners using brightness + saturation masking."""
    h, w = img_bgr.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, low_sat = cv2.threshold(hsv[..., 1], 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    paper = cv2.bitwise_and(bright, low_sat)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN,  np.ones((10, 10), np.uint8))

    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.15 * img_area:
        return None

    peri = cv2.arcLength(biggest, True)
    for eps_factor in (0.01, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(biggest, eps_factor * peri, True)
        if len(approx) == 4:
            return _order_corners(approx)

    rect = cv2.minAreaRect(biggest)
    return _order_corners(cv2.boxPoints(rect))


def warp_document(img_bgr: np.ndarray, corners: np.ndarray,
                  out_size: tuple[int, int] = DEWARP_SIZE) -> np.ndarray:
    """Apply perspective warp to flatten the paper."""
    out_w, out_h = out_size
    dst = np.array([[0, 0], [out_w - 1, 0],
                    [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img_bgr, M, (out_w, out_h))


def align_voucher(img_bgr: np.ndarray, debug_dir: Optional[Path] = None) -> np.ndarray:
    """Detect the paper and warp it flat. Falls back to the raw image on failure."""
    corners = detect_document(img_bgr)
    if corners is None:
        if debug_dir:
            print("  [align] paper corners NOT detected -- using raw image")
        return img_bgr

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        viz = img_bgr.copy()
        for x, y in corners:
            cv2.circle(viz, (int(x), int(y)), 18, (0, 255, 0), 4)
        cv2.imwrite(str(debug_dir / "01_corners_detected.png"), viz)

    warped = warp_document(img_bgr, corners)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "02_dewarped.png"), warped)
        print(f"  [align] dewarped to {warped.shape[1]} x {warped.shape[0]}")
    return warped


def _crop(img: np.ndarray, frac: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = frac
    return img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]



def _paddle_predict(img: np.ndarray) -> list[tuple[str, float]]:
    """Run PaddleOCR on an image, return [(text, confidence), ...]."""
    out: list[tuple[str, float]] = []
    try:
        result = _PADDLE.predict(img)
        for page in (result or []):
            if isinstance(page, dict):
                texts  = page.get("rec_texts",  [])
                scores = page.get("rec_scores", [])
                for t, s in zip(texts, scores):
                    out.append((str(t).strip(), float(s)))
            elif isinstance(page, list):
                for entry in page:
                    try:
                        text, score = entry[1]
                        out.append((str(text).strip(), float(score)))
                    except (IndexError, TypeError, ValueError):
                        continue
    except Exception as e:
        print(f"  [ocr] PaddleOCR failed: {type(e).__name__}: {e}")
    return out


def _ocr(crop: np.ndarray, scales: tuple[int, ...] = (2,)) -> list[tuple[str, float]]:
    """Run PaddleOCR on a crop at multiple upscales. Returns combined [(text, conf)]."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = []
    for scale in scales:
        up = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        results.extend(_paddle_predict(up))
    return results



def extract_account_number(img: np.ndarray) -> dict:
    crop = _crop(img, REGIONS["debit_account"])
    raw = _ocr(crop, scales=(2,))

    weighted: dict[str, float] = {}
    for text, conf in raw:
        digits = re.sub(r"\D", "", text)
        if 10 <= len(digits) <= 13:
            weighted[digits] = weighted.get(digits, 0.0) + conf

    if not weighted:
        return {"account_number": None, "all_candidates": []}

    ranked = sorted(weighted.items(), key=lambda x: -x[1])
    return {
        "account_number": ranked[0][0],
        "all_candidates": [(d, round(s, 2)) for d, s in ranked],
    }


def _parse_amounts(results: list[tuple[str, float]]) -> dict[str, float]:
    """Extract amount-like strings from OCR results, weighted by confidence."""
    weighted: dict[str, float] = {}
    for text, conf in results:
        # Decimal amounts (most reliable format) get a 1.5x bonus
        for m in re.finditer(r"\d{3,5}\.\d{2}", text):
            weighted[m.group()] = weighted.get(m.group(), 0.0) + conf * 1.5
        # Whole-number amounts
        for m in re.finditer(r"\b\d{3,5}\b", text):
            v = m.group()
            weighted[v] = weighted.get(v, 0.0) + conf
    return weighted


def extract_amount(img: np.ndarray) -> dict:
    """Read amount from the Rupees line; cross-check against the Rs. box."""
    rupees_raw = _ocr(_crop(img, REGIONS["rupees_amount"]), scales=(2, 4))
    rs_box_raw = _ocr(_crop(img, REGIONS["rs_box_amount"]), scales=(2, 4))

    rupees_weighted = _parse_amounts(rupees_raw)
    rs_box_weighted = _parse_amounts(rs_box_raw)

    def _to_int(s: str) -> Optional[int]:
        try:
            return int(float(s.replace(",", "")))
        except (ValueError, AttributeError):
            return None

    # Cross-validation: amounts that appear in BOTH fields get a 2.5x boost
    rs_box_ints = {_to_int(a) for a in rs_box_weighted if _to_int(a) is not None}
    combined: dict[str, float] = {}
    for amt, weight in rupees_weighted.items():
        score = weight
        if _to_int(amt) in rs_box_ints:
            score *= 2.5
        combined[amt] = score

    if not combined and rs_box_weighted:
        best = max(rs_box_weighted.items(), key=lambda x: x[1])[0]
        return {
            "amount": best,
            "rupees_field_candidates": [],
            "rs_box_candidates": [(a, round(s, 2)) for a, s in
                                  sorted(rs_box_weighted.items(), key=lambda x: -x[1])],
        }
    if not combined:
        return {"amount": None, "rupees_field_candidates": [], "rs_box_candidates": []}

    ranked = sorted(combined.items(), key=lambda x: -x[1])
    return {
        "amount": ranked[0][0],
        "rupees_field_candidates": [(a, round(s, 2)) for a, s in
                                    sorted(rupees_weighted.items(), key=lambda x: -x[1])],
        "rs_box_candidates":       [(a, round(s, 2)) for a, s in
                                    sorted(rs_box_weighted.items(), key=lambda x: -x[1])],
        "combined_ranking":        [(a, round(s, 2)) for a, s in ranked[:5]],
    }



def extract_signature(crop_bgr: np.ndarray) -> Optional[dict]:
    """Locate the signature using HSV blue mask, return a tight color crop."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([85, 20, 20]), np.array([145, 255, 240]))

    # Suppress the printed label area at the top
    label_zone = int(0.30 * crop_bgr.shape[0])
    blue_mask[:label_zone, :] = 0
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    blue_mask = cv2.dilate(blue_mask, np.ones((4, 4), np.uint8), iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)
    big = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= 30]
    if not big:
        return None

    keep = np.zeros_like(blue_mask)
    for i in big:
        keep[labels == i] = 255

    ys, xs = np.where(keep > 0)
    pad = 12
    h_crop, w_crop = crop_bgr.shape[:2]
    x1 = max(0, int(xs.min()) - pad); x2 = min(w_crop, int(xs.max()) + pad)
    y1 = max(0, int(ys.min()) - pad); y2 = min(h_crop, int(ys.max()) + pad)

    tight_color = crop_bgr[y1:y2, x1:x2].copy()
    ink_pixels  = int((keep[y1:y2, x1:x2] > 0).sum())

    return {
        "color":      tight_color,
        "ink_pixels": ink_pixels,
        "size":       tight_color.shape[:2],
    }


def extract_all_signatures(img: np.ndarray, out_dir: Path) -> dict:
    """Save the color crop for each signature."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ("approved_by_sig", "main_signature"):
        sig = extract_signature(_crop(img, REGIONS[name]))
        if sig is None:
            results[name] = {"detected": False}
            continue
        color_path = out_dir / f"{name}_color.png"
        cv2.imwrite(str(color_path), sig["color"])
        results[name] = {
            "detected":   True,
            "ink_pixels": sig["ink_pixels"],
            "size":       sig["size"],
            "file":       str(color_path),
        }
    return results



def extract_voucher(image_path: str, out_dir: str = "voucher_output",
                    debug: bool = False) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    debug_dir = (out / "debug") if debug else None

    print(f"Processing {image_path}  ({img.shape[1]} x {img.shape[0]} px)")
    print("-" * 60)

    aligned = align_voucher(img, debug_dir=debug_dir)
    cv2.imwrite(str(out / "aligned.png"), aligned)

    acct = extract_account_number(aligned)
    print(f"Account number: {acct['account_number']}")

    amt = extract_amount(aligned)
    print(f"Amount:         {amt['amount']}")

    sigs = extract_all_signatures(aligned, out / "signatures")
    for name, info in sigs.items():
        status = (f"detected ({info['ink_pixels']} ink pixels)"
                  if info.get("detected") else "NOT detected")
        print(f"Signature [{name}]: {status}")

    summary = {
        "input": image_path,
        "aligned_image": str(out / "aligned.png"),
        "account_number": acct["account_number"],
        "amount":         amt["amount"],
        "signatures":     sigs,
        "diagnostics": {
            "account_number_candidates": acct["all_candidates"],
            "rupees_candidates":         amt.get("rupees_field_candidates", []),
            "rs_box_candidates":         amt.get("rs_box_candidates", []),
            "amount_combined_ranking":   amt.get("combined_ranking", []),
        },
    }
    summary_path = out / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nFull report -> {summary_path}")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract account number, amount, and signatures from a voucher."
    )
    p.add_argument("image", help="Path to the voucher JPG/PNG")
    p.add_argument("--out", default="voucher_output", help="Output directory")
    p.add_argument("--debug", action="store_true",
                   help="Save intermediate alignment images for inspection")
    args = p.parse_args()
    extract_voucher(args.image, args.out, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())