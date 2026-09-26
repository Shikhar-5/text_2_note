"""
align_sample.py -- STEP 1: layout-constrained alignment of the known calibration text.

Uses the KNOWN structure of the sheet (12 rows, known number of units per row) to
find rows and units. No thresholds tuned per image, no template matching yet.
The 62 sample_04 glyphs are NOT used here -- they come in at the character-alignment
step, after these row/unit boxes are correct.

Usage (run from the Text2Note folder):
    python align_sample.py sample_02

Reads : handwriting_dataset/<sample>/03_no_lines.png   (white ink on black)
        handwriting_dataset/<sample>/01_grayscale.png  (only used to draw overlays)
Writes: handwriting_dataset/<sample>/alignment/
            debug_0_clean.png   ink mask after artifact removal (unchanged from v1)
            debug_1_roi.png     NEW: header/margin strip vs. handwriting content ROI
            debug_2_rows.png    12 row bands, now confined to the ROI
            debug_3_units.png   one box per expected unit (pair / digit / word)
            alignment.json      boxes + confidences

CHANGES vs. the previous version (see chat for full rationale):
  1. find_content_roi(): NEW. Strips the page header (Date/Page No, etc.) and
     surrounding margin before row detection runs, instead of treating the whole
     page as the 12-row body. debug_1_roi.png shows exactly what was stripped.
  2. find_row_bands(): now takes the ROI's (top, bot) instead of recomputing them
     from the whole image, so the 11 row cuts are only searched for *inside* the
     handwriting content.
  3. row_unit_boxes(): REPLACES split_units(). Instead of cutting at the widest
     empty *column* gaps (which broke on Aa/Bb pairs whose upper/lower glyphs sit
     at different heights, and on residual line noise), this clusters each row's
     connected components by x-center into exactly n_units groups (1-D weighted
     k-means, weight = ink area, k = the known unit count for that row). This
     naturally keeps an "Aa" pair's two vertically-offset glyphs together (same
     x-center), and is robust to a stray leftover artifact (it's just an
     under-weighted point pulled into whichever cluster is nearest).
  4. For single-word rows (6-11, n_units == 1) there is no clustering at all:
     row_unit_boxes() just returns one box around all filtered ink in the row,
     as requested -- no reason to run split logic when there's only one answer.
  5. clean_mask() and the skew diagnostic are UNCHANGED (confirmed acceptable).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DATASET = Path(__file__).resolve().parent / "handwriting_dataset"

# ---------------------------------------------------------------- known layout
UP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PAIRS = [u + u.lower() for u in UP]
LAYOUT = [                                   # each row = list of expected units
    PAIRS[0:10],
    PAIRS[10:20],
    PAIRS[20:26],
    list("0123456789"),
    "The quick brown fox jumps over the".split(),
    ["lazy", "dog"],
    ["Resistance"],
    ["resistance"],
    ["RESISTENCE"],                          # spelled exactly as written on the sheet
    ["Experimental"],
    ["apparatus"],
    ["galvanometer"],
]


# ---------------------------------------------------------------- helpers
def smooth(v, sigma):
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(v, k / k.sum(), mode="same")


def load(sample):
    d = DATASET / sample
    gray = cv2.imread(str(d / "01_grayscale.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(d / "03_no_lines.png"), cv2.IMREAD_GRAYSCALE)
    if gray is None or mask is None:
        raise SystemExit(f"Missing 01_grayscale.png or 03_no_lines.png in {d}")
    if gray.shape != mask.shape:
        raise SystemExit(f"Size mismatch: gray {gray.shape} vs mask {mask.shape}")
    return d, gray, (mask > 127).astype(np.uint8)


def clean_mask(ink):
    """Drop specks and obvious notebook artifacts (long thin lines, big border pieces).
    UNCHANGED from v1 -- confirmed acceptable on the real sample."""
    H, W = ink.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(20, int(H * W * 2e-6))
    ok = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        long_h = w > 0.35 * W and h < 0.012 * H
        long_v = h > 0.35 * H and w < 0.012 * W
        touches = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
        border_big = touches and (w > 0.2 * W or h > 0.2 * H)
        ok[i] = not (a < min_area or long_h or long_v or border_big)
    print(f"  components: {n-1} found, {int((~ok[1:]).sum())} removed as artifacts")
    return ok[lab].astype(np.uint8)


def estimate_skew(ink):
    """Diagnostic only: angle (deg) at which row profile is sharpest. ~0 is good."""
    small = cv2.resize(ink * 255, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best = (-1.0, 0.0)
    for ang in np.arange(-4, 4.01, 0.25):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(ang), 1.0)
        r = cv2.warpAffine(small, M, (w, h))
        score = float(np.var(r.sum(axis=1).astype(np.float64)))
        if score > best[0]:
            best = (score, float(ang))
    return best[1]


# ---------------------------------------------------------------- NEW: content ROI
def find_content_roi(ink, n_rows):
    """
    Strip header/margin above the handwriting body.

    Finds all vertical ink "blocks" (bridging tiny inter-letter dips). Since we
    KNOW the body has exactly n_rows row-blocks (a row's profile essentially
    never has a full-width empty gap running through the middle of it), any
    blocks beyond the last n_rows are header/margin content sitting above the
    body -- so if there are more than n_rows blocks total, the leading
    `len(blocks) - n_rows` of them are stripped, whatever their individual gap
    sizes. This is a count-based rule, not a gap-size threshold, so it doesn't
    need tuning per page. A gap-size check runs only as a non-blocking sanity
    warning. Returns (roi_top, roi_bot, header_bottom_or_None, blocks).
    """
    H, _ = ink.shape
    s = smooth(ink.sum(axis=1).astype(np.float64), max(2.0, H * 0.003))
    smax = s.max()
    eps = 0.015 * smax
    active = s > eps
    ys = np.where(active)[0]
    if len(ys) == 0:
        raise SystemExit("No ink found at all -- check debug_0_clean.png")
    top, bot = int(ys[0]), int(ys[-1])

    small_merge = max(2, int(0.01 * H))
    d = np.diff(np.concatenate([[0], active[top:bot + 1].astype(int), [0]]))
    starts = np.where(d == 1)[0] + top
    ends = np.where(d == -1)[0] + top
    raw_blocks = list(zip(starts.tolist(), ends.tolist()))
    blocks = [list(raw_blocks[0])]
    for b in raw_blocks[1:]:
        if b[0] - blocks[-1][1] <= small_merge:
            blocks[-1][1] = b[1]
        else:
            blocks.append(list(b))
    blocks = [tuple(b) for b in blocks]
    print(f"  ROI: {len(blocks)} ink block(s) found above/within the body "
          f"(need {n_rows} for the body alone)")

    extra = len(blocks) - n_rows
    if extra <= 0:
        return top, bot, None, blocks

    header_bottom = blocks[extra - 1][1]
    roi_top = blocks[extra][0]
    gap_after = roi_top - header_bottom
    later_gaps = [blocks[i + 1][0] - blocks[i][1] for i in range(extra, len(blocks) - 1)]
    if later_gaps and gap_after < np.median(later_gaps):
        print(f"  ROI WARNING: stripped {extra} leading block(s) as header, but the gap "
              f"after them ({gap_after}px) isn't larger than typical row gaps "
              f"({np.median(later_gaps):.0f}px) -- check debug_1_roi.png carefully")
    return roi_top, bot, header_bottom, blocks


# ---------------------------------------------------------------- rows (within ROI)
def find_row_bands(ink, roi_top, roi_bot, n_rows):
    """Choose exactly n_rows-1 cut lines at the emptiest horizontal valleys,
    searched only inside [roi_top, roi_bot]."""
    H, _ = ink.shape
    s = smooth(ink.sum(axis=1).astype(np.float64), max(2.0, H * 0.003))
    smax = float(s[roi_top:roi_bot + 1].max())
    top, bot = roi_top, roi_bot
    pitch = (bot - top) / n_rows
    lo, hi = top + int(0.3 * pitch), bot - int(0.3 * pitch)
    sep = int(0.5 * pitch)

    cost = s.copy()
    cost[:lo] = np.inf
    cost[hi + 1:] = np.inf
    cuts = []
    for _ in range(n_rows - 1):
        y = int(np.argmin(cost))
        if not np.isfinite(cost[y]):
            break
        tol = s[y] + 0.02 * smax
        a, b = y, y
        while a > lo and s[a - 1] <= tol:
            a -= 1
        while b < hi and s[b + 1] <= tol:
            b += 1
        c = (a + b) // 2                       # centre of the empty plateau
        cuts.append(c)
        cost[max(0, c - sep): c + sep + 1] = np.inf
    if len(cuts) != n_rows - 1:
        raise SystemExit(f"Only found {len(cuts)} valleys, need {n_rows-1}. Check debug_1_roi.png")
    cuts.sort()

    typical = float(np.median(s[top:bot + 1][s[top:bot + 1] > 0.02 * smax]))
    cut_ratio = [float(s[c] / typical) for c in cuts]
    return cuts, cut_ratio, s


# ---------------------------------------------------------------- NEW: unit clustering
def _kmeans_1d_weighted(x, w, k, iters=50):
    """Small weighted 1-D k-means (weight = component ink area). Init centroids
    evenly spaced across the *spatial* x-range (min..max), not across sorted
    point indices -- units in a row are roughly evenly spaced across the row's
    width, but the number of ink components per unit varies (a pair whose
    upper/lower glyph touch is 1 component, one where they don't is 2), so an
    index-based init can seed two centroids inside the same dense unit and
    none inside a sparser one. A spatial init avoids that regardless of how
    components happen to be distributed. k is tiny and known, so no need for
    k-means++ here."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    centroids = np.linspace(x.min(), x.max(), k)
    assign = np.zeros(len(x), dtype=int)
    for it in range(iters):
        d = np.abs(x[:, None] - centroids[None, :])
        new_assign = d.argmin(axis=1)
        if it > 0 and np.array_equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for c in range(k):
            m = assign == c
            if m.any():
                centroids[c] = np.average(x[m], weights=w[m])
    return assign, centroids


def row_unit_boxes(row_ink, n_units):
    """
    Split one row's ink into n_units unit boxes using the KNOWN unit count.

    n_units == 1 (rows 6-11): no clustering -- just the union bbox of every
    filtered component in the row, per your note that split logic is pointless
    here.

    n_units > 1 (pairs / digits / words): cluster component x-centers into
    n_units groups with weighted 1-D k-means (weight = ink area), then union
    each cluster's components into one box. This keeps an "Aa" pair's upper and
    lower glyphs together even though they sit at different heights, because
    they share an x-center -- no column-gap logic involved.

    Returns (boxes, confidence). boxes has exactly n_units entries in
    left-to-right order; an entry is None if that unit couldn't be recovered.
    """
    H, W = row_ink.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(row_ink, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.0008 * W * H:                      # local speck filter
            continue
        if w > 0.5 * W and h < 0.05 * H:             # local leftover ruled-line filter
            continue
        comps.append((int(x), int(y), int(w), int(h), int(a)))

    if not comps:
        return [None] * n_units, 0.0

    if n_units == 1:
        x0 = min(c[0] for c in comps)
        y0 = min(c[1] for c in comps)
        x1 = max(c[0] + c[2] for c in comps)
        y1 = max(c[1] + c[3] for c in comps)
        return [(x0, y0, x1, y1)], 1.0

    if len(comps) < n_units:
        return [None] * n_units, 0.0

    xs = np.array([c[0] + c[2] / 2 for c in comps])
    areas = np.array([c[4] for c in comps], dtype=np.float64)
    assign, centroids = _kmeans_1d_weighted(xs, areas, n_units)
    order = np.argsort(centroids)                    # left-to-right == expected order
    remap = {int(old): new for new, old in enumerate(order)}

    boxes = [None] * n_units
    for k in range(n_units):
        idxs = [j for j, a in enumerate(assign) if remap[int(a)] == k]
        if not idxs:
            continue
        x0 = min(comps[j][0] for j in idxs)
        y0 = min(comps[j][1] for j in idxs)
        x1 = max(comps[j][0] + comps[j][2] for j in idxs)
        y1 = max(comps[j][1] + comps[j][3] for j in idxs)
        boxes[k] = (x0, y0, x1, y1)

    if any(b is None for b in boxes):
        conf = 0.0
    else:
        gaps = [boxes[k + 1][0] - boxes[k][2] for k in range(n_units - 1)]
        conf = float(np.mean([1.0 if g >= 0 else 0.0 for g in gaps]))
    return boxes, conf


# ---------------------------------------------------------------- drawing
def label(img, text, org, color, scale, thick):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def save_small(path, img, max_w=1800):
    if img.shape[1] > max_w:
        f = max_w / img.shape[1]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------- main
def main(sample):
    d, gray, raw = load(sample)
    out = d / "alignment"
    out.mkdir(exist_ok=True)
    H, W = gray.shape
    print(f"{sample}: {W}x{H}")

    ink = clean_mask(raw)
    cv2.imwrite(str(out / "debug_0_clean.png"), ink * 255)
    print(f"  skew diagnostic: {estimate_skew(ink):+.2f} deg (want |x| < ~0.5)")

    n_rows = len(LAYOUT)
    roi_top, roi_bot, header_bottom, blocks = find_content_roi(ink, n_rows)
    if header_bottom is not None:
        print(f"  ROI: stripped header/margin rows 0-{header_bottom} ({header_bottom}px)")
    else:
        print("  ROI: no header detected above the body")
    print(f"  ROI: content body is rows {roi_top}-{roi_bot} ({roi_bot-roi_top}px)")

    roi_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = roi_img.copy()
    if header_bottom is not None:
        cv2.rectangle(overlay, (0, 0), (W, header_bottom), (0, 0, 255), -1)
        roi_img = cv2.addWeighted(overlay, 0.25, roi_img, 0.75, 0)
    cv2.rectangle(roi_img, (0, roi_top), (W, roi_bot), (0, 200, 0), 3)
    label(roi_img, "stripped (header/margin)", (10, 30), (0, 0, 255), 0.8, 2)
    label(roi_img, "content ROI", (10, roi_top + 30), (0, 200, 0), 0.8, 2)
    save_small(out / "debug_1_roi.png", roi_img)

    cuts, cut_ratio, prof = find_row_bands(ink, roi_top, roi_bot, n_rows)
    edges = [roi_top] + cuts + [roi_bot]

    sc = max(1.0, W / 1600)
    th = max(1, int(round(2 * sc)))
    fs = 0.7 * sc
    rows_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    units_img = rows_img.copy()
    result = {"sample": sample, "image_size": [W, H],
              "roi": {"top": roi_top, "bot": roi_bot, "header_bottom": header_bottom},
              "rows": []}

    for r, expected in enumerate(LAYOUT):
        y0, y1 = edges[r], edges[r + 1]
        band = ink[y0:y1]
        ys, xs = np.where(band > 0)
        if len(xs) == 0:
            print(f"  row {r}: EMPTY band")
            result["rows"].append({"row": r, "expected": expected, "bbox": None,
                                    "split_confidence": 0.0, "units": []})
            continue
        bx0, bx1, by0, by1 = int(xs.min()), int(xs.max() + 1), int(ys.min() + y0), int(ys.max() + 1 + y0)

        boxes, conf = row_unit_boxes(band, len(expected))
        n_missing = sum(1 for b in boxes if b is None)
        flag = ""
        if n_missing:
            flag = f"  <-- {n_missing} UNIT(S) MISSING"
        elif conf < 1.0:
            flag = "  <-- OVERLAP BETWEEN UNITS"
        if r > 0 and cut_ratio[r - 1] > 0.25:
            flag += "  <-- WEAK ROW CUT ABOVE"
        print(f"  row {r:2d}: {len(expected):2d} units, cluster conf {conf:.2f}, "
              f"height {by1-by0:4d}px{flag}")

        cv2.rectangle(rows_img, (bx0, by0), (bx1, by1), (0, 200, 0), th)
        label(rows_img, f"r{r}: {' '.join(expected)}", (bx0, max(15, by0 - 6)), (0, 255, 255), fs, th)

        row_rec = {"row": r, "expected": expected, "bbox": [bx0, by0, bx1, by1],
                   "split_confidence": conf, "units": []}
        for text, box in zip(expected, boxes):
            if box is None:
                row_rec["units"].append({"text": text, "bbox": None})
                continue
            u0, uy0, u1, uy1 = box
            uy0 += y0
            uy1 += y0
            color = (0, 200, 0) if conf >= 1.0 else (0, 0, 255)
            cv2.rectangle(units_img, (u0, uy0), (u1, uy1), color, th)
            label(units_img, text, (u0, max(15, uy0 - 4)), (0, 255, 255), fs * 0.8, th)
            row_rec["units"].append({"text": text, "bbox": [u0, uy0, u1, uy1]})
        result["rows"].append(row_rec)

    for c in cuts:
        cv2.line(rows_img, (0, c), (W, c), (255, 200, 0), 1)
    cv2.rectangle(rows_img, (0, roi_top), (W, roi_bot), (255, 0, 255), 1)

    pw = int(0.15 * W)
    panel = np.full((H, pw, 3), 255, np.uint8)
    pts = np.stack([(prof / prof.max() * (pw - 4)).astype(np.int32) + 2, np.arange(H)], axis=1)
    cv2.polylines(panel, [pts.reshape(-1, 1, 2)], False, (0, 0, 0), max(1, th - 1))
    for c in cuts:
        cv2.line(panel, (0, c), (pw, c), (255, 120, 0), max(1, th - 1))
    cv2.line(panel, (0, roi_top), (pw, roi_top), (255, 0, 255), max(1, th - 1))
    cv2.line(panel, (0, roi_bot), (pw, roi_bot), (255, 0, 255), max(1, th - 1))
    rows_img = np.hstack([rows_img, panel])

    save_small(out / "debug_2_rows.png", rows_img)
    save_small(out / "debug_3_units.png", units_img)
    (out / "alignment.json").write_text(json.dumps(result, indent=1))
    print(f"\nWrote debug images + alignment.json to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample_02")