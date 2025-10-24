#!/usr/bin/env python3
import os, cv2, numpy as np

OUT_DIR = "data/aruco_markers"
os.makedirs(OUT_DIR, exist_ok=True)

# Use the small, easy-to-detect dictionary
DICT_NAME = cv2.aruco.DICT_4X4_50
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_NAME)

# Generate 5 markers (IDs 0..4), 200x200 px each for printing
for marker_id in range(5):
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
    path = os.path.join(OUT_DIR, f"aruco_4x4_50_id{marker_id}.png")
    cv2.imwrite(path, marker)
    print("[OK] saved", path)

# (Optional) also make a simple A4/Letter PDF-like sheet image to print once
# 2 columns x 3 rows layout, each with a 15 px white margin
h, w = 842, 595  # "A4-ish" portrait in pixels (not exact DPI-critical)
sheet = 255 * np.ones((h, w), dtype=np.uint8)
cell_h, cell_w = h // 3, w // 2
ids_to_place = [0,1,2,3,4]
for i, mid in enumerate(ids_to_place):
    r, c = divmod(i, 2)
    marker = cv2.aruco.generateImageMarker(aruco_dict, mid, min(cell_h, cell_w) - 40)
    mh, mw = marker.shape
    y0 = r*cell_h + (cell_h - mh)//2
    x0 = c*cell_w + (cell_w - mw)//2
    sheet[y0:y0+mh, x0:x0+mw] = marker
    cv2.putText(sheet, f"ID {mid}", (x0, y0-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)

cv2.imwrite(os.path.join(OUT_DIR, "aruco_sheet.png"), sheet)
print("[OK] saved sheet", os.path.join(OUT_DIR, "aruco_sheet.png"))
#!/usr/bin/env python3
import os, cv2, numpy as np

OUT_DIR = "data/aruco_markers"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[INFO] Saving markers to {OUT_DIR}")

DICT_NAME = cv2.aruco.DICT_4X4_50
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_NAME)

for marker_id in range(5):
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
    path = os.path.join(OUT_DIR, f"aruco_4x4_50_id{marker_id}.png")
    cv2.imwrite(path, marker)
    print(f"[OK] Saved {path}")

# Optional sheet
h, w = 842, 595
sheet = 255 * np.ones((h, w), dtype=np.uint8)
cell_h, cell_w = h // 3, w // 2
ids_to_place = [0, 1, 2, 3, 4]
for i, mid in enumerate(ids_to_place):
    r, c = divmod(i, 2)
    marker = cv2.aruco.generateImageMarker(aruco_dict, mid, min(cell_h, cell_w) - 40)
    mh, mw = marker.shape
    y0 = r * cell_h + (cell_h - mh) // 2
    x0 = c * cell_w + (cell_w - mw) // 2
    sheet[y0:y0+mh, x0:x0+mw] = marker
    cv2.putText(sheet, f"ID {mid}", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)

sheet_path = os.path.join(OUT_DIR, "aruco_sheet.png")
cv2.imwrite(sheet_path, sheet)
print(f"[OK] Saved sheet to {sheet_path}")
