import cv2
import numpy as np

def load_image(path):
     """Load an image from a file path."""
     image = cv2.imread(path)
     if image is None:
          raise ValueError(f"Image not found at path: {path}")
     return image
def find_paper_contour(image, debug=False):
     """Detect the largest 4-point contour (the OMR sheet boundary)."""
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     blur = cv2.GaussianBlur(gray, (5,5), 0)
     edged = cv2.Canny(blur, 50, 150)

     cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

     for c in cnts:
          peri = cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, 0.02 * peri, True)
          if len(approx) == 4:  # Found rectangle
               if debug:
                    print("Found 4-point contour for sheet.")
               return approx.reshape(4,2)

     return None

def order_points(pts):
     """Order points as top-left, top-right, bottom-right, bottom-left."""
     rect = np.zeros((4,2), dtype="float32")
     s = pts.sum(axis=1)
     rect[0] = pts[np.argmin(s)]
     rect[2] = pts[np.argmax(s)]

     diff = np.diff(pts, axis=1)
     rect[1] = pts[np.argmin(diff)]
     rect[3] = pts[np.argmax(diff)]
     return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0,0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def preprocess_sheet(path, debug=False):
    """Load image, detect sheet, apply perspective correction."""
    img = load_image(path)
    contour = find_paper_contour(img, debug=debug)

    if contour is None:
        if debug:
            print("No contour found. Returning original image.")
        return img

    warped = four_point_transform(img, contour)
    return warped

if __name__ == "__main__":

    sheet_path = "C:/Users/mahis/OneDrive/Desktop/a4397defd26be38b74d96a98a34689d4.jpg"  
    output_path = "warped_sheet.jpg"

    warped_img = preprocess_sheet(sheet_path, debug=True)
    cv2.imwrite(output_path, warped_img)
    print(f"Warped sheet saved at {output_path}")