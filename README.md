# Shape and Color Detection System

A real-time computer vision application that detects, classifies, and interprets geometric shapes and colors from a video stream. The system identifies shapes (triangles, squares, circles) and their colors (red, green, blue), then classifies them based on predefined safety categories.

## Features

- Real-time shape detection and classification
- Color recognition using HSV color space
- Interactive parameter adjustment via trackbars
- Object classification system for safety categorization
- IP camera video stream support

## Requirements

```bash
pip install opencv-python numpy
```

## Usage

1. Configure your IP camera URL in the code:
```python
cap = cv2.VideoCapture('http://192.168.1.3:8080/video')
```

2. Run the application:
```bash
python shape_detector.py
```

3. Adjust detection parameters using the trackbars:
   - **Threshold 1**: Lower threshold for Canny edge detection
   - **Threshold 2**: Upper threshold for Canny edge detection
   - **Area**: Minimum contour area to filter out noise

4. Press 'q' to quit the application

## Classification System

The system categorizes detected objects into three classes:

| Shape | Color | Classification |
|-------|-------|----------------|
| Triangle | Red | Dangerous obstacle |
| Square | Blue | Boundary marker |
| Circle | Green | Safe zone |

## Technical Overview

### Core Methods

#### `get_limits(color)`

Calculates HSV color range boundaries for color detection.

**Purpose**: Converts a BGR color value into lower and upper HSV thresholds that can be used for color masking. This is essential because HSV color space is more robust for color detection than BGR under varying lighting conditions.

**Parameters**:
- `color`: BGR color value as a list [B, G, R]

**Returns**: Tuple of (lowerLimit, upperLimit) as numpy arrays

**Algorithm**:
1. Converts the input BGR color to HSV color space
2. Extracts the hue value from the converted color
3. Handles special cases for red color (hue wraps around at 180°):
   - If hue >= 165: Creates range near upper bound (165-180)
   - If hue <= 15: Creates range near lower bound (0-15)
   - Otherwise: Creates ±10 range around the hue value
4. Sets saturation (100-255) and value (100-255) ranges to ensure vibrant colors are detected

**Technical Details**: The function accounts for the circular nature of the HSV hue channel where red appears at both ends of the spectrum (0° and 180°). This prevents range overflow issues when detecting red objects.

---

#### `detect_color(hsv_image, contour)`

Identifies the dominant color within a detected contour.

**Purpose**: Determines which color (red, green, or blue) is most prevalent within the boundaries of a detected shape. This enables the system to classify objects based on their color characteristics.

**Parameters**:
- `hsv_image`: The frame converted to HSV color space
- `contour`: OpenCV contour representing the detected shape

**Returns**: String indicating the dominant color ('red', 'green', 'blue', or 'unknown')

**Algorithm**:
1. Creates a blank mask the same size as the input image
2. Draws the input contour filled with white (255) on the mask
3. Iterates through each color in the color dictionary:
   - Applies `cv2.inRange()` to create a color-specific mask
   - Performs bitwise AND between the contour mask and color mask
   - Counts non-zero pixels in the combined mask
4. Tracks which color has the maximum pixel count
5. Returns the color with the highest pixel presence

**Technical Details**: This method uses a voting mechanism where each color "votes" based on how many pixels within the contour match its HSV range. The color with the most votes wins, making the detection robust against noise and partial color matches.

---

### Main Processing Loop

The application processes video frames continuously with the following pipeline:

#### 1. Frame Acquisition and Preprocessing
```python
_, img = cap.read()
img = cv2.flip(img, 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
```

- Captures frame from IP camera stream
- Flips image horizontally for mirror effect
- Converts to HSV for color detection
- Converts to grayscale for edge detection
- Applies Gaussian blur to reduce noise

#### 2. Edge Detection
```python
imgcanny = cv2.Canny(gray, threshold1, threshold2)
imgDil = cv2.dilate(imgcanny, kernel, iterations=1)
```

**Canny Edge Detection**: Identifies edges in the grayscale image using two threshold values. The lower threshold catches weak edges while the upper threshold confirms strong edges. This creates a binary image where edges are white and background is black.

**Dilation**: Expands the detected edges using a 5x5 kernel. This operation connects nearby edge segments and fills small gaps, making contour detection more reliable. Dilation is particularly useful when shapes have slightly broken edges due to lighting or camera quality.

#### 3. Contour Detection and Analysis
```python
contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
```

**`cv2.findContours()`**: Extracts contours (continuous boundaries) from the dilated edge image.
- `cv2.RETR_EXTERNAL`: Retrieves only the outermost contours, ignoring nested shapes
- `cv2.CHAIN_APPROX_NONE`: Stores all contour points without compression

#### 4. Shape Classification

For each detected contour that exceeds the minimum area threshold:

**Polygon Approximation**:
```python
approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
```

Uses the Douglas-Peucker algorithm to approximate the contour with fewer vertices. The `0.04` parameter (4% of perimeter) controls approximation accuracy - smaller values preserve more detail while larger values create simpler polygons.

**Shape Determination**:
- **4 vertices**: Distinguishes between square and rectangle by calculating width-to-height ratio
  - Ratio between 0.9 and 1.1 indicates a square
  - Other ratios indicate a rectangle
- **3 vertices**: Classified as a triangle
- **More vertices**: Calculates circularity using the formula: `4π × (area / perimeter²)`
  - Values > 0.7 indicate a circle
  - This metric approaches 1.0 for perfect circles and decreases for irregular shapes

#### 5. Object Classification and Visualization

```python
label = classification.get((shape, color))
```

Looks up the shape-color combination in the classification dictionary to determine the object's meaning (e.g., "Dangerous obstacle", "Safe zone").

**Centroid Calculation**:
```python
M = cv2.moments(cnt)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
```

Computes the geometric center of the shape using image moments. The moments `m10` and `m01` represent the first-order moments, while `m00` is the zeroth moment (area). This provides the average position of all points in the contour, which is more stable than using corner coordinates.

## Performance Considerations

- **Gaussian Blur**: Reduces high-frequency noise that can create false edges
- **Area Filtering**: Eliminates small contours that likely represent noise or artifacts
- **HSV Color Space**: Provides better color constancy under different lighting conditions compared to RGB/BGR
- **External Contour Retrieval**: Improves performance by ignoring internal contours within shapes

## Limitations and Improvements

**Current Limitations**:
- Limited to three colors (red, green, blue)
- Fixed classification rules
- Requires manual parameter tuning for different environments

**Potential Enhancements**:
- Add support for additional colors and shapes
- Implement adaptive thresholding for varying lighting
- Add shape orientation detection
- Include distance estimation using known object sizes
- Implement object tracking across frames

## License

MIT License

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.
