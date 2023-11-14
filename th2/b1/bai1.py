import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load an image from file."""
    return cv2.imread(image_path)

def display_image(image, title="Image"):
    """Display an image."""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_shapes(image_path):
    """Detect shapes in an image using contours."""
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        cv2.drawContours(image, [approx], 0, (0, 0, 255), 5)

    display_image(image, "Shapes Detected")

def draw_geometry():
    """Draw basic geometric shapes."""
    image = np.zeros((500, 500, 3), dtype=np.uint8)

    # Draw a line
    cv2.line(image, (100, 100), (400, 400), (0, 255, 0), 2)

    # Draw a rectangle
    cv2.rectangle(image, (150, 150), (350, 350), (255, 0, 0), 2)

    # Draw a circle
    cv2.circle(image, (250, 250), 50, (0, 0, 255), 2)

    display_image(image, "Basic Geometry")

def plot_graph():
    """Plot a simple graph using matplotlib."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.plot(x, y)
    plt.title("Sine Wave")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

# Example usage
image_path = "123.png"
detect_shapes(image_path)
draw_geometry()
plot_graph()