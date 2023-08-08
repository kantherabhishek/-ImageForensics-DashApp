# ImageForensics-DashApp
This Python Dash application allows you to perform various image forensic analyses using different image processing functions. It utilizes histogram equalization, luminance gradient calculation, error level analysis, and perceptual hash for clone detection. The application also uses external libraries such as OpenCV, imageio, and dash to deliver an interactive and user-friendly experience.

# Click the image/video below to watch the example output.
[![Click to view the video](/image.jpg)](https://youtu.be/8AmAy0vozSY)

# How to Use the Application?
1. Upload an Image: Drag and drop or click to select an image for analysis.
2. Clone Detection Plot: Detect potential clone regions in the uploaded image by adjusting the tile and step sizes.
3. Error Level Analysis Plot: Visualize the error levels in the image using histogram equalization.
4. Noise Analysis Plot: Observe the noise in the image after histogram equalization.
5. Level Sweep Plot: Sweep through different levels and opacities to adjust image brightness and opacity.
6. Luminance Gradient Plot: Explore the luminance gradient of the image in 3D space.
7. Geo and Meta Tags: Retrieve geo and meta tags from the uploaded image.
8. JPEG Quantization Tables: Examine the Y and CbCr quantization tables used in the JPEG compression process.
9. New Feature: ORB (Oriented FAST and Rotated BRIEF) is a feature detection method used in computer vision to identify distinctive keypoints in an image. It combines the FAST keypoint detector and the BRIEF descriptor to provide a robust and efficient way to detect and describe features in images. ORB is particularly useful for tasks like object recognition, image stitching, and tracking, where identifying unique points of interest is crucial. The ORB feature detection plot showcases these keypoints overlaid on the original image, helping to highlight regions of interest that can be used for further analysis or applications.

# How to Run the Application?
Make sure you have all the required libraries installed. You can do this by running the following command:

<code>pip install dash dash-bootstrap-components plotly opencv-python dash-table dash_core_components dash_html_components imagehash numpy imageio pillow</code>

Copy and paste the provided code into a Python file (image_analysis.py).

  Run the Python file using the command:
<code>python image_analysis.py</code>

Access the application on your browser at the specified address (e.g., http://127.0.0.1:8050/).

# Additional Information

This application is designed for educational and research purposes and can be a valuable tool for image forensic analysis. It provides insights into various techniques used in digital image forensics and can be extended for further analysis and research.
