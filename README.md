# Projector T-Shirt Machine 🎨👕🏆

This is my project that won 2nd place prize at [Durhack 2024](https://durhack.com/) and here is my [devpost](https://devpost.com/software/projector-t-shirt-machine?ref_content=user-portfolio&ref_feature=in_progress). Projector T-Shirt Machine turns your own T-Shirt into a canvas. It uses OpenCV and MediaPipe for detecting hand gestures that allows you to paint your design on a virtual board, save it, and then project it directly onto yourself via laptop or projector, all while maintaining dynamic scaling as you move closer or further from the projector.

![Start Screen](./assets/1.png)

### How It Works

1. **The Start Screen (start.py)**: 

   When running the project, run `start.py`, which will display an abstract looking canvas of a start screen created using Pygame. The canvas start screen features splashes of paint in random spaces of the screen in various colours and a bunch of lines/walls inspired by [`Kandinsky's Jaune Rounge Bleu painting`](./assets/Kandinsky_-_Jaune_Rouge_Bleu.jpeg). A projector is drawn using squares and circles with the title written on top. The mouse acts as a dynamic light source casting rays using Bresenham's line algorithm to create a sun light effect and the start button takes you to the projector t-shirt machine.
   
2. **Hand Tracking(handTracking.py)**:
   The `handTracking.py` is used to capture your hand movements in real-time, using MediaPipe and OpenCV landmark detection to find the x and y coordinates for each finger joint through processing every frame. The getPosition method gets the 2D-pixel coordinates of each finger joint and the getUpFingers returns a list of booleans that indicate which fingers are raised based on the relative position of the joints to determine gestures such as fingers raised. This is used to turn your hands into a paintbrush and for navigation.
   
![Virtual Board](./assets/2.jpg)
   
3. **Virtual Canvas and Overlay (project.py)**:
   When you launch project.py through start.py, your webcam opens and displays an array of buttons on your camera window. These buttons allow you to select colours, change brush size, clear the canvas, toggle the drawing board, and save your creation. Buttons can be selected by raising a finger, with handTracker.py tracking your finger movements on the window.

   When you select the board button by raising your finger, the visibility of the board is toggled. The button, controlled by boardBtn, switches between "Board" and "Hide." When the board is visible, whiteBoard.drawReact(frame) renders it over the main frame, allowing you to draw on it.

   Raising your finger to select the colour palette button opens a colour menu, including an Eraser and Clear option. The menu, controlled by colorsBtn, toggles between "Colours" and "Hide" based on whether it’s open or hidden. When the index finger is extended (raised) and no other fingers are up, upFingers from handTracker.py detects this and activates a drawing mode. In drawing mode, cv2.line draws a line from the previous finger position (px, py), creating a continuous drawing effect as you move your finger. The RGB and alpha values adjust based on the selected colour. Selecting the Eraser sets the colour to black, which effectively removes parts of the drawing rather than adding to it.

   To change the brush size, penBtn displays available pen sizes, and selecting one updates brushSize. To indicate which colour or brush size is selected, the button’s transparency (alpha) is adjusted, making it clear what has been clicked.

   When you click saveBtn, the current drawing is saved as an image, board_drawing.png, in the [`tshirtsdesigns folder`](./tshirtsdesigns), with a transparent background applied using add_transparency. A copy of the image is also saved in the [`gallery folder`](./tshirtsdesigns/gallery), allowing it to be overlaid on the chest in other applications.

![Virtual Board Drawing](./assets/3.jpg)
![Virtual Board Drawing](./assets/6.jpg)

4. **Pose Detection with Overlay (project.py)**:
   After saving your art, PoseOverlay is initialised to project your art onto your T-Shirt by converting "board_drawing.png" into an RGB format using imgRGB which is required of MediaPipe and calculates the chest's central position and width based on landmark[11](left shoulder) and landmark[12](right shoulder) using "find_chest_area". If no landmarks of the chest are detected it returns none, none values for error handling and once identified, overlay_on_chest positions the image at the chest centre and scales it to match your chest width while adjusting its height for natural fit and transparency. Before applying the overlay, the method checks whether the new position fits within the image boundaries so you can move around and the projection will be accurate. A home button is also on this window so return home using left click.

![Drawing Projection via Laptop](./assets/4.jpg)

5. **Overlay Projection (project.py)**   
   Another window opens to project your art onto yourself in real life via a projector by using "display_overlay_only" method which projects the overlay in a separate, resizable window. This method calculates the size and position of the overlay dynamically based on the chest width relative to the primary window’s dimensions by defining the maximum and minimum projection sizes as proportions of the primary window width and uses linear scaling to adjust the overlay size between these bounds using "projection_mapping". The overlay is resized to this projection size and positioned according to the chest centre, with its movement limited by a specified factor, "overlay_movement_factor". 

   When projecting use "Extended Mode", drag the Overlay Projection window onto the extended side for clear image projection onto the T-Shirt. Also, unlike myself in this image, use darker lighting for a better effect.

![Drawing Projection via Projector](./assets/7.jpg)
![Drawing Projection via Projector](./assets/5.jpg)

### Installation Requirements

This project requires the following Python packages with specific versions. You can install them using `pip`:

```bash
pip install mediapipe==0.10.14 opencv-python==4.10.0 pygame==2.5.2 numpy==1.24.2

run start.py

position Overlay Projection Window on Extended Mode using Windows P and Drag it onto the extended side and use Darker Lighting
```

#### Package Details

- **OpenCV** (4.10.0): For image processing and video capture.
- **NumPy** (1.24.2): Essential for array computing.
- **MediaPipe** (0.10.14): For hand-tracking and other computer vision tasks.
- **Pygame** (2.5.2): For rendering graphics and handling user input.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jes4l/projector-t-shirt-machine
   ```
