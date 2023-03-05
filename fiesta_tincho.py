import math
import random
import cv2
import numpy as np
import time

# virtual_object = cv2.imread('virtual_object.png', -1)

# # Define the dimensions of the virtual object
# virtual_object_width = 100
# virtual_object_height = 100

orb = cv2.ORB_create()

x = 50
y = 50
dx = 5
dy = 3
color = (0, 0, 0)

def shuffle_cols(frame):
    # Get the width of the frame
    h, w, _ = frame.shape
    # Define the width of each group
    group_width = np.random.randint(1, 6)
    # Calculate the number of groups
    num_groups = w // group_width
    # Randomly shuffle the groups
    group_order = np.random.permutation(num_groups)
    # Create a list of group indices
    group_indices = [i * group_width for i in range(num_groups)]
    # Add the remaining columns as a last group
    group_indices.append(w)
    # Shuffle the columns within each group
    for i in range(num_groups):
        start = group_indices[group_order[i]]
        end = group_indices[group_order[i] + 1]
        col_index = np.random.permutation(end - start) + start
        frame[:, start:end, :] = frame[:, col_index, :]
    return frame

def borrar_caras(frame, gray):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        # Get the face region
        face_region = frame[y:y+h, x:x+w]
        
        # Apply the blur effect on the face region
        face_region = cv2.GaussianBlur(face_region, (23, 23), 30)
        
        # Put the blurred face region back into the frame
        frame[y:y+h, x:x+w] = face_region
    return frame

def pixeles_repetidos_horizontal(frame, pixel_art):
    # Crop the frame to the left half
    left_half = frame[:, :half_width]
    
    # Resize the left half to match the dimensions of the pixel_art array
    block = cv2.resize(left_half, (half_width, height))
    
    # Shift the pixels in the pixel_art array to the right
    pixel_art[:, 1:] = pixel_art[:, :-1]
    
    # Replace the first column of pixels with the new block
    pixel_art[:, 0] = block[:, -1]
    
    # Display the combined output
    screen = np.hstack([left_half, pixel_art])
    return screen, pixel_art

def pixeles_repetidos_horizontal(frame, pixel_art):
    # Crop the frame to the left half
    left_half = frame[:, :half_width]
    
    # Resize the left half to match the dimensions of the pixel_art array
    block = cv2.resize(left_half, (half_width, height))
    
    # Shift the pixels in the pixel_art array to the right
    pixel_art[:, 1:] = pixel_art[:, :-1]
    
    # Replace the first column of pixels with the new block
    pixel_art[:, 0] = block[:, -1]
    
    # Display the combined output
    screen = np.hstack([left_half, pixel_art])
    return screen, pixel_art

def pixeles_repetidos_vertical(frame, pixel_art):
    height, width, _ = frame.shape
    half_height = height // 2
    
    
        # Crop the frame to the top half
    top_half = frame[:half_height, :]
    
    # Resize the top half to match the dimensions of the pixel_art array
    block = cv2.resize(top_half, (width, half_height))
    
    # Shift the pixels in the pixel_art array up by one row
    pixel_art[1:, :] = pixel_art[:-1, :]
    
    # Replace the bottom row of pixels with the new block
    pixel_art[0,:] = block[-1,:]
    
    # Display the combined output
    screen = np.vstack([top_half, pixel_art])
    return screen, pixel_art

def add_text(frame):
    
    global x, y, dx, dy, color
    cv2.putText(frame, "@juanma.tuki", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    x += dx
    y += dy

    height, width, _ = frame.shape

    if x < 0 or x + 200 > width:
        dx = -dx
        color = (0, 0, 0) if color == (255, 255, 255) else (255, 255, 255)
    if y < 0 or y + 1 > height:
        dy = -dy
        color = (255, 255, 255) if color == (0, 0, 0) else (0, 0, 0)
        
    return frame
 
def detectar_caras(frame, gray):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

def mirror_randomly(frame):
    time.sleep(0.3)
    mirror_v = random.choice([True, False, False, False, True])
    mirror_h = random.choice([True, False, False, True, True])

    if mirror_v:
        frame = cv2.flip(frame, 0)

    if mirror_h:
        frame = cv2.flip(frame, 1)

    return frame

def juanma_tuki(frame):
    if time.time() % 3 < 1:
        # Apply a blue color filter
        # frame[:, :, 0] = 255
        frame[:, :, 1] %=2
        frame[:, :, 2] %=2
    elif time.time() % 3 <2:
        # Apply a green color filter
        frame[:, :, 0] %=2
        # frame[:, :, 1] = 255
        frame[:, :, 2] %=2

    else:
        # Apply a green color filter
        frame[:, :, 0] %=2
        # frame[:, :, 1] = 255
        frame[:, :, 1] %=2
        
        
    frame = add_text(frame)
    return frame

def scale_frame(frame, width=None, height=None):
    # Get the dimensions of the frame
    h, w = frame.shape[:2]

    # If width or height is not provided, use the original dimensions
    if width is None and height is None:
        return frame

    if width is None:
        # Calculate the scaling ratio based on the desired height
        ratio = height / h
        # Calculate the new width based on the ratio
        width = int(w * ratio)
    elif height is None:
        # Calculate the scaling ratio based on the desired width
        ratio = width / w
        # Calculate the new height based on the ratio
        height = int(h * ratio)

    # Resize the frame to the new dimensions using bilinear interpolation
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a VideoCapture object to read from the webcam
cap = cv2.VideoCapture(1)

# Set the window size
cv2.namedWindow("Cool Video Effect", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cool Video Effect", 640, 480)

ret, frame = cap.read()
height, width, _ = frame.shape
half_height = height // 2
half_width = width // 2
pixel_artv = np.zeros((half_height, width, 3), dtype=np.uint8)
pixel_arth = np.zeros((height, half_width, 3), dtype=np.uint8)

# Set the initial time
start_time = cv2.getTickCount()
# Set the interval time (5 minutes = 5*60*cv2.getTickFrequency())
interval_time = 0.5*60*cv2.getTickFrequency()

# Set the initial function to be called
current_function = juanma_tuki

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    # frame = add_text(frame)
    # frame = mirror_randomly(frame)
    # frame = juanma_tuki(frame)
    # frame, pixel_arth = pixeles_repetidos_horizontal(frame, pixel_arth)
    # frame, pixel_artv = pixeles_repetidos_vertical(frame, pixel_artv)
    
    if cv2.getTickCount() - start_time > interval_time:
        # Switch to the other function
        if current_function == borrar_caras:
            current_function = pixeles_repetidos_horizontal
        elif current_function == pixeles_repetidos_horizontal:
            current_function = detectar_caras
        elif current_function == detectar_caras:
            current_function = pixeles_repetidos_vertical
        elif current_function == pixeles_repetidos_vertical:
            current_function = juanma_tuki
        elif current_function == juanma_tuki:
            current_function = mirror_randomly
        elif current_function == mirror_randomly:
            current_function = shuffle_cols
        elif current_function == shuffle_cols:
            current_function = borrar_caras  

        # Update the start time
        start_time = cv2.getTickCount()
    
    # Call the current function
    if current_function == juanma_tuki:
        frame=current_function(frame)
    elif current_function == pixeles_repetidos_horizontal:
        frame, pixel_arth = current_function(frame, pixel_arth)
    elif current_function == detectar_caras:    
        frame=detectar_caras(frame, gray)
    elif current_function == pixeles_repetidos_vertical:
        frame, pixel_artv = current_function(frame, pixel_artv)
    elif current_function == mirror_randomly:
        frame = mirror_randomly(frame)
    elif current_function == juanma_tuki:
        frame = juanma_tuki(frame)
    elif current_function == borrar_caras:
        frame = borrar_caras(frame, gray)
    elif current_function == shuffle_cols:
        frame = shuffle_cols(frame)
    # Apply the cool effect to the frame
    # frame = detectar_caras(frame, gray)
    cv2.imshow("Cool Video Effect", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()