import cv2
import numpy as np
import time
import os
import mediapipe as mp
import argparse
import math

def main():
    """
    Advanced application to detect fingers and recognize gestures through webcam
    using MediaPipe Hands with special visual effects and virtual marker functionality
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced Finger Detection App")
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--video', type=str, help='Path to video file (instead of camera)')
    parser.add_argument('--max-hands', type=int, default=2, help='Maximum number of hands to detect')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5, help='Minimum detection confidence')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5, help='Minimum tracking confidence')
    parser.add_argument('--effects', action='store_true', help='Enable special effects')
    
    args = parser.parse_args()
    
    # Use video file if provided, otherwise use camera
    source = args.video if args.video else args.camera
    
    # Open camera/video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Print camera properties for debugging
    print(f"Camera resolution: {width}x{height}")
    print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Initialize canvas for drawing
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create window with a specific position
    window_name = 'Advanced Finger Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 100, 100)  # Position the window at x=100, y=100
    cv2.resizeWindow(window_name, width, height)  # Set window size to match video
    print(f"Window created: {window_name} at position (100, 100) with size {width}x{height}")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Configure MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    
    # Finger tip indices (for each finger)
    finger_tip_indices = [
        mp_hands.HandLandmark.THUMB_TIP,    # Thumb
        mp_hands.HandLandmark.INDEX_FINGER_TIP,  # Index
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,  # Middle
        mp_hands.HandLandmark.RING_FINGER_TIP,  # Ring
        mp_hands.HandLandmark.PINKY_TIP  # Pinky
    ]
    
    # Finger knuckle indices (for checking if finger is extended)
    finger_mcp_indices = [
        mp_hands.HandLandmark.THUMB_MCP,  # Knuckle of thumb
        mp_hands.HandLandmark.INDEX_FINGER_MCP,  # Knuckle of index
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,  # Knuckle of middle 
        mp_hands.HandLandmark.RING_FINGER_MCP,  # Knuckle of ring
        mp_hands.HandLandmark.PINKY_MCP  # Knuckle of pinky
    ]
    
    # Finger second joint indices
    finger_pip_indices = [
        mp_hands.HandLandmark.THUMB_IP,  # IP joint of thumb
        mp_hands.HandLandmark.INDEX_FINGER_PIP,  # PIP joint of index
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,  # PIP joint of middle
        mp_hands.HandLandmark.RING_FINGER_PIP,  # PIP joint of ring
        mp_hands.HandLandmark.PINKY_PIP  # PIP joint of pinky
    ]
    
    # Finger names
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    
    # Colors for different fingers (in BGR)
    finger_colors = [
        (0, 0, 255),    # Red for thumb
        (0, 255, 0),    # Green for index
        (255, 0, 0),    # Blue for middle
        (0, 255, 255),  # Yellow for ring
        (255, 0, 255)   # Magenta for pinky
    ]
    
    # Drawing parameters
    drawing_color = (0, 0, 255)  # Initial color: Red
    drawing_thickness = 5
    drawing_colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 255) # White
    ]
    current_color_index = 0
    
    # Pinch detection parameters
    is_pinched = False
    pinch_threshold = 0.05  # Distance threshold for pinch detection
    prev_pinch_point = None
    
    # Smoothing filter for drawing
    class SmoothingFilter:
        def __init__(self, alpha=0.5):
            self.alpha = alpha
            self.prev_point = None
            
        def update(self, point):
            if self.prev_point is None:
                self.prev_point = point
                return point
            
            # Apply exponential smoothing
            smoothed_x = self.alpha * point[0] + (1 - self.alpha) * self.prev_point[0]
            smoothed_y = self.alpha * point[1] + (1 - self.alpha) * self.prev_point[1]
            
            self.prev_point = (smoothed_x, smoothed_y)
            return self.prev_point
    
    # Initialize smoothing filter
    smoothing_filter = SmoothingFilter(alpha=0.3)
    
    # Particle system for special effects
    class ParticleSystem:
        def __init__(self, max_particles=100):
            self.particles = []
            self.max_particles = max_particles
            
        def add_particle(self, x, y, color, velocity=(0, 0), size=5, lifetime=20):
            if len(self.particles) < self.max_particles:
                # Add random variation to velocity
                vx = velocity[0] + np.random.uniform(-1, 1)
                vy = velocity[1] + np.random.uniform(-1, 1)
                
                self.particles.append({
                    'x': x,
                    'y': y,
                    'color': color,
                    'velocity': (vx, vy),
                    'size': size,
                    'lifetime': lifetime,
                    'remaining': lifetime
                })
                
        def update(self):
            # Update particles
            for particle in self.particles:
                # Apply velocity
                particle['x'] += particle['velocity'][0]
                particle['y'] += particle['velocity'][1]
                
                # Add gravity
                particle['velocity'] = (
                    particle['velocity'][0],
                    particle['velocity'][1] + 0.1
                )
                
                # Decrease lifetime
                particle['remaining'] -= 1
                
                # Decrease size
                particle['size'] *= 0.95
            
            # Remove dead particles
            self.particles = [p for p in self.particles if p['remaining'] > 0]
            
        def draw(self, frame):
            for particle in self.particles:
                # Calculate alpha based on remaining lifetime
                alpha = particle['remaining'] / particle['lifetime']
                
                # Get color with alpha
                color = particle['color']
                
                # Draw particle
                cv2.circle(
                    frame, 
                    (int(particle['x']), int(particle['y'])), 
                    int(particle['size']), 
                    color, 
                    -1
                )
    
    # Initialize particle system
    particles = ParticleSystem(max_particles=500)
    
    # Initialize gesture history (for smoothing)
    gesture_history = []
    max_history = 10
    
    # Gesture recognition parameters
    gesture_names = {
        "UNKNOWN": "Unknown",
        "FIST": "Fist",
        "OPEN_HAND": "Open Hand",
        "POINTING": "Pointing",
        "VICTORY": "Victory",
        "THUMBS_UP": "Thumbs Up",
        "PINCH": "Pinch"
    }
    
    current_gesture = "UNKNOWN"
    
    # Function to recognize hand gesture
    def recognize_gesture(landmarks, handedness):
        # Count extended fingers
        extended_fingers = []
        
        # Special case for thumb - compare x-coordinates based on handedness
        thumb_tip = landmarks.landmark[finger_tip_indices[0]]
        thumb_base = landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]  # Base of thumb
        
        # Thumb is extended if its tip is more to the side than its base (depends on handedness)
        if (handedness == "Right" and thumb_tip.x < thumb_base.x) or \
           (handedness == "Left" and thumb_tip.x > thumb_base.x):
            extended_fingers.append(0)  # Thumb is extended
        
        # For other fingers, compare y-coordinates
        for i in range(1, 5):  # Index, middle, ring, pinky
            tip = landmarks.landmark[finger_tip_indices[i]]
            pip = landmarks.landmark[finger_pip_indices[i]]  # Second joint
            
            # Finger is extended if tip is higher (smaller y) than second joint
            if tip.y < pip.y:
                extended_fingers.append(i)
        
        # Distance between thumb and index tips for pinch detection
        thumb_tip = landmarks.landmark[finger_tip_indices[0]]
        index_tip = landmarks.landmark[finger_tip_indices[1]]
        pinch_distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 + 
            (thumb_tip.z - index_tip.z)**2
        )
        
        # Recognize gestures based on extended fingers
        if len(extended_fingers) == 0:
            return "FIST"
        elif len(extended_fingers) == 5:
            return "OPEN_HAND"
        elif len(extended_fingers) == 1 and 1 in extended_fingers:
            return "POINTING"
        elif len(extended_fingers) == 2 and 1 in extended_fingers and 2 in extended_fingers:
            return "VICTORY"
        elif len(extended_fingers) == 1 and 0 in extended_fingers:
            return "THUMBS_UP"
        elif pinch_distance < 0.05:  # Threshold for pinch detection
            return "PINCH"
        else:
            return "UNKNOWN"
    
    # FPS tracking
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # UI control
    show_fps = True
    show_landmarks = True
    show_effects = args.effects
    show_gesture = True
    show_canvas = True
    
    print("Starting advanced finger detection...")
    print("Controls:")
    print("- Press 'q' or ESC to exit")
    print("- Press 'f' to toggle FPS display")
    print("- Press 'l' to toggle landmarks display")
    print("- Press 'e' to toggle special effects")
    print("- Press 'g' to toggle gesture recognition display")
    print("- Press 'c' to clear drawing canvas")
    print("- Press 'd' to toggle canvas drawing")
    print("- Press 'p' to cycle through drawing colors")
    print("- Press '+' to increase line thickness")
    print("- Press '-' to decrease line thickness")
    
    # Gesture tracking
    last_gesture_time = time.time()
    gesture_cooldown = 0.5  # seconds
    
    # Processing mode (0=normal, 1=edge, 2=binary)
    process_mode = 0
    
    # Pinch state debouncing
    pinch_state_buffer = []
    pinch_buffer_size = 5  # Number of frames to keep in the buffer
    
    # Main loop
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from video source")
            break
        
        # Print frame shape for debugging (first frame only)
        if frame_count == 0:
            print(f"Frame shape: {frame.shape}")
        
        # Flip the image horizontally for a more intuitive selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Apply image processing based on mode
        if process_mode == 1:
            # Edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            processed = cv2.Canny(blurred, 50, 150)
            # Convert back to color for drawing
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            # Combine with original
            frame = cv2.addWeighted(frame, 0.7, processed, 0.3, 0)
        elif process_mode == 2:
            # Binary threshold
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, processed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # Convert back to color for drawing
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            # Combine with original
            frame = cv2.addWeighted(frame, 0.5, processed, 0.5, 0)
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Convert to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)
        
        # Variable to track if pinch was detected in this frame
        current_frame_pinch = False
        current_pinch_point = None
        
        # Draw hand landmarks and handle gestures
        if results.multi_hand_landmarks:
            hand_index = 0
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks if enabled
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Determine handedness (left or right)
                handedness = "Left"
                if results.multi_handedness and len(results.multi_handedness) > hand_index:
                    handedness = results.multi_handedness[hand_index].classification[0].label
                
                # Recognize gesture
                gesture = recognize_gesture(hand_landmarks, handedness)
                
                # Check for pinch gesture specifically for drawing
                thumb_tip = hand_landmarks.landmark[finger_tip_indices[0]]
                index_tip = hand_landmarks.landmark[finger_tip_indices[1]]
                
                # Calculate distance between thumb and index finger tips
                pinch_distance = math.sqrt(
                    (thumb_tip.x - index_tip.x)**2 + 
                    (thumb_tip.y - index_tip.y)**2 + 
                    (thumb_tip.z - index_tip.z)**2
                )
                
                # Calculate midpoint between thumb and index tips (pinch point)
                pinch_x = int((thumb_tip.x + index_tip.x) / 2 * display_frame.shape[1])
                pinch_y = int((thumb_tip.y + index_tip.y) / 2 * display_frame.shape[0])
                pinch_point = (pinch_x, pinch_y)
                
                # Apply smoothing filter to the pinch point
                smoothed_point = smoothing_filter.update(pinch_point)
                
                # Detect pinch gesture based on distance
                if pinch_distance < pinch_threshold:
                    current_frame_pinch = True
                    current_pinch_point = smoothed_point
                    
                    # Draw a circle at the pinch point to indicate active drawing
                    # Ensure the point is a tuple of integers
                    circle_center = (int(smoothed_point[0]), int(smoothed_point[1]))
                    cv2.circle(display_frame, circle_center, 10, (0, 255, 255), -1)
                
                # Update gesture history
                gesture_history.append(gesture)
                if len(gesture_history) > max_history:
                    gesture_history.pop(0)
                
                # Use most common gesture from history for stability
                current_gesture = max(set(gesture_history), key=gesture_history.count)
                
                # Draw gesture text if enabled
                if show_gesture:
                    # Get wrist position for text placement
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wrist_x = int(wrist.x * display_frame.shape[1])
                    wrist_y = int(wrist.y * display_frame.shape[0])
                    
                    # Display gesture name
                    cv2.putText(
                        display_frame,
                        f"{gesture_names[current_gesture]}",
                        (wrist_x - 40, wrist_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )
                
                # Apply special effects based on gesture if enabled
                if show_effects:
                    # Get fingertip positions for particle effects
                    for i, tip_idx in enumerate(finger_tip_indices):
                        tip = hand_landmarks.landmark[tip_idx]
                        tip_x = int(tip.x * display_frame.shape[1])
                        tip_y = int(tip.y * display_frame.shape[0])
                        
                        # FIST gesture - no particles
                        if current_gesture == "FIST":
                            pass
                        
                        # OPEN_HAND gesture - particles from all fingertips
                        elif current_gesture == "OPEN_HAND":
                            particles.add_particle(
                                tip_x, 
                                tip_y, 
                                finger_colors[i],
                                velocity=(np.random.uniform(-2, 2), np.random.uniform(-5, -2)),
                                size=8,
                                lifetime=20
                            )
                        
                        # POINTING gesture - beam from index finger
                        elif current_gesture == "POINTING" and i == 1:  # Index finger
                            # Draw a beam/line from finger
                            knuckle = hand_landmarks.landmark[finger_mcp_indices[i]]
                            direction_x = tip.x - knuckle.x
                            direction_y = tip.y - knuckle.y
                            
                            # Normalize direction
                            length = math.sqrt(direction_x**2 + direction_y**2)
                            if length > 0:
                                direction_x /= length
                                direction_y /= length
                            
                            # Draw laser line
                            end_x = int(tip_x + direction_x * 500)
                            end_y = int(tip_y + direction_y * 500)
                            cv2.line(display_frame, (tip_x, tip_y), (end_x, end_y), (0, 255, 0), 5)
                            
                            # Add particles along the beam
                            for j in range(10):
                                particles.add_particle(
                                    tip_x + direction_x * j * 10,
                                    tip_y + direction_y * j * 10,
                                    (0, 255, 0),
                                    velocity=(direction_x * 2, direction_y * 2),
                                    size=5,
                                    lifetime=5
                                )
                        
                        # VICTORY gesture - particles from index and middle fingers
                        elif current_gesture == "VICTORY" and (i == 1 or i == 2):
                            particles.add_particle(
                                tip_x, 
                                tip_y, 
                                finger_colors[i],
                                velocity=(np.random.uniform(-3, 3), np.random.uniform(-5, -1)),
                                size=10,
                                lifetime=30
                            )
                        
                        # THUMBS_UP gesture - upward particles from thumb
                        elif current_gesture == "THUMBS_UP" and i == 0:
                            for _ in range(3):
                                particles.add_particle(
                                    tip_x + np.random.uniform(-10, 10), 
                                    tip_y + np.random.uniform(-10, 10), 
                                    (0, 0, 255),
                                    velocity=(np.random.uniform(-1, 1), np.random.uniform(-8, -4)),
                                    size=8,
                                    lifetime=25
                                )
                        
                        # PINCH gesture - burst of particles between thumb and index
                        elif current_gesture == "PINCH" and (i == 0 or i == 1):
                            thumb_tip = hand_landmarks.landmark[finger_tip_indices[0]]
                            index_tip = hand_landmarks.landmark[finger_tip_indices[1]]
                            
                            # Calculate midpoint
                            mid_x = int((thumb_tip.x + index_tip.x) / 2 * display_frame.shape[1])
                            mid_y = int((thumb_tip.y + index_tip.y) / 2 * display_frame.shape[0])
                            
                            # Add burst of particles at midpoint
                            if i == 0:  # Only do this once (for thumb)
                                for _ in range(5):
                                    particles.add_particle(
                                        mid_x, 
                                        mid_y, 
                                        (255, 255, 0),
                                        velocity=(np.random.uniform(-4, 4), np.random.uniform(-4, 4)),
                                        size=6,
                                        lifetime=15
                                    )
                
                # Increment hand index
                hand_index += 1
        
        # Update pinch state buffer
        pinch_state_buffer.append(current_frame_pinch)
        if len(pinch_state_buffer) > pinch_buffer_size:
            pinch_state_buffer.pop(0)
        
        # Debounce pinch state (majority voting)
        num_pinched = sum(pinch_state_buffer)
        new_is_pinched = num_pinched > (pinch_buffer_size / 2)
        
        # Drawing logic for virtual marker
        if show_canvas:
            # Start a new line if transitioning into pinched state
            if new_is_pinched and not is_pinched:
                prev_pinch_point = current_pinch_point
            
            # Draw line while pinched
            elif new_is_pinched and is_pinched and prev_pinch_point is not None and current_pinch_point is not None:
                # Ensure points are tuples of integers
                start_point = (int(prev_pinch_point[0]), int(prev_pinch_point[1]))
                end_point = (int(current_pinch_point[0]), int(current_pinch_point[1]))
                cv2.line(canvas, start_point, end_point, drawing_color, drawing_thickness)
                prev_pinch_point = current_pinch_point
        
        # Update pinch state
        is_pinched = new_is_pinched
        
        # Overlay the canvas on the display frame
        if show_canvas:
            # Blend canvas with display frame
            display_frame = cv2.addWeighted(display_frame, 1.0, canvas, 0.7, 0)
        
        # Update and draw particles
        if show_effects:
            particles.update()
            particles.draw(display_frame)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = end_time
        
        # Display information overlay
        if show_fps:
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (display_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # Display current drawing color
        cv2.circle(
            display_frame,
            (width - 30, height - 30),
            15,
            drawing_color,
            -1
        )
        
        # Display current thickness
        cv2.putText(
            display_frame,
            f"Thickness: {drawing_thickness}",
            (width - 150, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Display mode info
        mode_text = ["Normal", "Edge", "Binary"][process_mode]
        cv2.putText(
            display_frame,
            f"Mode: {mode_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Display controls info
        cv2.putText(
            display_frame,
            "Press 'm' to change mode | 'c' to clear canvas",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Show the frame
        try:
            cv2.imshow(window_name, display_frame)
        except Exception as e:
            print(f"Error displaying frame: {e}")
            break
        
        # Handle key presses
        try:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q
                print("Exit requested by user")
                break
            elif key == ord('f'):  # Toggle FPS display
                show_fps = not show_fps
            elif key == ord('l'):  # Toggle landmarks display
                show_landmarks = not show_landmarks
            elif key == ord('e'):  # Toggle special effects
                show_effects = not show_effects
            elif key == ord('g'):  # Toggle gesture recognition display
                show_gesture = not show_gesture
            elif key == ord('m'):  # Change processing mode
                process_mode = (process_mode + 1) % 3
            elif key == ord('c'):  # Clear canvas
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
            elif key == ord('d'):  # Toggle canvas drawing
                show_canvas = not show_canvas
            elif key == ord('p'):  # Cycle through drawing colors
                current_color_index = (current_color_index + 1) % len(drawing_colors)
                drawing_color = drawing_colors[current_color_index]
            elif key == ord('+') or key == ord('='):  # Increase line thickness
                drawing_thickness = min(30, drawing_thickness + 1)
            elif key == ord('-') or key == ord('_'):  # Decrease line thickness
                drawing_thickness = max(1, drawing_thickness - 1)
        except Exception as e:
            print(f"Error handling key press: {e}")
            break
    
    # Clean up
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed")

if __name__ == "__main__":
    main() 

#.\run_finger_detection.bat