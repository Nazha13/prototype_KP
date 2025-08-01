# realtime_client_with_hand_interaction.py

import requests
import cv2
import os
import re
import threading
import time
import mediapipe as handtrack
from concurrent.futures import ThreadPoolExecutor

class RealTimeARClient:
    """
    A class to manage the real-time AR tracking client, integrated with
    MediaPipe for hand-based interaction, with performance optimizations.
    """
    def __init__(self, server_url, droidcam_url):
        # --- Configuration ---
        self.server_url = server_url
        self.droidcam_url = droidcam_url
        self.prompt = "Point to the keyboard keys"
        self.dot_radius = 10
        self.dot_color = (0, 0, 255)

        # --- State Variables ---
        self.trackers = []
        self.tracking_active = False
        self.is_detecting = False
        self.is_typing_prompt = False
        self.typed_prompt = ""
        self.latest_hand_results = None # Store results from the hand tracking thread
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=10)

        # --- MediaPipe Hand Tracking Setup ---
        self.handtrack_hands = handtrack.solutions.hands
        self.hands = self.handtrack_hands.Hands(model_complexity=0, min_detection_confidence=0.7)
        self.handtrack_draw = handtrack.solutions.drawing_utils

    def _get_and_track_points(self, frame, prompt):
        """
        [Threaded] Gets points from the server and initializes 2D trackers.
        """
        print("\n[Thread] Sending frame to server for initial detection...")
        temp_frame_path = "temp_frame_for_thread.jpg"
        cv2.imwrite(temp_frame_path, frame)

        try:
            files = {'image': (temp_frame_path, open(temp_frame_path, 'rb'), 'image/jpeg')}
            payload = {'text': prompt}
            response = requests.post(self.server_url, files=files, data=payload)
            response.raise_for_status()
            result = response.json()

            answer_text = result.get('answer', '')
            point_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
            extracted_points = re.findall(point_pattern, answer_text)

            new_trackers = []
            if extracted_points:
                points = [(int(x), int(y)) for x, y in extracted_points]
                for point in points:
                    bbox_size = 50
                    bbox = (point[0] - bbox_size // 2, point[1] - bbox_size // 2, bbox_size, bbox_size)
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, bbox)
                    new_trackers.append(tracker)

            with self.lock:
                self.trackers = new_trackers
                self.tracking_active = True if self.trackers else False

        except requests.exceptions.RequestException as e:
            print(f"[Thread] Network Error: {e}")
        finally:
            with self.lock:
                self.is_detecting = False
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)

    def _update_trackers(self, frame):
        """
        Updates trackers and returns a list of their current center positions.
        """
        if not self.trackers:
            self.tracking_active = False
            return []

        futures = [self.executor.submit(tracker.update, frame) for tracker in self.trackers]

        updated_trackers = []
        dot_positions = []
        for i, future in enumerate(futures):
            success, bbox = future.result()
            if success:
                updated_trackers.append(self.trackers[i])
                center_x = int(bbox[0] + bbox[2] / 2)
                center_y = int(bbox[1] + bbox[3] / 2)
                dot_positions.append((center_x, center_y))

        self.trackers = updated_trackers
        if not self.trackers:
            self.tracking_active = False

        return dot_positions

    def _process_hands_in_background(self, frame):
        """
        [Threaded] Processes the frame for hand landmarks to avoid blocking the main loop.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        with self.lock:
            self.latest_hand_results = results

    def _draw_hud(self, frame):
        """
        Draws the Heads-Up Display on the frame.
        """
        if self.is_typing_prompt:
            cursor = "|" if int(time.time() * 2) % 2 == 0 else ""
            prompt_text = f"New Prompt: {self.typed_prompt}{cursor}"
            cv2.putText(frame, prompt_text, (20, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            if self.is_detecting:
                cv2.putText(frame, "Detecting, please stand still...", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            if not self.tracking_active and not self.is_detecting:
                cv2.putText(frame, "Press 's' to select object(s)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.putText(frame, "Press 'p' to change prompt", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"Prompt: {self.prompt}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def _handle_key_press(self, key, frame):
        """
        Handles all keyboard input for the application.
        """
        if key == 255: return True

        if self.is_typing_prompt:
            if key == 13: # Enter
                with self.lock:
                    self.prompt = self.typed_prompt
                    self.is_typing_prompt = False
                    self.typed_prompt = ""
            elif key == 8: # Backspace
                self.typed_prompt = self.typed_prompt[:-1]
            elif 32 <= key <= 126:
                self.typed_prompt += chr(key)
            return True

        if key == ord('q'): return False

        if key == ord('s'):
            with self.lock:
                self.tracking_active = False
                self.is_detecting = True
                self.trackers = []
            threading.Thread(target=self._get_and_track_points, args=(frame.copy(), self.prompt)).start()

        if key == ord('p'):
            with self.lock:
                self.tracking_active = False
                self.is_detecting = False
                self.trackers = []
                self.is_typing_prompt = True
                self.typed_prompt = ""

        return True

    def run(self):
        """
        Main application loop.
        """
        print(f"Initial prompt is: '{self.prompt}'.")
        cap = cv2.VideoCapture(self.droidcam_url)
        if not cap.isOpened():
            print(f"Error: Could not open DroidCam stream at {self.droidcam_url}")
            return

        hand_thread = None

        while True:
            ret, frame = cap.read()
            if not ret: break

            h, w, c = frame.shape

            dot_positions = []
            was_tracking = self.tracking_active
            if self.tracking_active:
                with self.lock:
                    dot_positions = self._update_trackers(frame)

            # --- Run hand processing in a background thread to prevent lag ---
            if hand_thread is None or not hand_thread.is_alive():
                hand_thread = threading.Thread(target=self._process_hands_in_background, args=(frame.copy(),))
                hand_thread.start()

            if self.latest_hand_results and self.latest_hand_results.multi_hand_landmarks:
                for hand_lms in self.latest_hand_results.multi_hand_landmarks:
                    index_finger_tip = hand_lms.landmark[self.handtrack_hands.HandLandmark.INDEX_FINGER_TIP]
                    ix, iy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                    surviving_trackers = []
                    surviving_dots = []
                    with self.lock:
                        for i, dot_pos in enumerate(dot_positions):
                            distance = ((ix - dot_pos[0])**2 + (iy - dot_pos[1])**2)**0.5
                            if distance < self.dot_radius:
                                print("Collision detected! Dot removed.")
                            else:
                                surviving_trackers.append(self.trackers[i])
                                surviving_dots.append(dot_pos)
                        self.trackers = surviving_trackers
                        dot_positions = surviving_dots

                    if dot_positions:
                        distances = [((ix - dx)**2 + (iy - dy)**2)**0.5 for dx, dy in dot_positions]
                        nearest_dot = dot_positions[distances.index(min(distances))]
                        cv2.arrowedLine(frame, (ix, iy), nearest_dot, (0, 255, 0), 3)

                    self.handtrack_draw.draw_landmarks(frame, hand_lms, self.handtrack_hands.HAND_CONNECTIONS)

            # --- Auto-prompt after last dot is popped ---
            if was_tracking and not self.trackers:
                with self.lock:
                    self.is_typing_prompt = True
                    self.typed_prompt = ""

            for dot_pos in dot_positions:
                cv2.circle(frame, dot_pos, self.dot_radius, self.dot_color, -1)

            self._draw_hud(frame)
            cv2.imshow("Real-time AR Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key_press(key, frame):
                break

        self.executor.shutdown()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SERVER_URL = "https://balanced-vaguely-mastodon.ngrok-free.app/inference/"
    DROIDCAM_URL = "http://192.168.133.7:4747/video"

    client = RealTimeARClient(SERVER_URL, DROIDCAM_URL)
    client.run()