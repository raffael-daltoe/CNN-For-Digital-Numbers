import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import os
import argparse
import sys
from datetime import datetime

class LCDDigitRecognizer:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.image_size = (28, 28)
        self.digit_classes = [str(i) for i in range(10)]
        self.digit_regions = [] 
        
    def select_multiple_regions(self, frame, num_digits):
        """Allow user to select 4 points for EACH digit sequentially in ONE window"""
        self.digit_regions = []
        base_frame = frame.copy()
        
        window_name = "Select Digits"
        cv2.namedWindow(window_name)
        
        print(f"Starting selection for {num_digits} digits.")
        print("Left Click: Select point | 'c': Confirm digit | 'r': Reset current digit | ESC: Quit")

        current_digit_points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(current_digit_points) < 4:
                current_digit_points.append((x, y))
                
        cv2.setMouseCallback(window_name, mouse_callback)
        
        digit_idx = 0
        while digit_idx < num_digits:
            display_img = base_frame.copy()
            
            # --- UI Instructions ---
            cv2.putText(display_img, f"Select Digit {digit_idx+1} of {num_digits}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_img, "Points: Left Click | Confirm: 'c' | Reset: 'r'", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # --- Draw the ACTIVE selection (Green) ---
            for pt in current_digit_points:
                cv2.circle(display_img, pt, 4, (0, 255, 0), -1)
            
            if len(current_digit_points) > 1:
                pts = np.array(current_digit_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                is_closed = (len(current_digit_points) == 4)
                cv2.polylines(display_img, [pts], is_closed, (0, 255, 0), 2)

            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and len(current_digit_points) == 4:
                # 1. Save the region
                self.digit_regions.append(np.array(current_digit_points, dtype="float32"))
                
                # 2. Draw this confirmed digit PERMANENTLY on base_frame (Red)
                pts = np.array(current_digit_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(base_frame, [pts], True, (0, 0, 255), 2)
                
                label_pos = (current_digit_points[0][0], max(0, current_digit_points[0][1] - 5))
                cv2.putText(base_frame, str(digit_idx + 1), label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                print(f"Digit {digit_idx+1} confirmed.")
                
                # 3. Reset for next digit
                current_digit_points.clear() 
                digit_idx += 1
                
            elif key == ord('r'):
                current_digit_points.clear()
                print(f"Reset points for Digit {digit_idx+1}")
                
            elif key == 27: # ESC
                cv2.destroyAllWindows()
                return False
            
        cv2.destroyAllWindows()
        return True
    
    def perspective_crop(self, image, points):
        rect = self.order_points(points)
        (tl, tr, br, bl) = rect
        
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped

    def order_points(self, points):
        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        return rect

    def preprocess_image(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if np.mean(binary) > 127:
            binary = 255 - binary
            
        return binary

    def recognize_digits(self, frame):
        recognized_digits = []
        debug_frame = frame.copy()
        
        for i, points in enumerate(self.digit_regions):
            digit_img = self.perspective_crop(frame, points)
            processed = self.preprocess_image(digit_img)
            
            resized = cv2.resize(processed, self.image_size)
            normalized = resized.astype('float32') / 255.0
            model_input = normalized.reshape(1, self.image_size[0], self.image_size[1], 1)
            
            predictions = self.model.predict(model_input, verbose=0)
            digit_class = np.argmax(predictions)
            confidence = np.max(predictions)
            
            recognized_digits.append({
                'digit': str(digit_class),
                'confidence': float(confidence),
                'region_index': i
            })
            
            pts = points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug_frame, [pts], True, (0, 255, 0), 1)
            cv2.putText(debug_frame, str(digit_class), (pts[0][0][0], pts[0][0][1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return recognized_digits, debug_frame

    def process_video(self, video_path, num_digits, decimal_point, output_csv='output.csv'):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video detected FPS: {fps}")
        
        if fps <= 0 or np.isnan(fps):
            print("Warning: FPS not found or invalid. Defaulting to 30 FPS.")
            fps = 30
            
        frames_per_second = int(fps)
        if frames_per_second == 0:
            frames_per_second = 0.5
            
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame from video")
        
        print(f"Please select the regions for {num_digits} digits...")
        
        if not self.select_multiple_regions(first_frame, num_digits=num_digits):
            print("Selection cancelled")
            return
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        with open(output_csv, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Header generation
            header = ['Second', 'Full Value'] + [f'Digit_{i+1}' for i in range(len(self.digit_regions))]
            csv_writer.writerow(header)
            
            current_second = 0
            frame_count = 0
            
            print("Processing video...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frames_per_second == 0:
                    try:
                        results, debug_frame = self.recognize_digits(frame)
                        
                        # --- Logic to insert decimal point ---
                        raw_digits = [r['digit'] for r in results]
                        
                        if 0 < decimal_point < len(raw_digits):
                            # Split list and insert dot
                            part_int = "".join(raw_digits[:decimal_point])
                            part_dec = "".join(raw_digits[decimal_point:])
                            full_value_str = f"{part_int}.{part_dec}"
                        else:
                            full_value_str = "".join(raw_digits)
                        
                        row = [current_second, full_value_str] + raw_digits
                        csv_writer.writerow(row)
                        
                        print(f"Second {current_second}: {full_value_str}")
                        cv2.imshow("Processing Video", debug_frame)
                        current_second += 1
                        
                    except Exception as e:
                        print(f"Error at second {current_second}: {e}")
                
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Results saved to {output_csv}")

def main_video_processing():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="LCD Digit Recognition from Video")
    
    parser.add_argument(
        '--video', 
        type=str, 
        required=True, 
        help='Path to the input video file'
    )
    
    parser.add_argument(
        '--digits', 
        type=int, 
        default=5, 
        required=True,
        help='Number of digits to read (default: 5)'
    )
    
    parser.add_argument(
        '--decimal', 
        type=int, 
        default=0, 
        required=True,
        help='Position of decimal point from left (e.g., 3 for 147.59). Use 0 for no decimal.'
    )
    
    args = parser.parse_args()

    # Model Check
    model_path = 'lcd_digit_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    recognizer = LCDDigitRecognizer(model_path)
    output_csv = 'video_analysis_results.csv'
    
    try:
        print(f"Analyzing: {args.video}")
        print(f"Digits: {args.digits}")
        print(f"Decimal Point Position: {args.decimal}")
        
        recognizer.process_video(
            video_path=args.video, 
            num_digits=args.digits, 
            decimal_point=args.decimal,
            output_csv=output_csv
        )
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main_video_processing()