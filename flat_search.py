import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os.path

parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
args = parser.parse_args()

if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
pipeline = rs.pipeline()
config = rs.config()

rs.config.enable_device_from_file(config, args.input)
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)


pipeline.start(config)
align = rs.align(rs.stream.color)


MIN_AREA = 2000       
RECTANGULARITY = 0.8  
ASPECT_RATIO = (0.5, 2.0)  

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        
        depth_data = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        
        depth_8u = cv2.convertScaleAbs(depth_data, alpha=0.03)
        blurred = cv2.GaussianBlur(depth_8u, (9,9), 0)
        _, thresh = cv2.threshold(blurred, 100, 200, cv2.THRESH_BINARY_INV)
        
        
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        best_rectangle = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
                
            
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            
            rect_area = cv2.contourArea(box)
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            
            width = max(rect[1])
            height = min(rect[1])
            aspect = width / height if height > 0 else 0
            valid_aspect = ASPECT_RATIO[0] <= aspect <= ASPECT_RATIO[1]
            
            
            score = rectangularity * area if valid_aspect else 0
            
            if score > best_score:
                best_score = score
                best_rectangle = box

        if best_rectangle is not None:
            
            cv2.drawContours(color_image, [best_rectangle], 0, (0,255,0), 2)
            
            
            M = cv2.moments(best_rectangle)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(color_image, f"Rect Score: {best_score:.2f}", (cX-50, cY-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        
        cv2.imshow("Threshold", cleaned)
        cv2.imshow("Rectangular Surface Detection", color_image)
        
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()