from collections import defaultdict
import onnxruntime as ort
import cv2
import numpy as np

from ultralytics import YOLO

import json
import cv2, time

from TrafficLaneDetector import UltrafastLaneDetectorV2

from TrafficLaneDetector.ufldDetector.utils import LaneModelType, OffsetType, CurvatureType


def display_multiple_sign_info(frame, sign_info_list):

    # Text and background properties
    background_color = (120, 0, 0) # Dark blue
    text_color = (255, 255, 255)  # White
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 # Size of the font

    padding = 10 # Padding around the text
    line_spacing = 5 # Additional spacing between lines

    # Calculate total height needed for the text bar
    total_text_height = 0
    max_text_width = 0

    # First pass: Calculate total dimensions needed
    for sign_id in sign_info_list:
        if sign_info_list[sign_id] != 'id':
            
            info_text = f"Sign: {sign_id} - {sign_info_list[sign_id]['name']}"
            (text_width, text_height), baseline = cv2.getTextSize(info_text, font, font_scale, font_thickness)
            total_text_height += (text_height + baseline + line_spacing) # Height for current line + spacing
            max_text_width = max(max_text_width, text_width)

    # Adjust total height and bar dimensions
    bar_height = total_text_height + 2 * padding - line_spacing # Subtract last line_spacing as it's not needed below the last line
    bar_width = frame.shape[1] # Use frame width

    # Draw the background rectangle at the top
    cv2.rectangle(frame, (0, 0), (bar_width, bar_height), background_color, -1)

    # Second pass: Overlay each line of text
    current_y = padding # Starting Y position for the first line

    for sign_id in sign_info_list:
        if sign_info_list[sign_id] != 'id':
            info_text = f"Sign: {sign_id} - {sign_info_list[sign_id]['name']}"
            (text_width, text_height), baseline = cv2.getTextSize(info_text, font, font_scale, font_thickness)
    
            # Position for the current line
            text_x = padding
            # Add text_height to current_y to get the baseline for putText
            cv2.putText(frame, info_text, (text_x, current_y + text_height), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
            # Move Y position down for the next line
            current_y += (text_height + baseline + line_spacing)

    return frame






with open('signs_pl.json', 'r', encoding='utf-8') as file:
    signs_pl = json.load(file)
with open('signs_en.json', 'r', encoding='utf-8') as file:
    signs_en = json.load(file)
    
model = YOLO("road.onnx")
sign_model = YOLO("sign_best.onnx")
signs=signs_en #select signs_pl or signs_en
video_path = "driving.mp4" # choose video file of dashcam recording (0 or 1,2,3... for camera input) 
cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])


lane_config = {
	"model_path": "./TrafficLaneDetector/models/culane_res18.onnx",
	"model_type" : LaneModelType.UFLDV2_CULANE
}
cv2.namedWindow("YOLO11 Tracking", cv2.WINDOW_NORMAL)	
UltrafastLaneDetectorV2.set_defaults(lane_config)
laneDetector = UltrafastLaneDetectorV2()


sign_display_counter={}

counter = 0
#set reset interval in seconds
reset_int=60
#set framerate
framerate=30

sign_delay=10

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        try:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        except:
            print("no ids")
        frame_show = frame.copy()
        #print(list(results[0].boxes.cls.cpu().numpy()))
        
        annotated_frame = results[0].plot()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)


            if int(list(results[0].boxes.cls.cpu().numpy())[track_ids.index(track_id)]) ==2:
                sign_display_counter["Green Light"]={"time":15, "type":"", "name":"Go"}
            if int(list(results[0].boxes.cls.cpu().numpy())[track_ids.index(track_id)]) ==7:
                sign_display_counter["Red Light"]={"time":3, "type":"", "name":"Stop"}

            if int(list(results[0].boxes.cls.cpu().numpy())[track_ids.index(track_id)]) in (1,6,10):
                box_sign=boxes[track_ids.index(track_id)]
                x_s, y_s, w_s, h_s = box_sign
            
                x_s, y_s, w_s, h_s=int(x_s), int(y_s), int(w_s), int(h_s)

                x1=x_s-(w_s//2)
                x2=x_s+(w_s//2)
                y1=y_s-(h_s//2)
                y2=y_s+(h_s//2)

            
                crop=frame[y1:y2, x1:x2]
            
                if crop.shape[0]>35 and crop.shape[1]>35 and counter%5==0:
                    print("a"*1111)
                    signs_results=sign_model.predict(crop)
                    print(signs_results[0].boxes.cls)
                    for x in list(signs_results[0].boxes.cls):
                        sign_symbol = signs_results[0].names[int(x)]
                        if sign_symbol in ('A-7', 'A-14',  'A-16',  'A-17', 'A-24',  'A-30',  'B-1',  'B-2', 'B-20','B-25','C-13','D-1','D-2','D-3','D-18','G-3'):
                            sign_type=signs[sign_symbol]['type']
                            sign_name=signs[sign_symbol]['name']
                            
                            print(sign_symbol,sign_type,sign_name)
                            sign_display_counter[sign_symbol]={"time":15, "type":sign_type, "name":sign_name}


        
                        
            
        
        if counter%10==0:
            for element in list(sign_display_counter.keys()):
            
                sign_display_counter[element]["time"]-=1
                if sign_display_counter[element]["time"]==0:
                    sign_display_counter.pop(element)
                    
        print(sign_display_counter)        
        display_multiple_sign_info(annotated_frame, sign_display_counter)

        
        lane_time = time.time()
        laneDetector.DetectFrame(frame_show)
        lane_infer_time = round(time.time() - lane_time, 4)

        laneDetector.DrawDetectedOnFrame(annotated_frame)
        laneDetector.DrawAreaOnFrame(annotated_frame)

        cv2.imshow("YOLO11 Tracking", annotated_frame)

        try:
            previous_id=last_id
        except:
            previous_id=0
        last_id=list(track_history.keys())[-1]


        
        counter+=1
        if counter>reset_int*framerate:
            counter = 0
            model = YOLO("road.pt")
            track_history = defaultdict(lambda: [])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
