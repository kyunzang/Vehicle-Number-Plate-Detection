import cv2
import pickle
import re
import easyocr
import numpy as np
import streamlit as st
from ultralytics import YOLO
from db_config import get_user_details
from datetime import datetime
import pandas as pd
import numpy as np
import time
import os

# Load YOLO model for number plate detection
model = YOLO("license_plate_detector(Trained).pt")

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

def detect_number_plate(image_path):
    """Detect the number plate region using YOLO."""
    results = model.predict(source=image_path)  # Get prediction results
    if results and results[0].boxes:  # Ensure results contain bounding boxes
        img = results[0].orig_img  # Original image
        boxes = results[0].boxes  # Bounding boxes object
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
            cropped_plate = img[y1:y2, x1:x2]  # Crop the region
            return cropped_plate  # Return the first detected number plate
    return None


def extract_text_with_easyocr(image):
    """Extract text from an image using EasyOCR and filter for license plate format."""
    results = reader.readtext(image)
    # Combine all detected text
    detected_text = " ".join([res[1] for res in results])
    # Define a regex pattern for license plate formats
    pattern = r"\b[A-Z]{2,3}-\d{1,2}-[A-Z]{1,2}\d{1,4}\b"  # Adjust based on specific plate format requirements
    matches = re.findall(pattern, detected_text)  # Find all matches in the text
    if matches:
        detected_plate = matches[0]
        # Remove incorrect prefixes like "L" and duplicate sequences such as "BPBP"
        detected_plate = re.sub(r"^L", "BP", detected_plate)  # Replace starting "L" with "BP"
        detected_plate = re.sub(r"^(BP)+", "BP", detected_plate)  # Ensure no duplicate "BP"
        return detected_plate
    return None

# Parking space variables
rectW, rectH = 200, 400
try:
    with open('Parking', 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    posList = []

# Streamlit UI
st.title("Smart Parking System")

# Tabs for different functionalities
tab1, tab2 = st.tabs(["🔍 Number Plate Detection", "🚗 Parking Space Monitoring"])

# 1. Number Plate Detection
with tab1:
    st.header("Car Number Plate Detection and Details Retrieval")
    uploaded_image = st.file_uploader("Upload an Image of the Car Number Plate", type=["jpg", "jpeg", "png"])

    if uploaded_image:
         # Display uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        file_bytes = uploaded_image.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, image)

        # Detect the number plate
        st.write("Detecting number plate...")
        plate_image = detect_number_plate(temp_image_path)

        if plate_image is not None:
            gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            st.image(plate_image, caption="Detected Number Plate", use_container_width=True)

            st.write("Extracting text from the number plate...")
            detected_plate = extract_text_with_easyocr(gray_plate)

            if detected_plate:
                st.write(f"Detected Plate: `{detected_plate}`")
                st.write("Fetching details from the database...")
                car_details = get_user_details(detected_plate)
                if car_details:
                    st.subheader("Car Owner Details")
                    st.markdown(
                        f"""
                        <div style="background-color:#060606; padding:15px; border-radius:10px; border:1px solid #555;">
                            <h4 style="color:#FFF; margin-bottom:10px;">Owner Name:</h4>
                            <p style="font-size:20px; color:#6CD56C;">{car_details['owner_name']}</p>
                            <h4 style="color:#FFF; margin-bottom:10px;">Contact Info:</h4>
                            <p style="font-size:20px; color:#6CD56C;">{car_details['contact_info']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("This car does not belong to Government Technology Agency, Royal Government of Bhutan.")
                    st.empty()
            else:
                st.error("Failed to extract a valid number plate. Try another image.")
                st.empty()
        else:
            st.error("No number plate detected. Please upload a clearer image.")
            st.empty()
    else:
        st.info("Please upload an image of a car number plate to start detection.")

## 2. Parking Space Monitoring
with tab2:
    st.header("Parking Space Monitoring")
    
    try:
        with open('Parking', 'rb') as f:
            posList = pickle.load(f)
            totalSpaces = len(posList)
    except FileNotFoundError:
        st.error("Error: Parking position file not found.")
        st.stop()
    
    # Create layout with columns for metrics
    col1, col2, col3 = st.columns([1, 1, 2])
    
    # Initialize metrics
    available_spaces = col1.metric(
        "Available Spaces",
        "0/{}".format(totalSpaces),
        help="Number of available parking spaces"
    )
    
    occupancy_rate = col2.metric(
        "Occupancy Rate",
        "0%",
        help="Percentage of occupied parking spaces"
    )
    
    time_display = col3.empty()
    
    # Camera feed toggle
    show_camera = st.toggle('Show Live Camera Feed', value=False)
    
    # Create placeholders for video feed and table
    video_placeholder = st.empty()
    table_placeholder = st.empty()

    # Initialize parking status DataFrame
    parking_status = pd.DataFrame({
        "Parking Spot": [f"Spot {i+1}" for i in range(totalSpaces)],
        "Status": ["Free" for _ in range(totalSpaces)],
        "Time Parked": ["" for _ in range(totalSpaces)]
    })

    # Add CSS styles
    st.markdown("""
        <style>
            .parking-spot {
                border: 2px solid #ccc;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
                text-align: center;
            }
            .spot-number {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 5px;
            }
            .spot-status {
                font-size: 14px;
                margin-bottom: 5px;
            }
            .spot-time {
                font-size: 12px;
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)

    # Calculate grid layout
    spots_per_row = 4
    num_rows = (totalSpaces + spots_per_row - 1) // spots_per_row

    # Create placeholders for each parking spot
    spot_placeholders = []
    for row in range(num_rows):
        cols = st.columns(spots_per_row)
        row_placeholders = []
        for col in range(spots_per_row):
            spot_index = row * spots_per_row + col
            if spot_index < totalSpaces:
                placeholder = cols[col].empty()
                row_placeholders.append(placeholder)
            else:
                row_placeholders.append(None)
        spot_placeholders.append(row_placeholders)

    def check_and_update(frame, imgPro):
        spaceCount = 0
        
        for idx, pos in enumerate(posList):
            x, y = pos
            crop = imgPro[y:y + rectH, x:x + rectW]
            count = cv2.countNonZero(crop)
            
            if count < 900:
                spaceCount += 1
                color = (0, 255, 0)  # Green for free space
                thick = 5
                # Update parking status
                if parking_status.loc[idx, "Status"] == "Car Parked":
                    parking_status.loc[idx, "Time Parked"] = ""
                parking_status.loc[idx, "Status"] = "Free"
            else:
                color = (0, 0, 255)  # Red for occupied space
                thick = 2
                # Update parking status with timestamp
                if parking_status.loc[idx, "Status"] == "Free":
                    parking_status.loc[idx, "Time Parked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                parking_status.loc[idx, "Status"] = "Car Parked"

            # Draw rectangles on frame
            cv2.rectangle(frame, (x, y), (x + rectW, y + rectH), color, thick)
            
            # Update spot visualization
            row = idx // 4
            col = idx % 4
            bg_color = "#90EE90" if count < 900 else "#FFB6C1"
            time_text = "" if count < 900 else parking_status.loc[idx, "Time Parked"]
            status = "Available" if count < 900 else "Occupied"
            
            spot_html = f"""
                <div class="parking-spot" style="background-color: {bg_color}">
                    <div class="spot-number">Spot {idx + 1}</div>
                    <div class="spot-status">{status}</div>
                    <div class="spot-time">{time_text}</div>
                </div>
            """
            spot_placeholders[row][col].markdown(spot_html, unsafe_allow_html=True)

        # Add space count overlay to frame
        cv2.rectangle(frame, (45, 30), (250, 75), (180, 0, 180), -1)
        cv2.putText(frame, f'Free: {spaceCount}/{totalSpaces}', (50, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        return spaceCount, frame

    # Main video processing loop
    try:
        camera_url = 'http://192.168.43.67:8080/video'
        cap = cv2.VideoCapture(camera_url)
        
        if not cap.isOpened():
            st.error(f"Error: Could not connect to camera at {camera_url}")
            st.info("Please make sure your camera is connected and the URL is correct.")
            st.stop()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera feed")
                break
            
            # Process frame for parking detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 1)
            thresh = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 25, 16
            )
            blur = cv2.medianBlur(thresh, 5)
            kernel = np.ones((3, 3), np.uint8)
            dilate = cv2.dilate(blur, kernel, iterations=1)
            
            # Update parking spots and get available count
            available_count, processed_frame = check_and_update(frame, dilate)
            
            # Update metrics
            occupancy_percent = ((totalSpaces - available_count) / totalSpaces) * 100
            available_spaces.metric(
                "Available Spaces",
                f"{available_count}/{totalSpaces}"
            )
            occupancy_rate.metric(
                "Occupancy Rate",
                f"{occupancy_percent:.1f}%"
            )
            
            # Update time display
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            time_display.markdown(
                f"<h3 style='text-align: center;'>Current Time: {current_time}</h3>",
                unsafe_allow_html=True
            )
            
            # Update video feed if enabled
            if show_camera:
                video_placeholder.image(
                    processed_frame,
                    channels="BGR",
                    use_container_width=True,
                    caption="Live Parking Feed"
                )
            else:
                video_placeholder.empty()
            
            # Display parking status table
            table_placeholder.dataframe(parking_status)
            
            # Add a small delay to prevent excessive updates
            time.sleep(0.1)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()