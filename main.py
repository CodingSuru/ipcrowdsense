from flask import Flask, render_template, request, Response, jsonify, send_from_directory
import cv2
import json
import base64
import numpy as np
import os
import threading
import time
from detection import PersonCounter

app = Flask(__name__)

# Initialize counter only once when not in debug mode or when needed
counter = None
camera_thread = None
camera_frame = None
camera_running = False
camera_lock = threading.Lock()

def get_counter():
    global counter
    if counter is None:
        counter = PersonCounter()
    return counter

def camera_capture_thread(rtsp_url):
    global camera_frame, camera_running
    
    cap = cv2.VideoCapture(rtsp_url)
    
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Set timeout for read operations (in milliseconds)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    
    if not cap.isOpened():
        print(f"Failed to open RTSP stream: {rtsp_url}")
        camera_running = False
        return
    
    print(f"Successfully connected to RTSP stream: {rtsp_url}")
    
    while camera_running:
        ret, frame = cap.read()
        if ret:
            with camera_lock:
                camera_frame = frame.copy()
        else:
            print("Failed to read frame from RTSP stream")
            time.sleep(1)  # Wait before retrying
            
    cap.release()
    print("Camera capture thread stopped")

def generate_camera_frames():
    global camera_frame
    
    while camera_running:
        with camera_lock:
            if camera_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', camera_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/video/<filename>')
def serve_video(filename):
    video_dir = os.path.join(os.getcwd(), 'video')
    return send_from_directory(video_dir, filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_thread, camera_running
    
    try:
        data = request.json
        rtsp_url = data.get('rtsp_url', '')
        
        if not rtsp_url:
            return jsonify({'success': False, 'message': 'RTSP URL is required'}), 400
        
        # Stop existing camera if running
        if camera_running:
            stop_camera()
        
        camera_running = True
        camera_thread = threading.Thread(target=camera_capture_thread, args=(rtsp_url,))
        camera_thread.daemon = True
        camera_thread.start()
        
        # Wait a moment to check if connection is successful
        time.sleep(2)
        
        if camera_running:
            return jsonify({'success': True, 'message': 'Camera started successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to connect to camera'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_running, camera_thread, camera_frame
    
    camera_running = False
    if camera_thread and camera_thread.is_alive():
        camera_thread.join(timeout=5)
    
    with camera_lock:
        camera_frame = None
    
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/camera_feed')
def camera_feed():
    if not camera_running:
        return jsonify({'error': 'Camera not running'}), 400
    
    return Response(generate_camera_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_camera_frame', methods=['POST'])
def get_camera_frame():
    global camera_frame
    
    try:
        with camera_lock:
            if camera_frame is None:
                return jsonify({'error': 'No camera frame available'}), 400
            
            # Encode frame as base64
            ret, buffer = cv2.imencode('.jpg', camera_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                return jsonify({'frame': f'data:image/jpeg;base64,{frame_b64}'})
            else:
                return jsonify({'error': 'Failed to encode frame'}), 500
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        frame_data = data['frame']
        boundaries = data.get('boundaries', [])
        max_people = data.get('maxPeople', 0)
        
        # Decode base64 image
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get original frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Scale boundaries to match original frame dimensions
        scaled_boundaries = []
        for boundary in boundaries:
            scaled_boundary = []
            for point in boundary:
                scaled_x = point['x'] * (frame_width / data.get('canvasWidth', frame_width))
                scaled_y = point['y'] * (frame_height / data.get('canvasHeight', frame_height))
                scaled_boundary.append({'x': scaled_x, 'y': scaled_y})
            scaled_boundaries.append(scaled_boundary)
        
        # Process frame
        result = get_counter().process_frame_with_boundaries(frame, scaled_boundaries, max_people)
        
        return jsonify(result)
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure video directory exists
    os.makedirs('video', exist_ok=True)
    app.run(debug=False, host='0.0.0.0', port=8081, use_reloader=False, threaded=True)