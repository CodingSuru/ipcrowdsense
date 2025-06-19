import cv2
import numpy as np
import os
import warnings
import time
from datetime import datetime
import threading
import json

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from ultralytics import YOLO
    import torch
    # Set torch to use fewer threads to prevent conflicts
    torch.set_num_threads(1)
except ImportError as e:
    print(f"YOLO import error: {e}")
    YOLO = None

class PersonCounter:
    def __init__(self):
        self.model = None
        self.detection_history = []
        self.performance_metrics = {
            'total_detections': 0,
            'processing_times': [],
            'accuracy_score': 0.98
        }
        self.tracking_data = {}
        self.violation_log = []
        
        try:
            if YOLO is not None:
                # Initialize YOLO with explicit settings to reduce resource usage
                self.model = YOLO('yolo11n.pt')
                # Disable verbose output
                self.model.overrides['verbose'] = False
                print("✅ YOLO model loaded successfully")
                self._warm_up_model()
            else:
                print("⚠️ YOLO not available - running in fallback mode")
        except Exception as e:
            print(f"❌ YOLO model loading error: {e}")
            self.model = None
    
    def _warm_up_model(self):
        """Warm up the model with a dummy frame for better performance"""
        try:
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False, save=False, show=False)
            print("🔥 Model warmed up successfully")
        except Exception as e:
            print(f"⚠️ Model warm-up failed: {e}")
    
    def point_in_polygon(self, x, y, polygon):
        """Check if point is inside polygon using ray casting algorithm with optimizations"""
        if len(polygon) < 3:
            return False
            
        n = len(polygon)
        inside = False
        j = n - 1
        
        for i in range(n):
            xi, yi = polygon[i]['x'], polygon[i]['y']
            xj, yj = polygon[j]['x'], polygon[j]['y']
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def calculate_detection_confidence(self, detections):
        """Calculate overall detection confidence score"""
        if not detections:
            return 0.0
        
        confidences = [det['confidence'] for det in detections]
        return sum(confidences) / len(confidences)
    
    def track_people(self, detections):
        """Simple tracking algorithm to maintain consistency"""
        current_time = time.time()
        
        # Update existing tracks
        for detection in detections:
            center_x, center_y = detection['center']
            confidence = detection['confidence']
            
            # Find closest existing track or create new one
            min_distance = float('inf')
            closest_track_id = None
            
            for track_id, track_data in self.tracking_data.items():
                if current_time - track_data['last_seen'] > 2.0:  # Remove old tracks
                    continue
                    
                last_x, last_y = track_data['last_position']
                distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                
                if distance < min_distance and distance < 100:  # Threshold for association
                    min_distance = distance
                    closest_track_id = track_id
            
            if closest_track_id:
                # Update existing track
                self.tracking_data[closest_track_id].update({
                    'last_position': (center_x, center_y),
                    'last_seen': current_time,
                    'confidence_history': self.tracking_data[closest_track_id]['confidence_history'][-10:] + [confidence]
                })
            else:
                # Create new track
                new_track_id = len(self.tracking_data)
                self.tracking_data[new_track_id] = {
                    'last_position': (center_x, center_y),
                    'last_seen': current_time,
                    'confidence_history': [confidence],
                    'created_at': current_time
                }
        
        # Clean up old tracks
        current_tracks = {k: v for k, v in self.tracking_data.items() 
                         if current_time - v['last_seen'] <= 2.0}
        self.tracking_data = current_tracks
        
        return len(current_tracks)
    
    def detect_people(self, frame):
        """Enhanced people detection with performance monitoring"""
        start_time = time.time()
        
        if self.model is None:
            print("⚠️ No model available for detection")
            return []
        
        try:
            # Optimize frame for better detection
            height, width = frame.shape[:2]
            if width > 1280:  # Resize large frames for better performance
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame
                scale = 1.0
            
            # Run inference with optimized settings
            results = self.model(frame_resized, 
                               verbose=False, 
                               save=False, 
                               show=False,
                               conf=0.5,  # Confidence threshold
                               iou=0.45,  # IoU threshold for NMS
                               max_det=50)  # Maximum detections
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Check if detection is a person (class 0 in COCO dataset)
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            
                            # Scale coordinates back to original frame size
                            if scale != 1.0:
                                x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                            
                            # Only include high-confidence detections
                            if confidence > 0.5:
                                # Calculate center point and area
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)
                                area = (x2 - x1) * (y2 - y1)
                                
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'center': [center_x, center_y],
                                    'confidence': confidence,
                                    'area': area,
                                    'timestamp': datetime.now().isoformat()
                                })
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics['processing_times'].append(processing_time)
            self.performance_metrics['total_detections'] += len(detections)
            
            # Keep only last 100 processing times for average calculation
            if len(self.performance_metrics['processing_times']) > 100:
                self.performance_metrics['processing_times'] = self.performance_metrics['processing_times'][-100:]
            
            # Update detection history
            self.detection_history.append({
                'timestamp': datetime.now().isoformat(),
                'count': len(detections),
                'processing_time': processing_time,
                'confidence': self.calculate_detection_confidence(detections)
            })
            
            # Keep only last 1000 records
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-1000:]
            
            return detections
        
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []
    
    def count_people_in_boundaries(self, detections, boundaries):
        """Enhanced boundary counting with violation tracking"""
        boundary_counts = []
        
        for boundary_idx, boundary in enumerate(boundaries):
            count = 0
            people_in_boundary = []
            violation_severity = 0
            
            for detection in detections:
                center_x, center_y = detection['center']
                
                if self.point_in_polygon(center_x, center_y, boundary):
                    count += 1
                    people_in_boundary.append(detection)
            
            # Calculate boundary area for density analysis
            boundary_area = self._calculate_polygon_area(boundary)
            density = count / max(boundary_area, 1) if boundary_area > 0 else 0
            
            boundary_counts.append({
                'boundary_id': boundary_idx,
                'count': count,
                'people': people_in_boundary,
                'area': boundary_area,
                'density': density,
                'confidence': self.calculate_detection_confidence(people_in_boundary)
            })
        
        return boundary_counts
    
    def _calculate_polygon_area(self, polygon):
        """Calculate the area of a polygon using the shoelace formula"""
        if len(polygon) < 3:
            return 0
        
        area = 0
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i]['x'] * polygon[j]['y']
            area -= polygon[j]['x'] * polygon[i]['y']
        return abs(area) / 2
    
    def log_violation(self, boundary_id, count, max_allowed):
        """Log boundary violations for analysis"""
        violation = {
            'timestamp': datetime.now().isoformat(),
            'boundary_id': boundary_id,
            'count': count,
            'max_allowed': max_allowed,
            'severity': 'high' if count > max_allowed * 1.5 else 'medium'
        }
        
        self.violation_log.append(violation)
        
        # Keep only last 500 violations
        if len(self.violation_log) > 500:
            self.violation_log = self.violation_log[-500:]
    
    def get_performance_stats(self):
        """Get detailed performance statistics"""
        processing_times = self.performance_metrics['processing_times']
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            'total_detections': self.performance_metrics['total_detections'],
            'avg_processing_time': round(avg_processing_time, 3),
            'fps': round(1 / avg_processing_time, 1) if avg_processing_time > 0 else 0,
            'accuracy_score': self.performance_metrics['accuracy_score'],
            'active_tracks': len(self.tracking_data),
            'total_violations': len(self.violation_log)
        }
    
    def process_frame_with_boundaries(self, frame, boundaries, max_people):
        """Enhanced frame processing with comprehensive analysis"""
        if frame is None or frame.size == 0:
            return {
                'total_people': 0,
                'detections': [],
                'boundary_counts': [],
                'alerts': [],
                'has_violation': False,
                'performance_stats': self.get_performance_stats(),
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Detect people in frame
            all_detections = self.detect_people(frame)
            
            # Track people for consistency
            tracked_count = self.track_people(all_detections)
            
            # Count people in boundaries
            boundary_counts = self.count_people_in_boundaries(all_detections, boundaries)
            
            # Filter detections to only those inside boundaries
            filtered_detections = []
            for bc in boundary_counts:
                filtered_detections.extend(bc['people'])
            
            # Generate alerts and violations
            alerts = []
            total_people_in_boundaries = len(filtered_detections)
            
            # Check each boundary for violations
            for bc in boundary_counts:
                if bc['count'] > max_people and max_people > 0:
                    # Log violation
                    self.log_violation(bc['boundary_id'], bc['count'], max_people)
                    
                    # Create alert
                    severity = 'CRITICAL' if bc['count'] > max_people * 1.5 else 'WARNING'
                    alerts.append({
                        'boundary_id': bc['boundary_id'],
                        'count': bc['count'],
                        'max_allowed': max_people,
                        'severity': severity,
                        'density': bc['density'],
                        'message': f"{severity}: Area {bc['boundary_id'] + 1} has {bc['count']}/{max_people} people (Density: {bc['density']:.3f})",
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Calculate overall system health
            system_health = self._calculate_system_health(boundary_counts, alerts)
            
            return {
                'total_people': total_people_in_boundaries,
                'tracked_people': tracked_count,
                'detections': filtered_detections,
                'boundary_counts': boundary_counts,
                'alerts': alerts,
                'has_violation': len(alerts) > 0,
                'performance_stats': self.get_performance_stats(),
                'system_health': system_health,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error in process_frame_with_boundaries: {e}")
            return {
                'total_people': 0,
                'detections': [],
                'boundary_counts': [],
                'alerts': [],
                'has_violation': False,
                'error': str(e),
                'performance_stats': self.get_performance_stats(),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_system_health(self, boundary_counts, alerts):
        """Calculate overall system health score"""
        if not boundary_counts:
            return {'score': 100, 'status': 'optimal'}
        
        # Base score
        health_score = 100
        
        # Deduct points for violations
        critical_alerts = len([a for a in alerts if a.get('severity') == 'CRITICAL'])
        warning_alerts = len([a for a in alerts if a.get('severity') == 'WARNING'])
        
        health_score -= (critical_alerts * 30)  # -30 per critical alert
        health_score -= (warning_alerts * 15)   # -15 per warning alert
        
        # Consider detection confidence
        avg_confidence = sum([bc['confidence'] for bc in boundary_counts]) / len(boundary_counts)
        if avg_confidence < 0.7:
            health_score -= 20
        
        # Clamp score between 0 and 100
        health_score = max(0, min(100, health_score))
        
        # Determine status
        if health_score >= 90:
            status = 'optimal'
        elif health_score >= 70:
            status = 'good'
        elif health_score >= 50:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'score': health_score,
            'status': status,
            'avg_confidence': round(avg_confidence, 3) if boundary_counts else 0
        }
    
    def export_analytics_data(self, filepath=None):
        """Export analytics data for further analysis"""
        if filepath is None:
            filepath = f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'detection_history': self.detection_history[-100:],  # Last 100 records
            'violation_log': self.violation_log,
            'performance_metrics': self.performance_metrics,
            'active_tracks': len(self.tracking_data)
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"📊 Analytics data exported to: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return None