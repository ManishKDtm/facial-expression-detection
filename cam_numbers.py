import cv2
import platform
import subprocess
import re

def list_available_cameras():
    """
    Detect and list all available cameras on the system.
    Returns a list of camera indices and their status.
    """
    available_cameras = []
    max_test_cameras = 10  # Maximum number of cameras to check
    
    print("Checking for available cameras...")
    
    # Try opening each camera index
    for i in range(max_test_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera properties if available
            camera_info = {
                'index': i,
                'name': f"Camera {i}",
                'resolution': (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'backend': cap.getBackendName(),
                'working': True
            }
            
            # Try to get more detailed name if possible
            if platform.system() == 'Windows':
                try:
                    # This works better on Windows with DirectShow
                    cap_dshow = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap_dshow.isOpened():
                        camera_info['name'] = cap_dshow.getBackendName()
                        cap_dshow.release()
                except:
                    pass
            
            available_cameras.append(camera_info)
            cap.release()
        else:
            # Camera not available at this index
            pass
    
    # Additional methods for Linux/Mac to get more camera info
    if platform.system() == 'Linux':
        try:
            # Use v4l2-ctl to get more detailed camera info
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                devices = result.stdout.split('\n\n')
                for device in devices:
                    lines = device.strip().split('\n')
                    if len(lines) >= 2:
                        cam_name = lines[0].strip()
                        cam_devices = [line.strip() for line in lines[1:] if '/dev/video' in line]
                        for dev in cam_devices:
                            index = int(dev.split('/dev/video')[-1])
                            for cam in available_cameras:
                                if cam['index'] == index:
                                    cam['name'] = cam_name
        except:
            pass
    
    elif platform.system() == 'Darwin':  # macOS
        try:
            # Use system_profiler to get camera info on Mac
            result = subprocess.run(['system_profiler', 'SPCameraDataType'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                camera_names = re.findall(r'([A-Za-z0-9\s]+):\n\s*Connected: Yes', result.stdout)
                for i, name in enumerate(camera_names):
                    if i < len(available_cameras):
                        available_cameras[i]['name'] = name.strip()
        except:
            pass
    
    return available_cameras

def display_camera_info(cameras):
    """Display information about detected cameras"""
    if not cameras:
        print("No cameras found!")
        return
    
    print("\nDetected Cameras:")
    print("=" * 60)
    for i, cam in enumerate(cameras, 1):
        print(f"{i}. Index: {cam['index']}")
        print(f"   Name: {cam['name']}")
        print(f"   Resolution: {cam['resolution'][0]}x{cam['resolution'][1]}")
        print(f"   FPS: {cam['fps']:.1f}")
        print(f"   Backend: {cam['backend']}")
        print("-" * 60)

def test_camera(index):
    """Test a specific camera by index"""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Could not open camera at index {index}")
        return
    
    print(f"\nTesting camera {index} - Press 'q' to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            cv2.imshow(f"Camera {index} Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Detect and list available cameras
    cameras = list_available_cameras()
    display_camera_info(cameras)
    
    # Optionally test a camera
    if cameras:
        try:
            cam_num = int(input("\nEnter camera number to test (0 to skip): "))
            if 0 <= cam_num < len(cameras):
                test_camera(cameras[cam_num]['index'])
        except ValueError:
            print("Invalid input, skipping camera test")