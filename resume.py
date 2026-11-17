"""Resume YOLO training from last.pt; auto-resume again if a save glitch crashes it."""
import ctypes, glob, sys, time
from ultralytics import YOLO

if __name__ == "__main__":
    if sys.platform == "win32":  # keep the PC awake while this script runs (released on exit)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    for attempt in range(1, 9):
        last = glob.glob("runs/**/arecanut_yolo11/weights/last.pt", recursive=True)[0]
        print(f"=== resume attempt {attempt} from {last}", flush=True)
        try:
            YOLO(last).train(resume=True)
            print("=== TRAINING FINISHED", flush=True)
            break
        except Exception as e:
            if "nothing to resume" in str(e):
                print("=== TRAINING FINISHED", flush=True)
                break
            print(f"=== crashed: {e!r} -> resuming in 10 s", flush=True)
            time.sleep(10)
