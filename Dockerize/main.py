# main.py

from inferencing import main as inference_main
from monitor import main as monitor_main

if __name__ == "__main__":
    print("🔄 Starting batch inferencing...")
    inference_main()
    print("✅ Batch inferencing complete.\n")

    print("🔍 Starting model monitoring...")
    monitor_main()
    print("✅ Monitoring complete.")
