#!/usr/bin/env python3
"""
Performance comparison script to show improvements
"""

import time
import cv2
import numpy as np
from src.yolo_v12.config import load_config
from src.yolo_v12.detector import Detector


def test_model_performance(model_path, conf_threshold, test_frames=10):
    """Test detection performance with different models"""

    # Create test configuration
    test_config = {
        "person": {
            "weights": model_path,
            "conf": conf_threshold,
            "iou": 0.45,
            "classes": [0]
        }
    }

    try:
        detector = Detector(test_config)

        # Create test frame (640x480)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Warm up
        for _ in range(3):
            detector.infer(test_frame)

        # Time the detection
        start_time = time.time()
        for _ in range(test_frames):
            dets = detector.infer(test_frame)
        end_time = time.time()

        avg_time = (end_time - start_time) / test_frames
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            "model": model_path,
            "confidence": conf_threshold,
            "avg_time_ms": round(avg_time * 1000, 1),
            "fps": round(fps, 1),
            "success": True
        }

    except Exception as e:
        return {
            "model": model_path,
            "confidence": conf_threshold,
            "error": str(e),
            "success": False
        }


def main():
    print("🚀 Productivity Tracker Performance Comparison")
    print("=" * 50)

    # Test different model configurations
    test_configs = [
        ("yolov8n.pt", 0.35, "Nano (Fastest)"),
        ("yolov8s.pt", 0.35, "Small (Optimized)"),
        ("yolov8m.pt", 0.35, "Medium (Original)"),
    ]

    results = []

    for model_file, conf, description in test_configs:
        print(f"\n📊 Testing {description}...")
        result = test_model_performance(model_file, conf)
        result["description"] = description
        results.append(result)

        if result["success"]:
            print(f"   ✅ Average inference time: {result['avg_time_ms']}ms")
            print(f"   ✅ Estimated FPS: {result['fps']}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")

    # Display comparison table
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'Confidence':<12} {'Time (ms)':<12} {'FPS':<8} {'Status'}")
    print("-" * 70)

    for result in results:
        if result["success"]:
            print(f"{result['description']:<20} {result['confidence']:<12} "
                  f"{result['avg_time_ms']:<12} {result['fps']:<8} ✅")
        else:
            print(f"{result['description']:<20} {result['confidence']:<12} "
                  f"{'N/A':<12} {'N/A':<8} ❌")

    # Show recommendations
    print("\n" + "=" * 50)
    print("💡 RECOMMENDATIONS")
    print("=" * 50)

    successful_results = [r for r in results if r["success"]]
    if successful_results:
        # Find best performing model
        best_fps = max(r["fps"] for r in successful_results)
        best_model = next(
            r for r in successful_results if r["fps"] == best_fps)

        print(f"🏆 Best Performance: {best_model['description']}")
        print(f"   - FPS: {best_model['fps']}")
        print(f"   - Inference Time: {best_model['avg_time_ms']}ms")

        # Show optimization impact
        if len(successful_results) >= 2:
            slowest = min(r["fps"] for r in successful_results)
            improvement = ((best_fps - slowest) / slowest) * 100
            print(f"   - Performance Improvement: {improvement:.1f}%")

    print("\n🎯 Configuration Tips:")
    print("   - For real-time tracking: Use yolov8s.pt or yolov8n.pt")
    print("   - For accuracy: Use yolov8m.pt (if FPS > 5)")
    print("   - Confidence threshold: 0.35-0.45 for good balance")
    print("   - Frame skipping: Process every 2-3 frames for speed")


if __name__ == "__main__":
    main()

