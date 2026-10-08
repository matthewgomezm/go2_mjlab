# thin wrapper over export.py in src/ ; for cli
# mjlab exports automatically during training to ONNX. this
# is for exporting previous policies not on disk or a specific checkpoint

from unitree_vision_rl.export import main

if __name__ == "__main__":
  main()
