# Attribution
#
# This folder contains a copy of DotNeuralNet (MIT License) for BrailleLens
# research use and local improvements.
#
# Upstream: https://github.com/snoop2head/DotNeuralNet
# License:  see LICENSE (MIT, Copyright (c) 2023 snoop2head)
#
# Please cite when using in reports / papers:
#
#   Ahn, Young Jin. DotNeuralNet: Light-weight Neural Network for Optical
#   Braille Recognition in the Wild. 2023.
#
# Live preview (from BrailleLens repo root):
#   py -3.11 DotNeuralNet/live_camera.py --source http://PHONE_IP:8080/video
#
# Sinhala decode from cell patterns (Part 1):
#   py -3.11 DotNeuralNet/decode_patterns.py --patterns 100000 101011 --lang si
#
# Image → lines/words → Sinhala (Part 2):
#   $env:PYTHONIOENCODING = "utf-8"
#   py -3.11 DotNeuralNet/transcribe_image.py --image path\to\page.jpg --lang si --save-overlay out.png
