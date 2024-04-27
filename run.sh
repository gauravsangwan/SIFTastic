#!/bin/bash

# Open three terminal windows and run the Python scripts
gnome-terminal --title="Terminal 1" -- python3 app.py &
gnome-terminal --title="Terminal 2" -- python3 app_webcam.py &
gnome-terminal --title="Terminal 3" -- python3 app_fingerprint.py &
gnome-terminal --title="Terminal 4" -- python3 app_3d.py &