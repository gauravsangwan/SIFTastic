import dash
from dash import dcc
from dash import html
import cv2
import base64
from dash.dependencies import Output,Input, State
import time

# SIFT
sift = cv2.SIFT_create()

# Feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1("SIFT Feature Matching with Webcam"),
        html.Img(id="live-image"),
        html.Div(id="fps-display"),
        dcc.Interval(id='interval', interval=1000, n_intervals=0)
    ]
)

def process_frame(frame):
    start = time.time()

    img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('me.png')

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    end = time.time()
    fps = 1 / (end - start)
    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:300], img2, flags=2)
    # cv2.putText(img3, f'FPS: {int(fps)}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    _, encoded_frame = cv2.imencode(".jpg", img3)
    encoded_frame_base64 = base64.b64encode(encoded_frame.tobytes()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded_frame_base64}", fps

def get_live_frame():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        yield process_frame(frame)

    cap.release()

@app.callback(
    Output("live-image", "src"),
    Output("fps-display", "children"),
    Input("interval", "n_intervals"),
    prevent_initial_call=True,
)
def update_live_image(n):
    frame_generator = get_live_frame()
    frame, fps = next(frame_generator)
    return frame, f"FPS: {int(fps)}"

if __name__ == "__main__":
    app.run_server(debug=True, port = 8991)
