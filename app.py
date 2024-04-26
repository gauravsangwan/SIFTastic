import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from PIL import Image
import base64
import time
import torch
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import joblib 
import subprocess

app = dash.Dash(__name__)
server = app.server


def process_image1(image_path):
    # Visual classificer 
    return "The image belongs to XYZ Class"

def sift_keypoints(tensor):
    EDGE_THRESHOLD = 6
    sift = cv2.SIFT_create()
    sift.setEdgeThreshold(EDGE_THRESHOLD)
    array = tensor.numpy().transpose((1,2,0))*255
    array = array.astype(np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    key_points_sift, descriptor = sift.detectAndCompute(array,None)
    image = array.astype(np.float64)
    mean_img, std_img = cv2.meanStdDev(image)
    mean = np.mean(mean_img)
    std = np.mean(std_img)
    return len(key_points_sift), mean, std



def adv_predict(image):
    in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    image = in_transform(image)
    columns = ['KP', 'mean', 'std']
    clf = LogisticRegression()
    filename = 'finalized_model.sav'
    clf = joblib.load(filename)
    kp, mean, std = sift_keypoints(image)
    new_data = pd.DataFrame([[kp, mean, std]], columns= columns)
    prediction = clf.predict(new_data)
    #print(f'Prediction: {prediction[0]}')
    probability = clf.predict_proba(new_data)[:, 1]
    #print(f'Probability of positive class: {probability[0]:.4f}')
    return prediction[0]

def process_image2(image_path):
    # ADV Detector

    image = plt.imread(image_path)
    prediction = adv_predict(image)
    if prediction == 1:
        return "The input image is Adversarial Image"
    else:
        return "The input image is Non-Adversarial Image"


def process_image3(image_path):
    subprocess.Popen(["python", "app_webcam_chat.py"])
    return "Go to the another app_webcam.py to see the output"

def process_image4(image_path):
    return "Run the IPYNB notebook to generate the 3d reconstruction"

# Map the function names to the actual functions
image_processing_functions = {
    "Visual classification": process_image1,
    "Adversarial Detection": process_image2,
    "Live WebCam matching": process_image3,
    "3D Reconstruction from Camera Images": process_image4,
}
app.layout = html.Div(
    children=[
        html.H1("SIFTastic: Unveiling Reality: SIFT Applications for Visual Classification, Adversarial Detection, 3D Reconstruction, and Fingerprint Matching", style={"font-family": "Arial, sans-serif", "color": "#333", "margin-bottom": "10px"}),
        html.H2("Computer Vision CSL7360", style={"font-family": "Arial, sans-serif", "color": "#666"}),
        html.H3("Gaurav Sangwan (B20AI062),Mukul Shingwani (B20AI023), Shashank Asthana (B21CS093), Anushkaa Ambuj (B21ES006)", style={"font-family": "Arial, sans-serif", "color": "#999"}),
        dcc.Upload(
            id="upload-image",
            children=html.Div(["Drag and Drop or ", html.A("Select Image")]),
            style={
                "width": "50%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px auto",
                "font-family": "Arial, sans-serif",
                "color": "#333",
                "background-color": "#f0f0f0",
            },
            multiple=False,
        ),
        dcc.Dropdown(
            id="function-selector",
            options=[{'label': function_name, 'value': function_name} for function_name in image_processing_functions.keys()],
            value=list(image_processing_functions.keys())[0],  # Default to the first function
            style={"width": "50%", "margin": "10px auto"},
        ),
        html.Div(id="output-image", style={"font-family": "Arial, sans-serif", "color": "#333"}),
        html.Img(id="live-image"),
        html.Div(id="fps-display"),
        dcc.Interval(id='interval', interval=1000, n_intervals=0),
        html.H4("Links for webcam matching tasks, as it requires a completely different layout and callback"),
        html.A("Live WebCam matching", href="http://127.0.0.1:8991/"),
        html.P("Link for 3D Reconstruction from Camera Images's ipynb notebook but its output can be seen in window below."),
        html.A("3D Reconstruction from Camera Images", href="http://"),
    ],
    style={"margin": "20px", "text-align": "center"},
)

@app.callback(
    Output("output-image", "children"),
    Input("upload-image", "contents"),
    Input("function-selector", "value"),
    prevent_initial_call=True,
)
def update_output(contents, selected_function):
    if contents is not None:
        content_type, content_string = contents.split(",")
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(content_string))

        # Call the selected image processing function
        output_str = image_processing_functions[selected_function](image_path)
        # Display the prediction results
        output = [
            html.H3(output_str, style={"font-family": "Arial, sans-serif", "color": "#333"}),
        ]
        return output

    return ""
if __name__ == "__main__":
    app.run_server(debug=True, port=8059)