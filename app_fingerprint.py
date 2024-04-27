import os
import cv2
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import base64
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

app = dash.Dash(__name__)

# Define paths
base_path = '/home/iiticos/Desktop/College/8th sem/CV/Project/finger/socofing/SOCOFing'
real_images_path = os.path.join(base_path, 'Real')
altered_images_path = os.path.join(base_path, 'Altered', 'Altered-Hard')

# Function to read images
def read_image(filepath):
    return cv2.imread(filepath)

# Function to perform matching
def perform_matching(sample_image, real_images):
    best_score = 0
    best_match = None

    sift = cv2.SIFT_create()
    keypoints_1, des1 = sift.detectAndCompute(sample_image, None)

    for filename, fingerprint_img in tqdm(real_images.items(), desc='Matching'):
        keypoints_2, des2 = sift.detectAndCompute(fingerprint_img, None)

        matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(
            des1, des2, k=2
        )

        match_points = []
        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                match_points.append(p)

        keypoints = min(len(keypoints_1), len(keypoints_2))
        score = len(match_points) / keypoints * 100

        if score > best_score:
            best_score = score
            best_match = filename
    kp1, kp2, mp = keypoints_1, keypoints_2, match_points
    result = cv2.drawMatches(sample_image, kp1, fingerprint_img, kp2, mp, None)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    plt.figure(figsize=(10, 10))
    plt.imshow(result)
    plt.axis('off')
    plt.title('Best Match')
    plt.savefig('best_match_plot.png', bbox_inches='tight')  # Save the plot
    
    return best_match, best_score

# Read all real images
real_images = {filename: read_image(os.path.join(real_images_path, filename)) 
               for filename in os.listdir(real_images_path)}

# Get 5 random images from Altered-Hard folder
random_images = random.sample(os.listdir(altered_images_path), 5)
labels = ['Fingerprint1', 'Fingerprint2', 'Fingerprint3', 'Fingerprint4', 'Fingerprint5']
# App layout
app.layout = html.Div([
    html.H1("SIFT Fingerprint Matching"),
    html.H2("CSL7360"),
    html.P("An example of the 5 random finger prints from the test set is as follows."),
    html.Img(src=random_images[0]),
    dcc.Dropdown(
        id='image-dropdown',
        options=[{'label': labels[i], 'value': random_images[i]} for i in range(len(random_images))],
        value=random_images[0]
    ),
    html.Div(id='output-image-upload'),
    dcc.Interval(id='progress-interval', interval=10, n_intervals=0),
    html.Div(id='progress-bar-output')
])

# Callback to update progress bar
@app.callback(Output('progress-bar-output', 'children'),
              [Input('progress-interval', 'n_intervals')],
              [State('image-dropdown', 'value')])
def update_progress_bar(n_intervals, selected_image):
    total_images = len(real_images)
    progress_percentage = (n_intervals % total_images) / total_images * 100
    return html.Div([
        html.P('Matching Progress: {:.2f}%'.format(progress_percentage))
    ])

# Callback to display selected image and perform matching
@app.callback(Output('output-image-upload', 'children'),
              [Input('image-dropdown', 'value')])
def update_output(selected_image):
    # Read selected image
    sample_image = read_image(os.path.join(altered_images_path, selected_image))

    # Perform matching
    best_match, best_score = perform_matching(sample_image, real_images)

    # Display results
    result = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    encoded_image = base64.b64encode(cv2.imencode('.png', result)[1]).decode()

    output_image = html.Div([
        html.H4('Selected Image: {}'.format(selected_image)),
        html.H4('Best Match: {}'.format(best_match)),
        html.H4('Best Score: {}'.format(best_score)),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
        # html.Img(src='best_match_plot.png')  # Reference to the saved plot
    ])

    return output_image

if __name__ == '__main__':
    app.run_server(debug=True, port = '8060')
