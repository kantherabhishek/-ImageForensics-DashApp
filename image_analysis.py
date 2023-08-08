from __future__ import print_function
import cv2 as cv
import argparse
import os
import io
import base64
import cv2
import dash
from dash import dcc,html, dash_table    
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import imageio.v2 as imageio
import imagehash
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

def perceptual_hash(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hash_value = imagehash.average_hash(Image.fromarray(image))
    return hash_value

def find_duplicates(image, tile_size, step_size, threshold=10):

    hash_value = perceptual_hash(image)
    duplicate_regions = []

    for y in range(0, image.shape[0] - tile_size + 1, step_size):
        for x in range(0, image.shape[1] - tile_size + 1, step_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tile_hash = perceptual_hash(tile)
            hamming_distance = hash_value - tile_hash

            if hamming_distance <= threshold:
                duplicate_regions.append((y, x, y + tile_size, x + tile_size))

    return duplicate_regions

def clone_detection_function(image, tile_size, step_size, threshold=10):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image_clone_detection = image.copy()
    duplicates = find_duplicates(image, tile_size, step_size, threshold)
    for region in duplicates:
        y1, x1, y2, x2 = region
        cv2.rectangle(image_clone_detection, (x1, y1), (x2, y2), (255, 0, 0), 2)
    fig = go.Figure()
    fig.add_trace(go.Image(z=image_clone_detection))
    fig.update_layout(title="Clone Detection Plot")

    return fig

def perform_ela(image, scale=10):
    temp_filename = "temp.jpg"    
    cv2.imwrite(temp_filename, image, [cv2.IMWRITE_JPEG_QUALITY, scale])
    ela_image = cv2.imread(temp_filename)
    ela_image = cv2.absdiff(image, ela_image)
    ela_image = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
    ela_image = cv2.equalizeHist(ela_image)
    ela_image = cv2.cvtColor(ela_image, cv2.COLOR_GRAY2RGB)
    os.remove(temp_filename)
    return ela_image

def error_level_analysis_function(image):
    ela_image = perform_ela(image)
    fig = go.Figure()
    fig.add_trace(go.Image(z=ela_image))
    fig.update_layout(title="Error Level Analysis with Histogram Equalization")

    return fig


def histogram_equalization(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image

def noise_analysis_function(image):
    equalized_image = histogram_equalization(image)
    noise_image = cv2.absdiff(image, cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB))
    fig = go.Figure()
    fig.add_trace(go.Image(z=noise_image))
    fig.update_layout(title="Noise Analysis Plot (Histogram Equalization)")
    return fig

def adjust_brightness(image, alpha):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def adjust_opacity(image, opacity):  
    return cv2.addWeighted(image, opacity, np.zeros(image.shape, image.dtype), 1 - opacity, 0)

def level_sweep_function(image, levels, opacities):
    rows = len(levels)
    cols = len(opacities)
    fig = go.Figure()
    for i, alpha in enumerate(levels):
        for j, opacity in enumerate(opacities):
            adjusted_image = adjust_opacity(adjust_brightness(image.copy(), alpha), opacity)

            fig.add_trace(go.Image(z=adjusted_image, opacity=1/(rows*cols), name=f"Alpha={alpha}, Opacity={opacity}"))

    fig.update_layout(title="Level Sweep Plot",
                      grid=dict(rows=rows, columns=cols),
                      showlegend=False,
                      width=2024,
                      height=2000
                      )  
    return fig


def calculate_luminance_gradient(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradient_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
    gradient_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return magnitude

def luminance_gradient_function(image):
    luminance_gradient = calculate_luminance_gradient(image)
    x, y = np.meshgrid(np.arange(luminance_gradient.shape[1]), np.arange(luminance_gradient.shape[0]))
    fig = go.Figure(data=[go.Surface(z=luminance_gradient, colorscale='Viridis')])
    fig.update_layout(title="Luminance Gradient Plot",
                      scene=dict(
                          xaxis_title='X',
                          yaxis_title='Y',
                          zaxis_title='Luminance Gradient'
                      ))

    return fig

def extract_metadata_from_file(file_path):

    try:
        with Image.open(file_path) as img:
            exifdata = img._getexif()
        if exifdata:
            metadata = {Image.TAGS.get(tag): value for tag, value in exifdata.items() if Image.TAGS.get(tag)}
            return metadata
        else:
            return None
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None
  

def get_quantization_tables(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

    y_channel = img_ycrcb[:, :, 0]
    cb_channel = img_ycrcb[:, :, 1]
    cr_channel = img_ycrcb[:, :, 2]

    y_quantization_table = np.round(y_channel / 16).astype(int)
    cbcr_quantization_table = np.round((cb_channel + 128) / 16).astype(int)

    return y_quantization_table, cbcr_quantization_table

def create_quantization_table_data(quantization_table):
    return [{"Column {}".format(i+1): value for i, value in enumerate(row)} for row in quantization_table]


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout for the Upload component
upload_layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Image')
        ]),
        style={
            'width': '100%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload'),
    
])



# Layout for the Tab content
tab_content_layout = html.Div(
    [
        dcc.Tabs(
            id="tabs",
            value="tab-1",
            children=[
                dcc.Tab(
                    label="Original Image",
                    value="tab-1",
                    children=[
                        html.H4("Original Image"),
                        html.Div(id="tab1-content"),
                    ],
                ),
                dcc.Tab(
                    label="Clone Detection Plot",
                    value="tab-2",
                    children=[
                        html.Div([
                            html.Label("Tile Size (For Clone Detection only)"),
                            dcc.Slider(
                                id="tile-size-slider",
                                min=8,
                                max=128,
                                step=8,
                                value=32,
                                marks={size: str(size) for size in range(8, 129, 8)},
                            ),
                            html.Label("Step Size (For Clone Detection only)"),
                            dcc.Slider(
                                id="step-size-slider",
                                min=8,
                                max=128,
                                step=8,
                                value=16,
                                marks={size: str(size) for size in range(8, 129, 8)},
                            ),
                        ]),
                        html.Div(id="tab2-content"),
                    ],
                ),
                dcc.Tab(
                    label="Error Level Analysis Plot",
                    value="tab-3",
                    children=[
                        html.Div(id="tab3-content"),
                    ],
                ),
                dcc.Tab(
                    label="Noise Analysis Plot",
                    value="tab-4",
                    children=[
                        html.Div(id="tab4-content"),
                    ],
                ),
                dcc.Tab(
                    label="Level Sweep Plot",
                    value="tab-5",
                    children=[
                        html.Div([
                        html.Label("Enter the levels (comma-separated):"),
                        dcc.Input(
                            id='levels-input',
                            type='text',
                            value='0.2, 0.4, 0.6, 0.8, 1.0',
                            style={'width': '100%'}
                        ),
                    ], style={'margin': '10px'}),
                    html.Div([
                        html.Label("Enter the opacities (comma-separated):"),
                        dcc.Input(
                            id='opacities-input',
                            type='text',
                            value='0.2, 0.4, 0.6, 0.8, 1.0',
                            style={'width': '100%'}
                        ),
                    ], style={'margin': '10px'}),
                        html.Div(id="tab5-content"),
                    ],
                ),
                dcc.Tab(
                    label="Luminance Gradient Plot",
                    value="tab-6",
                    children=[
                        html.Div(id="tab6-content"),
                    ],
                ),
                dcc.Tab(
                    label="Geo and Meta Tags",
                    value="tab-7",
                    children=[
                        html.Div(id="tab7-content"),
                    ],
                ),
                dcc.Tab(
                    label="JPEG Quantization Tables",
                    value="tab-8",
                    children=[
                        html.Div(id="tab8-content"),
                    ],
                ),
                dcc.Tab(
                    label="Orb Detection Plot",
                    value="tab-9",
                    children=[
                        html.Div(id="tab9-content"),
                        dcc.Graph(id='feature-plot'),
                    ]
                )
            ],
        )
    ]
)

# Define the app layout
app.layout = html.Div([
    html.H1("Image Forensics", style={'text-align': 'center'}),
    upload_layout,
    html.Div([
        html.Label("Enter the image path:"),
        dcc.Input(
            id='image-path-input',
            type='text',
            value='C:/Users/Purple/OneDrive/Pictures/Camera Uploads/2013-01-28 19.01.06.jpg',
            style={'width': '100%'}
        ),
    ], style={'margin': '10px'}),
    html.Div([
    
    html.Button(
            id='submit-button',
            n_clicks=0,
            children='Submit',
            style={'width': '100%'}
        ),
    ], style={'margin': '10px'}),
    
    tab_content_layout
])

# Function to read the uploaded image
def read_image(contents):
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    image = imageio.imread(io.BytesIO(decoded))
    return image


# Callback to update the image on the Original Image tab
@app.callback(Output('tab1-content', 'children'),
              [Input('submit-button', 'n_clicks')],
              [Input('upload-image', 'contents')],
              [State('image-path-input', 'value')])
def update_original_image(n_clicks, contents, file_path):
    if n_clicks is not None:
        if contents is not None:
            image = read_image(contents)
            # Add your code here to perform magnification (if required)
            return html.Div([
                
                html.Img(src=contents, style={'paddingTop': '40px'}),
            ], style={'display': 'flex', 'justify-content': 'center'})
        elif file_path is not None:
            # read the image data using PIL
            image = Image.open(file_path)
            return html.Div([
                
                html.Img(src=image, style={'paddingTop': '40px'}),
            ], style={'display': 'flex', 'justify-content': 'center'})


# Callback to update the Clone Detection Plot with the user-defined tile and step sizes
@app.callback(Output('tab2-content', 'children'),
              [Input('submit-button', 'n_clicks')],
              [Input('upload-image', 'contents')],
              [State('image-path-input', 'value')],
               [Input('tile-size-slider', 'value')],
               [Input('step-size-slider', 'value')])
def update_clone_detection_plot(n_clicks,contents, file_path, tile_size, step_size):
    if n_clicks is not None:
        if contents is not None:
            image = read_image(contents)
        elif file_path is not None:
            # read the image data using PIL
            image = Image.open(file_path)
        else:
            # No image or file path provided
            return html.Div([
                html.P("Please upload an image or provide the file path."),
            ])

        clone_detection_plot = clone_detection_function(image, tile_size, step_size)
        return html.Div([
            dcc.Graph(figure=clone_detection_plot),
        ])


    

# Callback to update the Error Level Analysis Plot
@app.callback(Output('tab3-content', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('upload-image', 'contents'),
               State('image-path-input', 'value')])
def update_error_level_analysis_plot(n_clicks, contents, file_path):
    if contents is not None:
        image = read_image(contents)  
        error_level_analysis_plot = error_level_analysis_function(image)
        return html.Div([
            dcc.Graph(figure=error_level_analysis_plot),
        ])
    
    if n_clicks is not None:
        
        if file_path is not None:
            
            image = Image.open(file_path)
            image = np.array(image)[:, :, ::-1] 
            error_level_analysis_plot = error_level_analysis_function(image)
            return html.Div([
                dcc.Graph(figure=error_level_analysis_plot),
            ])
    else:
        return html.P(['Please upload an image or provide the file path.'])
    

# Callback to update the Noise Analysis Plot
@app.callback(Output('tab4-content', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('upload-image', 'contents'),
               State('image-path-input', 'value')])
def update_noise_analysis_plot(n_clicks,contents, file_path):
    if contents is not None:
        image = read_image(contents)
        noise_analysis_plot = noise_analysis_function(image)
        return html.Div([
            dcc.Graph(figure=noise_analysis_plot),
        ])
    if n_clicks is not None:
        if file_path is not None:
            
            image = Image.open(file_path)
            image = np.array(image)[:, :, ::-1]  
            noise_analysis_plot = noise_analysis_function(image)
            return html.Div([
                dcc.Graph(figure=noise_analysis_plot),
            ])
    else:
        return html.P(['Please upload an image or provide the file path.'])


# Callback to update the Level Sweep Plot
@app.callback(Output('tab5-content', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('upload-image', 'contents'),
               State('image-path-input', 'value'),
               State('levels-input', 'value'),
               State('opacities-input', 'value')])
def update_level_sweep_plot(n_clicks, contents, file_path, levels_input, opacities_input):
    if contents is not None:
        image = read_image(contents)
    elif file_path is not None:
       
        image = Image.open(file_path)
        image = np.array(image)[:, :, ::-1]  
    else:
        return html.P(['Please upload an image or provide the file path.'])

    try:
        levels = [float(level.strip()) for level in levels_input.split(',')]
        opacities = [float(opacity.strip()) for opacity in opacities_input.split(',')]
    except ValueError:
        return html.P(['Invalid input. Please enter valid levels and opacities.'])

    level_sweep_plot = level_sweep_function(image, levels, opacities)
    return html.Div([
        dcc.Graph(figure=level_sweep_plot),
    ], style={'Height': '100%', 'Width': '100%'})


# Callback to update the Luminance Gradient Plot
@app.callback(Output('tab6-content', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('upload-image', 'contents'),
               State('image-path-input', 'value')])
def update_luminance_gradient_plot(n_clicks,contents, file_path):
    if contents is not None:
        image = read_image(contents) 
        luminance_gradient_plot = luminance_gradient_function(image)
        return html.Div([
            dcc.Graph(figure=luminance_gradient_plot),
        ])

    if n_clicks is not None:
        if file_path is not None:
            
            image = Image.open(file_path)
            image = np.array(image)[:, :, ::-1]  
            luminance_gradient_plot = luminance_gradient_function(image)
            return html.Div([
                dcc.Graph(figure=luminance_gradient_plot),
            ])

# Callback to update the Geo and Meta Tags
@app.callback(Output('tab7-content', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('upload-image', 'contents'),
               State('image-path-input', 'value')])
def update_geo_meta_tags(n_clicks, contents, file_path):
    if contents is not None:
        image = Image.open(io.BytesIO(contents.encode('utf-8')))

    if n_clicks is not None:

        
        if file_path is not None:
            image = Image.open(file_path)
        else:
            return html.Div([
                html.H3("Geo and Meta Tags Analysis"),
                html.P("Please upload an image or provide the image path."),
            ])

        exifdata = image.getexif()
        tags_list = []
        data_list = []
        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            if isinstance(data, bytes):
                data = data.decode()
            tags_list.append(tag)
            data_list.append(data)
        tags_data_str = "\n".join([f"{tag:25}: {data}" for tag, data in zip(tags_list, data_list)])

        return html.Div([
            html.H3("Geo and Meta Tags Analysis"),
            html.Pre(tags_data_str),
        ])

        
# Callback to update the JPEG and String Analysis
@app.callback(Output('tab8-content', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('upload-image', 'contents'),
               State('image-path-input', 'value')])
def update_jpeg_string_analysis(n_clicks, contents, file_path):
    if not file_path and not contents:
        return html.Div("Please enter an image path.")
        
    if contents is not None:
        
        y_quantization_table, cbcr_quantization_table = get_quantization_tables(contents)
        if y_quantization_table is None or cbcr_quantization_table is None:
            return html.Div("Error: Image not found or unsupported format.")

        y_table_data = create_quantization_table_data(y_quantization_table)
        cbcr_table_data = create_quantization_table_data(cbcr_quantization_table)

        return html.Div([
            dcc.Tabs(id='tabs-example', value='tab-1-example', children=[
                dcc.Tab(
                    label='Y Quantization Table', 
                    value='tab-2-example',
                    children=[
                        html.Div(dash_table.DataTable(
                        id='y-table',
                        columns=[{'name': col, 'id': col} for col in y_table_data[0].keys()],
                        data=y_table_data,
                        style_table={'overflowX': 'auto'}))]
                        ),
                dcc.Tab(
                    label='CbCr Quantization Table', 
                    value='tab-3-example',
                    children=[
                        html.Div(dash_table.DataTable(
                        id='y-table',
                        columns=[{'name': col, 'id': col} for col in cbcr_table_data[0].keys()],
                        data=cbcr_table_data,
                        style_table={'overflowX': 'auto'}))]
                    ),
            ])
            
        ])
    
    
    elif n_clicks is not None:
            
        y_quantization_table, cbcr_quantization_table = get_quantization_tables(file_path)
        if y_quantization_table is None or cbcr_quantization_table is None:
            return html.Div("Error: Image not found or unsupported format.")

        y_table_data = create_quantization_table_data(y_quantization_table)
        cbcr_table_data = create_quantization_table_data(cbcr_quantization_table)

        return html.Div([
            html.H3("Y Quantization Table:"),
            dash_table.DataTable(
                id='y-table',
                columns=[{'name': col, 'id': col} for col in y_table_data[0].keys()],
                data=y_table_data,
                style_table={'overflowX': 'auto'},
            ),
            html.H3("CbCr Quantization Table:"),
            dash_table.DataTable(
                id='cbcr-table',
                columns=[{'name': col, 'id': col} for col in cbcr_table_data[0].keys()],
                data=cbcr_table_data,
                style_table={'overflowX': 'auto'},
            ),
        ])
    
def orb_detection_function(image_path):
    
    parser = argparse.ArgumentParser(description='Code for Feature Detection tutorial.')
    parser.add_argument('--input', help='Path to input image.', default=image_path)
    args = parser.parse_args()
    # Load the image for SURF keypoint detection
    src = cv.imread(cv.samples.findFile(image_path), cv.IMREAD_GRAYSCALE)
    if src is not None:
        # Detect ORB keypoints
        orb = cv.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(src, None)
        
        # Draw keypoints
        img_keypoints = cv.drawKeypoints(src, keypoints, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Convert to PIL Image for Dash display
        orb_img = Image.fromarray(img_keypoints)
        orb_fig = px.imshow(np.array(orb_img))
        orb_fig.update_layout(margin=dict(l=10, r=10, t=0, b=0))
    else:
        print ("Could not load the image")

    return orb_fig

@app.callback(
    [Output('tab9-content', 'children'),
     Output('feature-plot', 'figure')],
    [Input('submit-button', 'n_clicks')],
    [State('upload-image', 'contents'),
     State('image-path-input', 'value')]
)
def update_orb_detection_plot(n_clicks, uploaded_contents, image_path):
    if image_path is not None:
        orb_detection_plot = orb_detection_function(image_path)
        return html.Div([
            dcc.Graph(figure=orb_detection_plot),
        ]), orb_detection_plot
    else:
        return html.P(['Please upload an image or provide the file path for the analysis.']), dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)
