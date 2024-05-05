from bottle import Bottle, request, response, run, template
import base64
import json

from os.path import join as pjoin
import cv2
import os
import numpy as np


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))
    
def resize_height_by_longest_edge_image_bytes(image_bytes, resize_length=800):
    # Convert the raw image data into a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode the numpy array into an OpenCV image (BGR format)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = cv2_img.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))

app = Bottle()

@app.route('/')
def index():
    # Render HTML form for uploading an image
    return '''
        <form action="/infer" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*">
            <input type="submit" value="Upload Image">
        </form>
        <br>
        <form action="/render" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*">
            <input type="submit" value="Render Image">
        </form>
    '''

@app.route('/infer', method='POST')
def infer():
    try:
        # Check if the request contains a file
        image_data = request.files.get('image')
        
        if not image_data:
            response.status = 400
            return {'error': 'No image sent'}
        
        # Set the correct content type to render the image in the browser
        response.content_type = image_data.content_type
        
        response_data, _ = get_rendered_image(image_data.file.read())

        # Return JSON response
        response.content_type = 'application/json'
        return json.dumps(response_data)
    
    except Exception as e:
        response.status = 500
        return {'error': f'Internal Server Error: {str(e)}'}


@app.route('/render', method='POST')
def render():
    try:
        # Check if the request contains a file
        image_data = request.files.get('image')
        
        if not image_data:
            response.status = 400
            return {'error': 'No image sent'}
        
        # Set the correct content type to render the image in the browser
        response.content_type = image_data.content_type
        
        _, rendered_image = get_rendered_image(image_data.file.read())

        return rendered_image
    
    except Exception as e:
        response.status = 500
        return {'error': f'Internal Server Error: {str(e)}'}


def get_rendered_image(raw_image):
    '''
        ele:min-grad: gradient threshold to produce binary map         
        ele:ffl-block: fill-flood threshold
        ele:min-ele-area: minimum area for selected elements 
        ele:merge-contained-ele: if True, merge elements contained in others
        text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
        text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

        Tips:
        1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
        2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
        3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
        4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

        mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':50, 'max-word-inline-gap':6, 'max-line-gap':1}
        web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}
    '''
    key_params = {'min-grad':25, 'ffl-block':5, 'min-ele-area':50,
                  'merge-contained-ele':False, 'merge-line-to-paragraph':False, 'remove-bar':False}

    # resized_height = resize_height_by_longest_edge(input_path_img, resize_length=800)
    resized_height = resize_height_by_longest_edge_image_bytes(raw_image, resize_length=1366)

    is_ip = True
    result_img = None
    uicompos = None

    if is_ip:
        import detect_compo.ip_region_proposal as ip
        uicompos, result_img = ip.compo_detection_img_bytes(raw_image, key_params,
                           classifier=None, resize_by_height=resized_height, show=False)
    
    img_bytes = None

    if result_img is not None:
        # Encode the image as JPEG
        _, img_encoded = cv2.imencode('.jpg', result_img)

        # Convert the image bytes to raw bytes
        img_bytes = img_encoded.tobytes()

    return uicompos, img_bytes


if __name__ == '__main__':
    # Run the Bottle app
    run(app, host='localhost', port=8080)
