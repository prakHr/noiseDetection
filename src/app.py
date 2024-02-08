import datetime
import os
import dash
import cv2
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from erlang_noise_done import erlang_noise
from exponential_noise_done import exponential_noise
from gaussian_noise_done import gaussian_noise
from poisson_noise_done import poisson_noise
from rayleigh_noise_done import rayleigh_noise
from speckle_noise_done import speckle_noise
from uniform_noise_done import uniform_noise
from saltandpepper_noise_done import saltandpepper_noise
from lognormal_noise_done import lognormal_noise
from dash_canvas import DashCanvas
from dash_canvas.utils import array_to_data_url, parse_jsonstring, image_string_to_PILImage
import numpy as np



from dash import Dash, page_container, dcc, clientside_callback, ClientsideFunction, Output, Input

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app =  Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(


    [
        html.Div('Please upload images upto 10kb by compressing them so not to face any upload size limit issues while using dash core components for the same...'),
        html.Div('Upload image'),
        dcc.Upload(
        id='image_path',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    dcc.Dropdown(['lognormal Noise', 'salt and pepper Noise', 'erlang Noise',
            'exponential Noise','gaussian Noise','poisson Noise',
            'rayleigh Noise','speckle Noise','uniform Noise'], 
            'lognormal Noise', id='demo-dropdown'),
        html.Div(id='output-image-upload'),

    ]
)

@app.callback(Output('output-image-upload', 'children'),
    Input('image_path', 'contents'),
    State('image_path', 'filename'),
    State('image_path', 'last_modified'),
    Input('demo-dropdown', 'value'))
def update_output(image_path,names,dates,selected_noise):
    def get_pixelated_components(img_srces,number_of_white_pix):
        img_srces_new,number_of_white_pix_new = [],[]
        minima = pow(10,64)-1
        for i,pix in enumerate(number_of_white_pix):
            if pix>0:
                minima = min(minima,pix)
        for i,pix in enumerate(number_of_white_pix):
            if pix==minima:
                img_srces_new.append(img_srces[i])
                number_of_white_pix_new.append(number_of_white_pix[i])
                break
        img_srces,number_of_white_pix=img_srces_new,number_of_white_pix_new
        rv  = [
                html.Div(
                    [
                        html.H2(
                            f'White Pixel Count = {pix}',
                            id='title'
                        ),
                        html.Img(src=img_src)
                    ]) for img_src,pix in zip(img_srces,number_of_white_pix)]
            
        return rv

    children=[]
    if image_path is None:
            raise dash.exceptions.PreventUpdate

    else:   

            img = image_string_to_PILImage(image_path)
            float_arr = np.array(img)
            uint_img = np.array(float_arr*255).astype('uint8')
            image_path = cv2.cvtColor(uint_img, cv2.COLOR_RGB2GRAY)
            noise_dict = {
                'lognormal Noise':'lognormal_noise(image_path)',
                'salt and pepper Noise':'saltandpepper_noise(image_path)',
                'exponential Noise':'exponential_noise(image_path)',
                'gaussian Noise':'gaussian_noise(image_path)',
                'poisson Noise':'poisson_noise(image_path)',
                'rayleigh Noise':'rayleigh_noise(image_path)',
                'speckle Noise':'speckle_noise(image_path)',
                'uniform Noise':'uniform_noise(image_path)'



            }
            for k,v in noise_dict.items():
                if selected_noise == k:

                    children+=[html.Br(),k,html.Br()]
                    lognormal_noise_imgs = eval(v)   
                    number_of_white_pix = [np.sum(img == 255) for img in lognormal_noise_imgs]
                    img_srces = [array_to_data_url((lognormal_noise_img).astype(np.uint8)) for lognormal_noise_img in lognormal_noise_imgs]
                    my_list = get_pixelated_components(img_srces,number_of_white_pix)
                    children+=my_list
            

    return children

        

server = app.server
if __name__ == '__main__':
    # app.run_server(debug=True, host='0.0.0.0', port=8080)
    app.run_server(debug=True)
    