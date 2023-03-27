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
            'rayleigh Noise','speckle_noise','uniform_noise'], 
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

    children=['Please upload images upto 10kb by compressing them so not to face any upload size limit issues while using dash core components for the same...']
    if image_path is None:
            raise dash.exceptions.PreventUpdate

    else:   

            img = image_string_to_PILImage(image_path)
            float_arr = np.array(img)
            uint_img = np.array(float_arr*255).astype('uint8')
            image_path = cv2.cvtColor(uint_img, cv2.COLOR_RGB2GRAY)
            if selected_noise == 'lognormal Noise':
                children+=[html.Br(),'lognormal Noise',html.Br()]
                lognormal_noise_imgs = lognormal_noise(image_path)   
                number_of_white_pix = [np.sum(img == 255) for img in lognormal_noise_imgs]
                img_srces = [array_to_data_url((lognormal_noise_img).astype(np.uint8)) for lognormal_noise_img in lognormal_noise_imgs]
                my_list = get_pixelated_components(img_srces,number_of_white_pix)
                children+=my_list
            if selected_noise == 'salt and pepper Noise':

                children+=[html.Br(),'salt and pepper Noise',html.Br()]
                saltandpepper_noise_imgs = saltandpepper_noise(image_path)   
                number_of_white_pix = [np.sum(img == 255) for img in saltandpepper_noise_imgs]
                img_srces = [array_to_data_url((saltandpepper_noise_img).astype(np.uint8)) for saltandpepper_noise_img in saltandpepper_noise_imgs]
                my_list = get_pixelated_components(img_srces,number_of_white_pix)
                children+=my_list

            if selected_noise == 'erlang Noise':    

                children+=[html.Br(),'erlang Noise',html.Br()]
                erlang_noise_imgs = erlang_noise(image_path)   
                number_of_white_pix = [np.sum(img == 255) for img in erlang_noise_imgs]
                img_srces = [array_to_data_url((erlang_noise_img).astype(np.uint8)) for erlang_noise_img in erlang_noise_imgs]
                my_list = get_pixelated_components(img_srces,number_of_white_pix)
                children+=my_list


            if selected_noise == 'exponential Noise':
                children+=[html.Br(),'exponential Noise',html.Br()]
                exponential_noise_imgs = exponential_noise(image_path)    
                number_of_white_pix = [np.sum(img == 255) for img in exponential_noise_imgs]
                img_srces = [array_to_data_url((exponential_noise_img).astype(np.uint8)) for exponential_noise_img in exponential_noise_imgs]
                my_list = get_pixelated_components(img_srces,number_of_white_pix)
                children+=my_list
            if selected_noise == 'gaussian Noise':
                children+=[html.Br(),'gaussian Noise',html.Br()]
                gaussian_noise_imgs = gaussian_noise(image_path)    
                number_of_white_pix = [np.sum(img == 255) for img in gaussian_noise_imgs]
                img_srces = [array_to_data_url((gaussian_noise_img).astype(np.uint8)) for gaussian_noise_img in gaussian_noise_imgs]
                my_list = get_pixelated_components(img_srces,number_of_white_pix)
                    
                children+=my_list
            if selected_noise == 'poisson Noise':
                children+=[html.Br(),'poisson Noise',html.Br()]
                poisson_noise_imgs = poisson_noise(image_path)    
                number_of_white_pix = [np.sum(img == 255) for img in poisson_noise_imgs]
                img_srces = [array_to_data_url((poisson_noise_img).astype(np.uint8)) for poisson_noise_img in poisson_noise_imgs]
                my_list = get_pixelated_components(img_srces,number_of_white_pix)
                    
                children+=my_list
            if selected_noise == 'rayleigh Noise':
                children+=[html.Br(),'rayleigh Noise',html.Br()]
                rayleigh_noise_imgs = rayleigh_noise(image_path)  
                number_of_white_pix = [np.sum(img == 255) for img in rayleigh_noise_imgs]
                img_srces = [array_to_data_url((rayleigh_noise_img).astype(np.uint8)) for rayleigh_noise_img in rayleigh_noise_imgs]
                my_list = get_pixelated_components(img_srces,number_of_white_pix)
                    
                children+=my_list
            if selected_noise == 'speckle_noise':
                children+=[html.Br(),'speckle_noise',html.Br()]
                speckle_noise_imgs = speckle_noise(image_path)    
                number_of_white_pix = [np.sum(img == 255) for img in speckle_noise_imgs]
                img_srces = [array_to_data_url((speckle_noise_img).astype(np.uint8)) for speckle_noise_img in speckle_noise_imgs]
                my_list = get_pixelated_components(img_srces,number_of_white_pix)
                    
                children+=my_list
            if selected_noise == 'uniform_noise':
                children+=[html.Br(),'uniform_noise',html.Br()]
                uniform_noise_imgs = uniform_noise(image_path)    
                number_of_white_pix = [np.sum(img == 255) for img in uniform_noise_imgs]
                img_srces = [array_to_data_url((uniform_noise_img).astype(np.uint8)) for uniform_noise_img in uniform_noise_imgs]
                my_list = get_pixelated_components(img_srces,number_of_white_pix)
                children+=my_list
            

    return children

        

server = app.server
if __name__ == '__main__':
    # app.run_server(debug=True, host='0.0.0.0', port=8080)
    app.run_server(debug=True)
    