import datetime
import os
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from erlang_noise_done import erlang_noise
from exponential_noise_done import exponential_noise
from gaussian_noise_done import gaussian_noise
from poisson_noise_done import poisson_noise
from rayleigh_noise_done import rayleigh_noise
from speckle_noise_done import speckle_noise
from uniform_noise_done import uniform_noise
from dash_canvas import DashCanvas
from dash_canvas.utils import array_to_data_url, parse_jsonstring
import numpy as np
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(


    [html.Div('Enter image path '),
    dcc.Input(id = 'image_path'),
    html.Div(id='output-image-upload'),]
)

@app.callback(Output('output-image-upload', 'children'),
              Input('image_path','value'))
def update_output(image_path):
    def get_pixelated_components(img_srces,number_of_white_pix):
        rv  = [
        html.Div(
            [
                html.H2(
                    f'White Pixel Count = {pix}',
                    id='title'
                ),
                html.Img(src=img_src)
            ]) 
        for img_src,pix in zip(img_srces,number_of_white_pix)]
        
        return rv

    children=[]
    if image_path!=None and os.path.exists(image_path):
        children+=[html.Br(),'erlang Noise',html.Br()]
        erlang_noise_imgs = erlang_noise(image_path)   
        number_of_white_pix = [np.sum(img == 255) for img in erlang_noise_imgs]
        img_srces = [array_to_data_url((erlang_noise_img).astype(np.uint8)) for erlang_noise_img in erlang_noise_imgs]
        my_list = get_pixelated_components(img_srces,number_of_white_pix)
        children+=my_list
        children+=[html.Br(),'exponential Noise',html.Br()]
        exponential_noise_imgs = exponential_noise(image_path)    
        number_of_white_pix = [np.sum(img == 255) for img in exponential_noise_imgs]
        img_srces = [array_to_data_url((exponential_noise_img).astype(np.uint8)) for exponential_noise_img in exponential_noise_imgs]
        my_list = get_pixelated_components(img_srces,number_of_white_pix)
        
        children+=my_list
        children+=[html.Br(),'gaussian Noise',html.Br()]
        gaussian_noise_imgs = gaussian_noise(image_path)    
        number_of_white_pix = [np.sum(img == 255) for img in gaussian_noise_imgs]
        img_srces = [array_to_data_url((gaussian_noise_img).astype(np.uint8)) for gaussian_noise_img in exponential_noise_imgs]
        my_list = get_pixelated_components(img_srces,number_of_white_pix)
        
        children+=my_list
        children+=[html.Br(),'poisson Noise',html.Br()]
        poisson_noise_imgs = poisson_noise(image_path)    
        number_of_white_pix = [np.sum(img == 255) for img in poisson_noise_imgs]
        img_srces = [array_to_data_url((poisson_noise_img).astype(np.uint8)) for poisson_noise_img in poisson_noise_imgs]
        my_list = get_pixelated_components(img_srces,number_of_white_pix)
        
        children+=my_list
        children+=[html.Br(),'rayleigh Noise',html.Br()]
        rayleigh_noise_imgs = rayleigh_noise(image_path)  
        number_of_white_pix = [np.sum(img == 255) for img in rayleigh_noise_imgs]
        img_srces = [array_to_data_url((rayleigh_noise_img).astype(np.uint8)) for rayleigh_noise_img in rayleigh_noise_imgs]
        my_list = get_pixelated_components(img_srces,number_of_white_pix)
        
        children+=my_list
        children+=[html.Br(),'speckle_noise',html.Br()]
        speckle_noise_imgs = speckle_noise(image_path)    
        number_of_white_pix = [np.sum(img == 255) for img in speckle_noise_imgs]
        img_srces = [array_to_data_url((speckle_noise_img).astype(np.uint8)) for speckle_noise_img in speckle_noise_imgs]
        my_list = get_pixelated_components(img_srces,number_of_white_pix)
        
        children+=my_list
        children+=[html.Br(),'uniform_noise',html.Br()]
        uniform_noise_imgs = uniform_noise(image_path)    
        number_of_white_pix = [np.sum(img == 255) for img in uniform_noise_imgs]
        img_srces = [array_to_data_url((uniform_noise_img).astype(np.uint8)) for uniform_noise_img in uniform_noise_imgs]
        my_list = get_pixelated_components(img_srces,number_of_white_pix)
        children+=my_list









    return children
if __name__ == '__main__':
    app.run_server(debug=True)