from flask import *

#importing style transfer function from main file
from caption_model import caption_generator

import os #for interacing with os
app = Flask(__name__)  #creating the Flask class object   

# force browser to hold no cache. Otherwise old result might return.
#@app.after_request
#def set_response_headers(response):
#    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
#    response.headers['Pragma'] = 'no-cache'
#    response.headers['Expires'] = '0'
#    return response

#the / URL is bound to the main function which is responsible for returning the server response. 
#It can return a string to be printed on the browser's window or we can use the HTML template to return the HTML file as a response from the server. 
@app.route('/')  #The route() function of the Flask class defines the URL mapping of the associated function
def upload():  
    return render_template("webpage.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        #getting both the images from webpage
        style_file = request.form['submit_button']
        if(style_file == 'Caption1'):
            caption_generator('./static/a.jpg')
        if(style_file == 'Caption2'):
            caption_generator('./static/b.jpg')
        if(style_file == 'Caption3'):
            caption_generator('./static/c.jpg')
        if(style_file == 'Caption4'):
            caption_generator('./static/d.jpg')
        if(style_file == 'Caption5'):
            caption_generator('./static/e.jpg')
        if(style_file == 'Caption6'):
            caption_generator('./static/f.jpg')
        if(style_file == 'Caption7'):
            caption_generator('./static/g.jpg')
        if(style_file == 'Caption8'):
            caption_generator('./static/h.jpg')
        
        #saving the images in specified path
        #style_file.save(os.path.join('./static/io','a.jpg'))
  
        return render_template("webpage.html")  

if __name__ == '__main__':  #We need to pass the name of the current module, i.e. __name__ as the argument into the Flask constructor
    app.run(debug = True)  #Finally, the run method of the Flask class is used to run the flask application on the local development server.