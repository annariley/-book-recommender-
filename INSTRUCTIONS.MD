###Running the Web App

To run the web app first make sure you are in the webapp/book-recommender directory.
Here run npm install to install the required modules for the front-end. You may also need to
run the following commands as well:
npm install react-scripts start
npm install react-tinder-card
npm install @material-ui/core
npm install @material-ui/icons

For the backend you will need to install the the following python modules (listed below as the pip command to run):
pip install flask
pip install flask-restful
pip install flask-pymongo
pip install torch
pip install pandas
pip install dnspython
pip install pymongo
pip install numpy

Now you can start the backend by calling python api.py, and start the front end by calling npm start
If you run into any issues running these you may be missing a package, check the error you received, and if you are missing a package install it using similar commands as above.

###Understanding the file structure of this repo:
webapp/book-recommender: This folder contains all the files needed for the webapps operation
cleanup_dataset.py: This is a separate script that was used to upload the books to MongoDB - do not run it