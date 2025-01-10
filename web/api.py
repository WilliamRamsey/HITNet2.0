import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.datastructures import MultiDict

# FLASK CONFIGURATION
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "C:/Users/willi/OneDrive/Desktop/HITNET/web/video_upload"
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = "Superdoopersecretkey"

# FILE UPLOAD PAGE
@app.route("/upload_video", methods=["POST", "GET"])
def upload_video():

    # receiving posted files
    if request.method == "POST":

        # checks if post includes correct form attribute
        if "file" not in request.files:
            flash("FILENAME NOT INCLUDED IN POST")
            # if file upload fails reloads webpage
            return redirect(request.url)
        file = request.files["file"]
        
        # checks if file was selected
        if file.filename == "":
            flash("PLEASE SELECT A FILE")
            return redirect(request.url)
        
        # checks if file object exists and if file extension is valid
        if file and file.filename.split(".")[1] == "mp4":
            # sanitized filename
            filename = secure_filename(file.filename)
            # saves file to downloads directory
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            # send to annotation page
            return "Good job"

        print(request.files["file"])

    # redirect to origional web
    with open("web/upload_video.html") as file:
        return file.read()

# ANNOTATIONS PAGE
@app.route("/annotate", methods=["POST", "GET"])
def get_annotations():
    
    # Read Track ID form
    if request.method == "POST":
        pass # USE FORM ELEMENT
    
    return

# Model returns first unassigned frame
# for frame in frames:
#   if assigned_label not in labels in frames:
#       prompt user to select segmentation
#       returns selected id
# iterate to next frame

# User selects and labels segmentations
# User selection coorisponds particular segmentation ID to

# Advances until next unknown moment


app.run(host="0.0.0.0", port=5500)