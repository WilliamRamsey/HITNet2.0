from flask import Flask, request

app = Flask(__name__)


@app.route('/upload_video_form', methods=['GET'])
def upload_video_form():
    with open("upload_video_form.html", "r") as file:
        return file.read()

@app.route('/upload_video_form.css', methods=["GET"])
def upload_video_form_css():
    with open("upload_video_form.css", "r") as file:
        return file.read()

@app.route('/API/video_analysis_request', methods=['POST'])
def video_analysis_request():
    # Process the incoming POST data here
    print(request.content_type)
    print(request.form.get("name"))
    print(request.form.get("email"))
    print(request.form.get("phone"))
    video_file = request.files['video']
    video_file.save("video.mp4")
    return "Form Received. Thank you! Expect to hear from williamdawsonramsey@gmail.com within 24 hours.", 200


if __name__ == '__main__':
    app.run(debug=True)