import requests

# URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/process_video/"
# Path to the video file to be uploaded
video_file_path = "/home/sayandebroy/codings/depth/resources/racing.mp4" # Replace with the path to your video file
# Path to save the output depth video
output_file_path = "output_depth_video.avi"

# Open the video file in binary mode
with open(video_file_path, 'rb') as video_file:
    # Prepare the files dictionary for the POST request
    files = {'file': video_file}
    
    # Send the POST request to the API endpoint
    response = requests.post(url, files=files)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Save the returned depth video
        with open(output_file_path, 'wb') as output_file:
            output_file.write(response.content)
        print(f"Depth map video saved as {output_file_path}")
    else:
        print(f"Failed to process video. Status code: {response.status_code}, Response: {response.text}")
