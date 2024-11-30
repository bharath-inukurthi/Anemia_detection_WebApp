import gradio as gr
import requests as r
from io import BytesIO
from PIL import Image

def numpy_array_to_bytes(np_array, format='png'):

    img = Image.fromarray(np_array)
    with BytesIO() as output:
        img.save(output, format=format)
        img_bytes = output.getvalue()
    return img_bytes


def call_api(img):
    img_data = {"file": ("image.jpg", img, "image/jpeg")}
    response = r.post("http://127.0.0.1:8000/predict/", files=img_data)

    if response.status_code == 200:
        hgl = response.json().get("hgl")
        status = response.json().get("status")
        result = f"{hgl}"
        return result, status
    else:
        return "Error: Unable to process the image", "Error"


def process_image_from_upload(image):
    if image is not None:

        img_bytes = numpy_array_to_bytes(image)
        return call_api(img_bytes)
    return "No image uploaded", "Error"


def process_image_from_camera(image):
    if image is not None:

        img_bytes = numpy_array_to_bytes(image)
        return call_api(img_bytes)
    return "No image captured", "Error"


upload_interface = gr.Interface(
    fn=process_image_from_upload,
    inputs=gr.Image(label="Upload conjunctiva image"),
    outputs=[gr.Label(label="Himoglobin Levels"),gr.Label(label="You are")],
    title="Anemia Detector"
)

camera_interface = gr.Interface(
    fn=process_image_from_camera,
    inputs=gr.Image(
    label="Capture when conjunctiva is visible\nNOTE: conjunctiva is red part under down eyelid"),
    outputs=[gr.Label(label="Himoglobin Levels"),gr.Label(label="You are")]
    #placeholder="if using Webcam Click CAMERA icon appears below to capture image\nNOTE:After clicking capturing may take sometime"
)

gr.TabbedInterface([upload_interface, camera_interface], ["Upload Image", "Take Photo"]).launch(share=True)
