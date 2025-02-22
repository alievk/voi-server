import io
import base64
from PIL import Image


def blob_to_image(blob):
    prefix = b"data:image/png;base64,"
    data = blob[len(prefix):]
    data = base64.b64decode(data)
    image = Image.open(io.BytesIO(data))
    return image


def image_to_openai_url(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        data = base64.b64encode(data)
        return f"data:image/png;base64,{data.decode('utf-8')}"


def blob_to_openai_url(blob):
    return image_to_openai_url(blob_to_image(blob))
