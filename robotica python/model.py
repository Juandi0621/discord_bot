from keras.models import load_models
from Pil import Image, ImageOps
import numpy as np

def get_class(model_path, labels_path, image part):
    np.set_printoptions(suppress= True)
    model = load_model(model_path, compile=False)
    class_names = open(labels_path, "r",encoding="utf-8").readlines()
    data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")

    size = (224, 224)

    image = ImageOps.fit(image,size,Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5)
    data[0] = normalized_image_array

    predection = model.predict(data)
    index = np.argmax(predection)
    class_name = class_names[index]
    confidence_score = predection[0][index]

    return (class_name[2:], confidence_score)