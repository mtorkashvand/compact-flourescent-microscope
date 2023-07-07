import os
import onnxruntime

def load_ort(model_filename):
    return(onnxruntime.InferenceSession(
        os.path.join(
        os.path.split(os.path.abspath(__file__))[0], model_filename)
        )
    )