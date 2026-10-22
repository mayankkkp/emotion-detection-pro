# inference.py
import json
import numpy as np
from keras.layers import Dense, InputLayer
from keras.models import model_from_json as _original_model_from_json
import config

# ==========================================
# KERAS COMPATIBILITY PATCH
# Strips unrecognized parameters from newer Keras models
# ==========================================
_original_dense_init = Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return _original_dense_init(self, *args, **kwargs)
Dense.__init__ = _patched_dense_init

_original_input_init = InputLayer.__init__
def _patched_input_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    kwargs.pop('optional', None)
    return _original_input_init(self, *args, **kwargs)
InputLayer.__init__ = _patched_input_init

def model_from_json(json_string, custom_objects=None):
    cfg = json.loads(json_string)
    def clean_config(obj):
        if isinstance(obj, dict):
            obj.pop('quantization_config', None)
            obj.pop('batch_shape', None)
            obj.pop('optional', None)
            if 'dtype' in obj and isinstance(obj['dtype'], dict):
                dtype_info = obj['dtype']
                if 'config' in dtype_info and 'name' in dtype_info['config']:
                    obj['dtype'] = dtype_info['config']['name']
                elif 'name' in dtype_info:
                    obj['dtype'] = dtype_info['name']
                else:
                    obj['dtype'] = 'float32'
            if 'class_name' in obj and obj['class_name'] == 'InputLayer' and 'config' in obj:
                if 'batch_input_shape' not in obj['config'] and 'batch_shape' in obj['config']:
                    obj['config']['batch_input_shape'] = obj['config']['batch_shape']
            for k, v in obj.items():
                clean_config(v)
        elif isinstance(obj, list):
            for item in obj:
                clean_config(item)
    clean_config(cfg)
    return _original_model_from_json(json.dumps(cfg), custom_objects)

# ==========================================
# MODEL LOADING
# ==========================================
print("Loading model...")
with open(config.MODEL_JSON_PATH, "r") as json_file:
    model_json = json_file.read()
    
model = model_from_json(model_json)
model.load_weights(config.MODEL_WEIGHTS_PATH)
print("Model loaded successfully.")

def predict_emotion(face_roi):
    """
    Takes a preprocessed 48x48 face array, expands dimensions,
    and runs model prediction.
    """
    # Reshape for model: (1, 48, 48, 1)
    input_data = np.expand_dims(face_roi, axis=-1)
    input_data = np.expand_dims(input_data, axis=0)
    
    # Predict
    preds = model.predict(input_data, verbose=0)
    
    # Get highest probability label
    best_class_index = np.argmax(preds)
    emotion = config.EMOTION_LABELS[best_class_index]
    confidence = float(np.max(preds))
    
    return emotion, confidence