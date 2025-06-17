import tensorflow as tf

def save_model(model, keras_filename="automobile.keras", tflite_filename="automobile.tflite"):
    tf.keras.models.save_model(model, keras_filename)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
