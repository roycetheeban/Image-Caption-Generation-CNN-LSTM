def predict_caption(model, image, tokenizer, max_length, features):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        # Predict next word
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)

        # Convert predicted index to word
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break

        # Append word to the input text
        in_text += ' ' + word

        if word == 'endseq':
            break

    return in_text.split(' ')[1:-1]  # remove 'startseq' and 'endseq'

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def generate_caption_for_new_image(image_path, model, tokenizer, max_length, cnn_model):
    #Load and preprocess the new image (resize to 224x224)
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    # Extract features using custom CNN
    image_features = cnn_model.predict(img)

    #  Generate the caption
    caption = predict_caption(model, image_path, tokenizer, max_length, image_features)

    # Display the image and caption
    plt.figure(figsize=(6, 6))
    plt.imshow(img[0])
    plt.axis('off')
    plt.title(' '.join(caption))
    plt.show()

# Example usage:
image_path = '/content/bigstock-Kids-Play-Football-Cute-Littl-471646067.jpg'
generate_caption_for_new_image(image_path, caption_model, tokenizer, max_length, custom_cnn)
