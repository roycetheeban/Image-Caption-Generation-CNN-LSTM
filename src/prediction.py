samples = test.sample(15)
samples.reset_index(drop=True,inplace=True)

for index,record in samples.iterrows():

    img = load_img(os.path.join(image_path,record['image']),target_size=(224,224))
    img = img_to_array(img)
    img = img/255.

    caption = predict_caption(caption_model, record['image'], tokenizer, max_length, features)
    samples.loc[index,'caption'] = caption

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

for i in range(len(samples)):
    image_file = samples.loc[i, 'image']
    caption = samples.loc[i, 'caption']

    # Load image
    img_path = os.path.join(image_path, image_file)
    img = load_img(img_path, target_size=(224, 224))

    # Plot
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption)
    plt.show()
