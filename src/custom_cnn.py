import pickle

tokenizer_path = '/content/drive/MyDrive/WORKING_DIR/tokenizer_c3.pkl'

# Save the tokenizer to a .pkl file
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"Tokenizer saved to {tokenizer_path}")


custom_cnn = build_custom_cnn()
img_size = 224

# Load and preprocess each image
features = {}
for image in tqdm(data['image'].unique().tolist()):
    img = load_img(os.path.join(image_path, image), target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Get features from your CNN
    feature = custom_cnn.predict(img, verbose=0)
    features[image] = feature[0]
