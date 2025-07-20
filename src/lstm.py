from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, concatenate,
    Reshape, Bidirectional, BatchNormalization, add, Dense
)

# Inputs
input1 = Input(shape=(256,))  # Image feature from your custom CNN
input2 = Input(shape=(max_length,))  # Partial caption

# Caption embedding
sentence_features = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=False)(input2)

# Reshape image to match LSTM input format (as 1 time step)
img_features_reshaped = Reshape((1, 256))(input1)

# Concatenate image as first time step
merged = concatenate([img_features_reshaped, sentence_features], axis=1)

# Bidirectional LSTM with 128 units (output = 256 dims total)
sentence_features = Bidirectional(LSTM(128))(merged)  # Output shape = (256,)

# Dropout + batch norm
x = Dropout(0.5)(sentence_features)
x = BatchNormalization()(x)

# Add skip connection with original image feature
x = add([x, input1])

# Dense transformation
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

# Final prediction
output = Dense(vocab_size, activation='softmax')(x)

# Build and compile model
caption_model = Model(inputs=[input1, input2], outputs=output)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
