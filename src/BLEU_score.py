from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_bleu_score(test_df, tokenizer, model, features, max_length):
    references = []
    predictions = []

    for index, record in test_df.iterrows():
        # Predict the caption for the image
        image = record['image']
        predicted_caption = predict_caption(model, image, tokenizer, max_length, features)

        # Get the ground truth captions (all captions for that image)
        ground_truth_captions = record['caption'].split(" endseq")[0].replace("startseq", "").strip().split()

        # Add reference (ground truth) and prediction to the lists
        references.append([ground_truth_captions])
        predictions.append(predicted_caption.split())


    smoothing_function = SmoothingFunction().method4
    bleu_score = corpus_bleu(references, predictions, smoothing_function=smoothing_function)

    return bleu_score

# Compute BLEU score for the test set
bleu = compute_bleu_score(samples, tokenizer, caption_model, features, max_length)
print(f'BLEU score: {bleu:.4f}')

