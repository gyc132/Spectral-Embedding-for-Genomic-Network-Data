import mira

def train_topic_predict(atac_data, save_path):
    example_atac_model = mira.topics.make_model(
        feature_type = 'accessibility',
        n_samples = atac_data.shape[0],
        n_features = atac_data.shape[1],
    ).fit(atac_data)

    example_atac_model.save(save_path)

    return example_atac_model