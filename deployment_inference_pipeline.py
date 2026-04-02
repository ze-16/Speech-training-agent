import streamlit as st # pyright: ignore[reportMissingImports]
from speech_training_agent import BERTClassifier, FusionModel, Wav2Model
import torch # type: ignore
import torchaudio # type: ignore
import whisper # type: ignore
from transformers import BertTokenizer, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor # type: ignore
import tempfile

@st.cache_resource()
def load_models():
    state_dict = torch.load('model_weights/best_model_weights.pth', map_location=torch.device('cpu'))
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    state_dict_v = torch.load('model_weights/best_model_weights_0.7007.pth', map_location=torch.device('cpu'))
    new_state_dict_v = {k.replace('_orig_mod.', ''): v for k, v in state_dict_v.items()}


    bert_token = BertTokenizer.from_pretrained('bert-base-uncased')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
    transcribe = whisper.load_model("base")

    bert_model = BERTClassifier(n_classes=4)
    bert_model.load_state_dict(new_state_dict)
    bert_model.eval()

    wav_for_fusion = Wav2Model()
    wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained('superb/wav2vec2-base-superb-er', num_labels=4)
    wav2vec_model.eval()

    fusion_model = FusionModel(wav_for_fusion, bert_model)
    fusion_model.load_state_dict(new_state_dict_v)
    fusion_model.eval()


    return bert_token, feature_extractor, transcribe, bert_model, wav2vec_model, fusion_model

bert_token, feature_extractor, transcribe, bert_model, wav2vec_model, fusion_model = load_models()



audio = st.audio_input("Upload an audio file", sample_rate=16000)

if audio is not None:
    st.write("Transcribing audio")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio.read())
        tmp_file_path = tmp_file.name
    
    transcript = transcribe.transcribe(tmp_file_path)
    st.write("Transcription:")
    st.write(transcript['text'])

    audio_waveform, sample_rate = torchaudio.load(tmp_file_path)
    st.write("Extracting features")
    features = feature_extractor(audio_waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt")

    tokenize = bert_token(transcript['text'], padding='max_length',return_attention_mask=True, add_special_tokens=True, truncation=True, max_length=128, return_tensors="pt")

    with torch.no_grad():
        wav2vec_output = wav2vec_model(features['input_values']).logits
        wav2vec_output = torch.argmax(wav2vec_output, dim=1).item()

        bert_output = bert_model(tokenize['input_ids'], tokenize['attention_mask'])
        bert_output = torch.argmax(bert_output, dim=1).item()

        fusion_output = fusion_model(features['input_values'], tokenize['input_ids'], tokenize['attention_mask'])
        fusion_output = torch.argmax(fusion_output, dim=1).item()

    st.write(f"Wav2Vec2 Prediction: {wav2vec_output}")
    st.write(f"BERT Prediction: {bert_output}")
    st.write(f"Fusion Model Prediction: {fusion_output}")


