# Loading dependancies 
import os
import streamlit as st
from speech_training_agent import BERTClassifier, FusionModel, Wav2Model
import torch
import librosa
import whisper
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


audio = None
# Consent checkbox to comply with ethical project objectives
st.checkbox("Please check the box in order to give consent for the models to load and process your audio",
            value=False, key="consent_checkbox")

if st.session_state.consent_checkbox == True:
    # Capturing audio input from the user
    audio = st.audio_input("Upload an audio file", sample_rate=16000)
else:
    st.write("Please check the box to give consent for the models to load and process your audio.")



if audio is not None:
    # Transcribing the audio using Whisper
    st.write("Transcribing audio")
    tmp_file_path = None
    # File deletion to comply with project ethical objectives
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio.read())
            tmp_file_path = tmp_file.name
        
        transcript = transcribe.transcribe(tmp_file_path)

        st.write("Transcription:")
        st.write(transcript['text'])

        audio_waveform, sample_rate = librosa.load(tmp_file_path, sr=16000)
    finally:
        if tmp_file_path is not None and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


    # Tokenizing and extracting features from the audio
    st.write("Extracting features")
    features = feature_extractor(audio_waveform, sampling_rate=sample_rate, return_tensors="pt")

    tokenize = bert_token(transcript['text'], padding='max_length',return_attention_mask=True, add_special_tokens=True, truncation=True, max_length=128, return_tensors="pt")


    # Passing the features through the models and getting predictions
    with torch.no_grad():
        wav2vec_output = wav2vec_model(features['input_values']).logits
        wav2vec_output = torch.argmax(wav2vec_output, dim=1).item()

        bert_output = bert_model(tokenize['input_ids'], tokenize['attention_mask'])
        bert_output = torch.argmax(bert_output, dim=1).item()

        fusion_output = fusion_model(features['input_values'], tokenize['input_ids'], tokenize['attention_mask'])
        fusion_output = torch.argmax(fusion_output, dim=1).item()



    # Mapping the predictions to their corresponding labels and displaying the results
    # It is important to note that the wav model for prediction is pretrained and different from the wav model passed into the fusion model
    # Because the wav model is pretrained and different from my custom models, the labels are mapped differently
    # The fix is easy, I just utilised different label dictionaries
    wav_dict = {0: 'neutral', 1: 'happy', 2: 'angry', 3: 'sad'}
    fusion_bert_dict = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry'}

    wav_pred = wav_dict[wav2vec_output]
    bert_pred = fusion_bert_dict[bert_output]
    fusion_pred = fusion_bert_dict[fusion_output]

    # Fusion predicts intent
    # BERT classfies semantics of the spoken works
    # Wav2Vec2 classifies the tonality of the speech

    # All interpretations are my own and based on my personal understanding of the models
    # My interpretations are based on my time working as a Brand ambassador
    # Others are free to interpret the results differently

    # It is also very important to note the accuracy of these models, at best my models acheived around 70% accuracy
    # It is always better to rely on a human interpreter, not only because a highly skilled human can interpret these modalities better
    # but they also have visual cues to look at facial expressions and body language
    # Humans also can understand edge cases better

    # Another very important thing to note is that these models are inherently bias because of the data they are trained on
    # Any data is inherently bias, and the models will learn these biases, so it is important to be mindful of this when interpreting the results
    # For example, the actor x in the IEMOCAP dataset could inherently show emotion in a certain way
    # This is an issue becuase everyone expresses emotions through speech differently
    # The actors in the IEMOCAP dataset are from USA, meaning there could be hidden cultural bias

    if fusion_pred == bert_pred == wav_pred:
        st.write(f"All models agree on the prediction: {fusion_pred}")

        st.write("Your tonality is aligned with your intent and what you are putting into words, no improvements suggested.")
        # This represents correct speech, where all speech modalities are aligned
    
    elif fusion_pred == bert_pred:
        st.write(f"BERT and Fusion models agree on the prediction: {fusion_pred}")
        st.write(f"Wav2Vec2 Prediction: {wav_pred}")

        st.write("Your tonality misrepresents your intent and your words, consider improving your tonality to better align with your intent and words.")
        # This is just a case of the user not being able to express their intent through their tonality


    elif fusion_pred == wav_pred:
        st.write(f"Wav2Vec2 and Fusion models agree on the prediction: {fusion_pred}")
        st.write(f"BERT Prediction: {bert_pred}")

        st.write("Your tonality and intent are aligned, but they aren't aligned with what you are putting into words.")
        # The users intent is aligned with their tonality, this could be a possible indicator for lying
        # Detecting lying through speech only is incredibly difficult even for humans
        # Detecting a lie is heavily reliant on visual cues

    elif bert_pred == wav_pred:
        st.write(f"BERT and Wav2Vec2 models agree on the prediction: {bert_pred}")
        st.write(f"Fusion Model Prediction: {fusion_pred}")

        st.write("Your tonality and what you are putting into words are aligned, but they aren't the same as what you are intending to express.")
        # The users tonality is aligned with their words, meaning they are effectively transmitting the words
        # However they are hiding their intent
        # It is important to note that lack of intent isnt the same as lying
    
    else:
        st.write("All models disagree on the prediction.")
        st.write(f"Wav2Vec2 Prediction: {wav_pred}")
        st.write(f"BERT Prediction: {bert_pred}")
        st.write(f"Fusion Model Prediction: {fusion_pred}")



else:
    st.write("Please upload an audio file to see the predictions.")


