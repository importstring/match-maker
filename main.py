from transformers import pipeline
from PIL import Image
import speech_recognition as sr
import numpy as np
from scipy import stats

# Initialize NLP pipelines
text_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize image analysis model
image_analyzer = pipeline("image-classification", model="nateraw/vit-finetuned-celeba")

# Initialize speech recognition
r = sr.Recognizer()

def analyze_text_advanced(text, user_profile):
    try:
        labels = user_profile["interests"] + user_profile["personality_traits"]
        results = text_classifier(text, candidate_labels=labels)
        sentiment = sentiment_analyzer(text)[0]
        return {
            "traits": {label: score for label, score in zip(results['labels'], results['scores'])},
            "sentiment": sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
        }
    except Exception as e:
        print(f"Error in text analysis: {e}")
        return {"traits": {}, "sentiment": 0}

def analyze_image_advanced(image_path):
    try:
        image = Image.open(image_path)
        results = image_analyzer(image)
        attractiveness_score = sum(result['score'] for result in results if 'attractive' in result['label'].lower())
        return attractiveness_score
    except Exception as e:
        print(f"Error in image analysis: {e}")
        return 0

def transcribe_audio_advanced(audio_file):
    try:
        with sr.AudioFile(audio_file) as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        return r.recognize_google(audio)
    except Exception as e:
        print(f"Error in audio transcription: {e}")
        return ""

def calculate_personality_match_advanced(traits_scores, user_profile):
    score = 0
    weights = user_profile.get("trait_weights", {})
    for trait, trait_score in traits_scores.items():
        if trait in weights:
            score += trait_score * weights[trait] * 100
    return score

def evaluate_candidate_advanced(submission, user_profile):
    personality_score = 0
    attractiveness_score = 0
    sentiment_score = 0

    if 'text' in submission:
        text_analysis = analyze_text_advanced(submission['text'], user_profile)
        personality_score += calculate_personality_match_advanced(text_analysis['traits'], user_profile)
        sentiment_score += text_analysis['sentiment'] * 50

    if 'image' in submission:
        attractiveness_score = analyze_image_advanced(submission['image']) * 100

    if 'video' in submission:
        audio_text = transcribe_audio_advanced(submission['video'])
        video_analysis = analyze_text_advanced(audio_text, user_profile)
        personality_score += calculate_personality_match_advanced(video_analysis['traits'], user_profile)
        sentiment_score += video_analysis['sentiment'] * 50

    final_score = (0.15 * attractiveness_score) + (0.75 * personality_score) + (0.10 * sentiment_score)
    return final_score

def adaptive_threshold(scores, percentile=75, min_threshold=50):
    try:
        threshold = max(np.percentile(scores, percentile), min_threshold)
        return threshold
    except Exception as e:
        print(f"Error in calculating adaptive threshold: {e}")
        return min_threshold

def age_appropriate_filter(text, min_age=15):
    """
    Depends person-to-person
    """"
    return True

def batch_evaluate_candidates(candidates, user_profile, batch_size=10):
    all_scores = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        batch_scores = [evaluate_candidate_advanced(candidate, user_profile) for candidate in batch]
        all_scores.extend(batch_scores)
    return all_scores

# Example profile - This was mine
user_profile = {
    "interests": ["Artificial Intelligence", "Space Exploration", "Music", "Problem Solving", "Brainstorming Futuristic Ideas"],
    "personality_traits": ["curious", "logical", "focused", "humorous"],
    "trait_weights": {
        "Artificial Intelligence": 2,
        "Space Exploration": 1.5,
        "Music": 1,
        "Problem Solving": 1.5,
        "Brainstorming Futuristic Ideas": 2,
        "curious": 1.5,
        "logical": 1.5,
        "focused": 1,
        "humorous": 1
    }
}

# Initialize Candidates [Note you should modify this to fit your dataset.
candidates = [
    {
        'name': 'Candidate 1',
        'text': "I enjoy exploring new cultures and cuisines. I've backpacked through Europe and Asia.",
        'image': "candidate1_photo.jpg",
        'video': "candidate1_intro.mp4"
    },
    {
        'name': 'Candidate 2',
        'text': "I love discussing the mysteries of the universe and brainstorming futuristic ideas. I also enjoy playing the piano and solving puzzles.",
        'image': "candidate2_photo.jpg",
        'video': "candidate2_intro.mp4"
    }
]

all_scores = batch_evaluate_candidates(candidates, user_profile)
compatibility_threshold = adaptive_threshold(all_scores, percentile=70, min_threshold=60)

compatibility_scores = {
    candidate['name']: score 
    for candidate, score in zip(candidates, all_scores) 
    if score >= compatibility_threshold and age_appropriate_filter(candidate['text'])
}

print("Compatibility Scores:")
for name, score in compatibility_scores.items():
    print(f"{name}: {score:.2f}")
