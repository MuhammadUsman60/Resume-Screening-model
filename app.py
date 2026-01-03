from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import fitz  # PyMuPDF
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load your pre-trained model and vectorizer
stacked_model = joblib.load('./model/stacked_resume_model_V2.pkl')
tfidf = joblib.load('./model/tfidf_vectorizer_V2.pkl')

# Load models for rocommend For job
vectorizer = joblib.load('./model/Recommender/tfidf_vectorizer.pkl')
svd = joblib.load('./model/Recommender/svd_reducer.pkl')
kmeans = joblib.load('./model/Recommender/kmeans_model.pkl')
normalizer = joblib.load('./model/Recommender/normalizer.pkl')


# Text preprocessing function
def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = text.replace('â€¢', ' ')
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def recommend_job(resume_text, job_desc_text, positions="", job_position_name="", 
                 major_field_of_studies="", educational_requirements=""):
    try:
        # 1. Preprocess texts
        cleaned_resume = preprocess_text(resume_text)
        cleaned_job_desc = preprocess_text(job_desc_text)
        
        # 2. Vectorize
        resume_vec = vectorizer.transform([cleaned_resume])
        job_vec = vectorizer.transform([cleaned_job_desc])
        
        # 3. Calculate cosine similarity
        cosine_sim = cosine_similarity(resume_vec, job_vec)[0][0]
        
        # 4. Calculate match flags (case insensitive)
        experience_match = 1 if str(positions).lower() in str(job_position_name).lower() else 0
        education_match = 1 if str(major_field_of_studies).lower() in str(educational_requirements).lower() else 0
        
        # 5. Combine all features exactly as in training
        combined = np.hstack([
            resume_vec.toarray(),
            np.array([[cosine_sim]]),
            np.array([[experience_match, education_match]])
        ])
        
        # 6. Dimensionality reduction and normalization
        reduced = svd.transform(combined)
        normalized = normalizer.transform(reduced)
        
        # 7. Predict cluster
        cluster = kmeans.predict(normalized)[0]
        
        # 8. Map to recommendation using BOTH cluster and similarity
        if cosine_sim > 0.7:
            recommendation = "Highly Recommended"
        elif cosine_sim > 0.5:
            recommendation = "Moderately Recommended" 
        elif cosine_sim > 0.3:
            recommendation = "Recommended"
        else:
            recommendation = "Not Recommended"
            
        # Override with cluster if it strongly disagrees
        if cluster == 0 and recommendation == "Highly Recommended":
            recommendation = "Not Recommended"
        elif cluster == 1 and recommendation == "Moderately Recommended":
            recommendation = "Recommended"
            
        return {
            "recommendation": recommendation,
            "cosine_similarity": float(cosine_sim),
            "cluster_id": int(cluster),
            "experience_match": experience_match,
            "education_match": education_match
        }
    
    except Exception as e:
        print(f"Error in recommend_job: {str(e)}")
        raise

def recommend_job_simple(resume_text, job_desc_text):
    try:
        # 1. Preprocess texts
        cleaned_resume = preprocess_text(resume_text)
        cleaned_job_desc = preprocess_text(job_desc_text)
        
        # 2. Vectorize
        resume_vec = vectorizer.transform([cleaned_resume])
        job_vec = vectorizer.transform([cleaned_job_desc])
        
        # 3. Calculate cosine similarity
        cosine_sim = cosine_similarity(resume_vec, job_vec)[0][0]
        
        # 4. Direct recommendation based on thresholds
        if cosine_sim > 0.7:
            return {"recommendation": "Highly Recommended", "score": cosine_sim}
        elif cosine_sim > 0.5:
            return {"recommendation": "Moderately Recommended", "score": cosine_sim}
        elif cosine_sim > 0.3:
            return {"recommendation": "Recommended", "score": cosine_sim}
        else:
            return {"recommendation": "Not Recommended", "score": cosine_sim}
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get resume and job description from request
        data = request.json
        resume_text = data['resume']
        job_desc_text = data['job_description']
        
        # Preprocess texts
        cleaned_resume = preprocess_text(resume_text)
        cleaned_job_desc = preprocess_text(job_desc_text)
        
        # Vectorize
        combined = [cleaned_resume, cleaned_job_desc]
        vectors = tfidf.transform(combined)
        
        # Calculate similarity
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
       
        # Prepare input for model
        x_input = np.hstack((vectors[0].toarray(), np.array(similarity).reshape(1, -1)))
        
        # Predict
        predicted_score = stacked_model.predict(x_input)[0]
        
        return jsonify({
            'matched_score': float(predicted_score*100),
            'similarity_score': float(similarity)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    try:
        result = recommend_job(
            data['resume'],
            data['job_description'],
            data.get('positions', ''),
            data.get('job_position_name', ''),
            data.get('major_field_of_studies', ''),
            data.get('educational_requirements', '')
        )
        
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)





