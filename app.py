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

# Download required NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
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

# def map_cluster_to_label(cluster):
#     """Map cluster IDs to human-readable labels."""
#     # Adjust these labels based on your clustering logic
#     labels = {
#         0: "Highly Recommended",
#         1: "Moderately Recommended",
#         2: "Recommended",
#         3: "Not Recommended"
#     }
#     return labels.get(cluster, "Unknown")

# def map_cluster_to_label(cluster, cosine_sim):
#     """Use both cluster AND similarity score for recommendation"""
#     if cosine_sim > 0.7:
#         return "Highly Recommended"
#     elif cosine_sim > 0.5:
#         return cluster_labels.get(cluster, "Moderately Recommended")
#     elif cosine_sim > 0.3:
#         return "Recommended"
#     else:
#         return "Not Recommended"

# def recommend_job(resume_text, job_desc_text):
#     # Preprocess the texts
#     cleaned_resume = preprocess_text(resume_text)
#     cleaned_job_desc = preprocess_text(job_desc_text)
    
#     # Vectorize
#     resume_vec = vectorizer.transform([cleaned_resume])
#     job_vec = vectorizer.transform([cleaned_job_desc])
    
#     # Calculate cosine similarity
#     cosine_sim = cosine_similarity(resume_vec, job_vec)[0][0]
    
#     # Combine features
#     combined = np.hstack([
#         resume_vec.toarray(),
#         np.array([[cosine_sim]]),
#         np.array([[0, 0]])  # Placeholder for experience/education match
#     ])
    
#     # Dimensionality reduction and normalization
#     reduced = svd.transform(combined)
#     normalized = normalizer.transform(reduced)
    
#     # Predict cluster
#     cluster = kmeans.predict(normalized)[0]
    
#     # Map to recommendation
#     recommendation = map_cluster_to_label(cluster)
    
#     return {
#         'recommendation': recommendation,
#         'cosine_similarity': float(cosine_sim*100),
#         'cluster_id': int(cluster)
#     }

# def recommend_job(resume_text, job_desc_text):
#     try:
#         # 1. Preprocess texts
#         cleaned_resume = preprocess_text(resume_text)
#         cleaned_job_desc = preprocess_text(job_desc_text)
        
#         # 2. Vectorize
#         resume_vec = vectorizer.transform([cleaned_resume])
#         job_vec = vectorizer.transform([cleaned_job_desc])
        
#         # 3. Verify vector shapes (debug)
#         print("Vector shapes:", resume_vec.shape, job_vec.shape)
        
#         # 4. Calculate cosine similarity (with checks)
#         cosine_sim = cosine_similarity(resume_vec, job_vec)
#         if cosine_sim.shape != (1, 1):
#             raise ValueError(f"Unexpected cosine similarity shape: {cosine_sim.shape}")
        
#         cosine_sim = cosine_sim[0][0]
#         print("Cosine similarity:", cosine_sim)  # Debug
        
#         # 5. Combine features and reduce dimensions
#         combined = np.hstack([
#             resume_vec.toarray(),
#             np.array([[cosine_sim]]),
#             np.array([[0, 0]])  # Placeholder for experience/education match
#         ])
        
#         reduced = svd.transform(combined)
#         normalized = normalizer.transform(reduced)
        
#         # 6. Predict cluster
#         cluster = kmeans.predict(normalized)[0]
        
#         return {
#             "recommendation": map_cluster_to_label(cluster),
#             "cosine_similarity": float(np.clip(cosine_sim, -1.0, 1.0)),  # Force to [-1, 1]
#             "cluster_id": int(cluster)
#         }
    
#     except Exception as e:
#         print(f"Error in recommend_job: {str(e)}")
#         raise
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
    

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     try:
#         data = request.json
#         if not data or 'resume' not in data or 'job_description' not in data:
#             return jsonify({"error": "Missing required fields"}), 400
        
#         result = recommend_job(data['resume'], data['job_description'])
#         return jsonify(result)
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

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
        
        # # Enforce minimum similarity threshold
        # if result['cosine_similarity'] < 0.15:  # Adjust threshold as needed
        #     result['recommendation'] = "Not Recommended"
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)











# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# import fitz  # PyMuPDF
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import Normalizer
# from sklearn.cluster import KMeans
# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# import ast

# # Initialize Flask app
# app = Flask(__name__)

# # Download required NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Initialize stopwords and lemmatizer
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# # Load your pre-trained stacked model and TF-IDF vectorizer
# stacked_model = joblib.load('./model/stacked_resume_model_V2.pkl')
# tfidf = joblib.load('./model/tfidf_vectorizer_V2.pkl')


# # ---------------- OLD MODEL ROUTE (DO NOT REMOVE) ----------------
# def preprocess_text(text):
#     if pd.isnull(text):
#         return ''
#     text = text.lower()
#     text = text.replace('â€¢', ' ')
#     text = re.sub(r'[^a-z\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     tokens = nltk.word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
#     return ' '.join(tokens)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         resume_text = data['resume']
#         job_desc_text = data['job_description']
        
#         cleaned_resume = preprocess_text(resume_text)
#         cleaned_job_desc = preprocess_text(job_desc_text)
        
#         combined = [cleaned_resume, cleaned_job_desc]
#         vectors = tfidf.transform(combined)
#         similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
#         x_input = np.hstack((vectors[0].toarray(), np.array(similarity).reshape(1, -1)))
#         predicted_score = stacked_model.predict(x_input)[0]

#         return jsonify({
#             'matched_score': float(predicted_score*100),
#             'similarity_score': float(similarity)
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# # ---------------- NEW ENHANCED MODEL ROUTE ----------------
# def enhanced_preprocess(text):
#     if pd.isnull(text) or text == '':
#         return ''
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)
#     text = re.sub(r"[^a-zA-Z\s]", '', text)
#     text = text.lower().strip()
#     tokens = nltk.word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
#     return ' '.join(tokens)

# def safe_convert_to_list(x):
#     if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
#         try:
#             return ast.literal_eval(x)
#         except:
#             return [x]
#     elif isinstance(x, list):
#         return x
#     else:
#         return [x]

# def parse_date(date_str):
#     try:
#         if pd.isna(date_str) or date_str in ['', 'N/A']:
#             return None
#         if isinstance(date_str, str) and date_str.lower() in ['current', 'till date', 'present']:
#             return datetime.now()
#         return datetime.strptime(str(date_str), '%b %Y')
#     except ValueError:
#         try:
#             return datetime.strptime(str(date_str), '%Y-%m-%d')
#         except:
#             return None

# def calculate_experience(start_dates, end_dates):
#     total_exp = 0
#     start_dates = safe_convert_to_list(start_dates)
#     end_dates = safe_convert_to_list(end_dates)
#     for i in range(min(len(start_dates), len(end_dates))):
#         start = parse_date(start_dates[i])
#         end = parse_date(end_dates[i])
#         if start and end and end >= start:
#             delta = relativedelta(end, start)
#             total_exp += delta.years + delta.months / 12 + delta.days / 365
#     return round(total_exp, 2)

# def map_cluster_to_label(cluster, df):
#     cluster_means = df.groupby('cluster_id')['Cosine_Similarity'].mean()
#     sorted_clusters = cluster_means.sort_values(ascending=False).index
#     if cluster == sorted_clusters[0]:
#         return 'Highly Recommended'
#     elif cluster == sorted_clusters[1]:
#         return 'Moderately Recommended'
#     elif len(sorted_clusters) > 2 and cluster == sorted_clusters[2]:
#         return 'Recommended'
#     else:
#         return 'Not Recommended'


# @app.route('/enhanced-recommendation', methods=['POST'])
# def enhanced_recommendation():
#     try:
#         data = request.json
#         df = pd.DataFrame([data])
#         df.fillna('', inplace=True)

#         # Step 1: Calculate experience
#         df['Experience_Years'] = df.apply(lambda x: calculate_experience(x['start_dates'], x['end_dates']), axis=1)

#         # Step 2: Combine resume and job description fields
#         df['Resume_Text'] = (
#             df['career_objective'].astype(str) + ' ' +
#             df['skills'].astype(str) + ' ' +
#             df['major_field_of_studies'].astype(str) + ' ' +
#             df['professional_company_names'].astype(str) + ' ' +
#             df['positions'].astype(str) + ' ' +
#             df['responsibilities'].astype(str) + ' ' +
#             df['Experience_Years'].astype(str)
#         )

#         df['Job_Description_Text'] = (
#             df['job_position_name'].astype(str) + ' ' +
#             df['skills_required'].astype(str) + ' ' +
#             df['responsibilities_1'].astype(str) + ' ' +
#             df['experiencere_requirement'].astype(str)
#         )

#         # Step 3: Text preprocessing
#         df['Cleaned_Resume'] = df['Resume_Text'].apply(enhanced_preprocess)
#         df['Cleaned_Job_Desc'] = df['Job_Description_Text'].apply(enhanced_preprocess)

#         # Step 4: TF-IDF vectorization
#         vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
#         resume_tfidf = vectorizer.fit_transform(df['Cleaned_Resume'])
#         jobdesc_tfidf = vectorizer.transform(df['Cleaned_Job_Desc'])

#         # Step 5: Cosine similarity
#         cosine_sim = cosine_similarity(resume_tfidf[0], jobdesc_tfidf[0])[0][0]
#         df['Cosine_Similarity'] = [cosine_sim]

#         # Step 6: Match checks
#         df['Experience_Match'] = df.apply(
#             lambda x: 1 if str(x['positions']).lower() in str(x['job_position_name']).lower() else 0,
#             axis=1
#         )
#         df['Education_Match'] = df.apply(
#             lambda x: 1 if str(x['major_field_of_studies']).lower() in str(x['educationaL_requirements']).lower() else 0,
#             axis=1
#         )

#         # Step 7: Combine all features
#         combined_features = np.hstack((
#             resume_tfidf.toarray(),
#             np.array([cosine_sim]).reshape(-1, 1),
#             df[['Experience_Match', 'Education_Match']].values
#         ))

#         # Step 8: Dimensionality reduction (fix for n_features issue)
#         n_features = combined_features.shape[1]
#         n_components = min(100, n_features - 1)
#         svd = TruncatedSVD(n_components=n_components, random_state=42)
#         reduced = svd.fit_transform(combined_features)
#         normalized = Normalizer().fit_transform(reduced)

#         # Step 9: Clustering & labeling
#         kmeans = KMeans(n_clusters=4, random_state=42)
#         df['cluster_id'] = kmeans.fit_predict(normalized)
#         df['Recommendation'] = df['cluster_id'].apply(lambda cid: map_cluster_to_label(cid, df))

#         # Step 10: Return result
#         return jsonify({
#             "Recommendation": df['Recommendation'].iloc[0],
#             "Cosine_Similarity": round(df['Cosine_Similarity'].iloc[0], 4),
#             "Experience_Years": df['Experience_Years'].iloc[0],
#             "Experience_Match": int(df['Experience_Match'].iloc[0]),
#             "Education_Match": int(df['Education_Match'].iloc[0])
#         })

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({'error': str(e)}), 500

# # ---------------- FLASK RUN ----------------
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


