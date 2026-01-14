"""
MBIC BIAS DETECTION - STREAMLIT WEB APPLICATION
================================================================================
A professional web interface for the MBIC Bias Detection System

Features:
- Single text analysis with LIME explanations
- Batch processing from text input or CSV upload
- Interactive visualizations
- Export results to CSV
- Model performance metrics

To run: streamlit run app.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import faiss
from lime.lime_text import LimeTextExplainer
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="MBIC Bias Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .biased {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .non_biased {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    .no_agreement {
        background-color: #f5f5f5;
        border-left: 4px solid #9e9e9e;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND ARTIFACTS (CACHED)
# ============================================================================

@st.cache_resource
def load_artifacts():
    """Load all model artifacts (cached for performance)."""
    try:
        label_encoder = joblib.load("label_encoder.pkl")
        classifier = joblib.load("best_classifier.pkl")
        faiss_index = faiss.read_index("mbic_faiss.index")
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        return {
            'label_encoder': label_encoder,
            'classifier': classifier,
            'faiss_index': faiss_index,
            'embedding_model': embedding_model,
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }

# ============================================================================
# BIAS DETECTION PIPELINE
# ============================================================================

class BiasDetectionPipeline:
    """Pipeline for bias detection with LIME explainability."""
    
    def __init__(self, embedding_model, classifier, label_encoder, faiss_index):
        self.embedding_model = embedding_model
        self.classifier = classifier
        self.label_encoder = label_encoder
        self.faiss_index = faiss_index
        self.lime_explainer = LimeTextExplainer(
            class_names=label_encoder.classes_,
            random_state=42
        )
    
    def embed_text(self, texts):
        """Convert text(s) to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.astype('float32')
    
    def predict_single(self, text):
        """Predict bias for a single text."""
        embedding = self.embed_text(text)
        pred_encoded = self.classifier.predict(embedding)[0]
        pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        result = {
            'text': text,
            'prediction': pred_label,
            'prediction_encoded': pred_encoded
        }
        
        if hasattr(self.classifier, 'predict_proba'):
            probs = self.classifier.predict_proba(embedding)[0]
            result['probabilities'] = {
                label: float(prob) 
                for label, prob in zip(self.label_encoder.classes_, probs)
            }
            result['confidence'] = float(max(probs))
        
        return result
    
    def predict_batch(self, texts):
        """Predict bias for multiple texts."""
        embeddings = self.embed_text(texts)
        predictions = self.classifier.predict(embeddings)
        
        results = []
        for i, (text, pred_encoded) in enumerate(zip(texts, predictions)):
            pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
            
            result = {
                'text': text,
                'prediction': pred_label,
                'prediction_encoded': pred_encoded
            }
            
            if hasattr(self.classifier, 'predict_proba'):
                probs = self.classifier.predict_proba(embeddings[i:i+1])[0]
                result['probabilities'] = {
                    label: float(prob) 
                    for label, prob in zip(self.label_encoder.classes_, probs)
                }
                result['confidence'] = float(max(probs))
            
            results.append(result)
        
        return results
    
    def predict_fn_for_lime(self, texts):
        """Prediction function for LIME."""
        embeddings = self.embed_text(texts)
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(embeddings)
        else:
            predictions = self.classifier.predict(embeddings)
            n_classes = len(self.label_encoder.classes_)
            probs = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                probs[i, pred] = 1.0
            return probs
    
    def explain_prediction(self, text, num_features=10):
        """Generate LIME explanation for a prediction."""
        prediction = self.predict_single(text)
        explanation = self.lime_explainer.explain_instance(
            text,
            self.predict_fn_for_lime,
            num_features=num_features,
            num_samples=3000,
            top_labels=len(self.label_encoder.classes_)
        )
        return explanation, prediction

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_probability_chart(probabilities):
    """Create a horizontal bar chart for probabilities."""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    
    colors = []
    for label in labels:
        if label == 'Biased':
            colors.append('#f44336')
        elif label == 'Non-biased':
            colors.append('#4caf50')
        else:
            colors.append('#9e9e9e')
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{v:.2%}' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Probability",
        yaxis_title="Class",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_lime_chart(word_contributions):
    """Create a bar chart for LIME word contributions."""
    words = [w[0] for w in word_contributions]
    weights = [w[1] for w in word_contributions]
    
    colors = ['#f44336' if w > 0 else '#4caf50' for w in weights]
    
    fig = go.Figure(data=[
        go.Bar(
            x=weights,
            y=words,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{w:+.4f}' for w in weights],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="LIME Word Contributions",
        xaxis_title="Contribution Weight",
        yaxis_title="Word",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_batch_results_chart(results_df):
    """Create a pie chart for batch predictions."""
    pred_counts = results_df['Prediction'].value_counts()
    
    colors = []
    for label in pred_counts.index:
        if label == 'Biased':
            colors.append('#f44336')
        elif label == 'Non-biased':
            colors.append('#4caf50')
        else:
            colors.append('#9e9e9e')
    
    fig = go.Figure(data=[
        go.Pie(
            labels=pred_counts.index,
            values=pred_counts.values,
            marker=dict(colors=colors),
            textinfo='label+percent',
            hole=0.3
        )
    ])
    
    fig.update_layout(
        title="Prediction Distribution",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🎯 MBIC Bias Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Text Bias Analysis with Explainability</div>', unsafe_allow_html=True)
    
    # Load artifacts
    with st.spinner("Loading model artifacts..."):
        artifacts = load_artifacts()
    
    if artifacts['status'] == 'error':
        st.error(f"❌ Error loading model: {artifacts['error_message']}")
        st.info("Please ensure all model files (label_encoder.pkl, best_classifier.pkl, mbic_faiss.index) are in the same directory as this script.")
        st.stop()
    
    # Initialize pipeline
    pipeline = BiasDetectionPipeline(
        artifacts['embedding_model'],
        artifacts['classifier'],
        artifacts['label_encoder'],
        artifacts['faiss_index']
    )
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Select Mode",
            ["Single Text Analysis", "Batch Processing", "Model Info"],
            index=0
        )
        
        st.divider()
        
        # Model metrics
        st.subheader("📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "74.59%")
            st.metric("F1 Score", "0.6655")
        with col2:
            st.metric("Model", "XGBoost")
            st.metric("Classes", "3")
        
        st.divider()
        
        # Example texts
        st.subheader("📝 Quick Examples")
        examples = {
            "Biased (Gender)": "Women are naturally better at caring for children than men.",
            "Non-biased": "The research shows that diverse teams perform better in problem-solving tasks.",
            "Biased (Age)": "All teenagers are irresponsible and lazy.",
            "Evidence-based": "According to the study, there was no significant difference between groups."
        }
        
        selected_example = st.selectbox("Load example:", ["None"] + list(examples.keys()))
    
    # Main content area
    if mode == "Single Text Analysis":
        st.header("🔍 Single Text Analysis")
        
        # Text input
        if selected_example != "None":
            default_text = examples[selected_example]
        else:
            default_text = ""
        
        text_input = st.text_area(
            "Enter text to analyze:",
            value=default_text,
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_btn = st.button("🎯 Analyze", type="primary", use_container_width=True)
        with col2:
            show_lime = st.checkbox("Show LIME Explanation", value=True)
        
        if analyze_btn and text_input:
            with st.spinner("Analyzing text..."):
                result = pipeline.predict_single(text_input)
            
            # Display prediction
            pred_class = result['prediction'].lower().replace(' ', '_').replace('-', '_')
            
            st.markdown(f"""
            <div class="prediction-box {pred_class}">
                <h3>Prediction: {result['prediction']}</h3>
                <p><strong>Confidence:</strong> {result.get('confidence', 0):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability chart
            if 'probabilities' in result:
                st.plotly_chart(
                    create_probability_chart(result['probabilities']),
                    use_container_width=True
                )
            
            # LIME explanation
            if show_lime:
                st.subheader("🔍 LIME Explanation")
                st.info("This shows which words influenced the prediction the most.")
                
                with st.spinner("Generating LIME explanation (30-60 seconds)..."):
                    explanation, _ = pipeline.explain_prediction(text_input, num_features=10)
                
                pred_class_idx = result['prediction_encoded']
                word_contributions = explanation.as_list(label=pred_class_idx)
                
                # LIME chart
                st.plotly_chart(
                    create_lime_chart(word_contributions),
                    use_container_width=True
                )
                
                # Explanation text
                st.markdown("**Interpretation:**")
                st.markdown("- 🔴 **Red bars (positive values)**: Words that increase the likelihood of the predicted class")
                st.markdown("- 🟢 **Green bars (negative values)**: Words that decrease the likelihood of the predicted class")
    
    elif mode == "Batch Processing":
        st.header("📦 Batch Processing")
        
        tab1, tab2 = st.tabs(["Text Input", "CSV Upload"])
        
        with tab1:
            st.subheader("Enter Multiple Texts")
            st.info("Enter one text per line")
            
            batch_input = st.text_area(
                "Texts to analyze:",
                height=300,
                placeholder="Enter multiple texts, one per line..."
            )
            
            if st.button("🎯 Analyze Batch", type="primary"):
                if batch_input:
                    texts = [t.strip() for t in batch_input.split('\n') if t.strip()]
                    
                    if texts:
                        with st.spinner(f"Analyzing {len(texts)} texts..."):
                            results = pipeline.predict_batch(texts)
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame([
                            {
                                'Text': r['text'],
                                'Prediction': r['prediction'],
                                'Confidence': r.get('confidence', None),
                                'Prob_Biased': r['probabilities'].get('Biased', None) if 'probabilities' in r else None,
                                'Prob_Non_Biased': r['probabilities'].get('Non-biased', None) if 'probabilities' in r else None,
                                'Prob_No_Agreement': r['probabilities'].get('No agreement', None) if 'probabilities' in r else None,
                            }
                            for r in results
                        ])
                        
                        # Display results
                        st.success(f"✅ Analyzed {len(texts)} texts successfully!")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            biased_count = (results_df['Prediction'] == 'Biased').sum()
                            st.metric("🔴 Biased", f"{biased_count}/{len(texts)}")
                        with col2:
                            non_biased_count = (results_df['Prediction'] == 'Non-biased').sum()
                            st.metric("🟢 Non-biased", f"{non_biased_count}/{len(texts)}")
                        with col3:
                            no_agreement_count = (results_df['Prediction'] == 'No agreement').sum()
                            st.metric("⚪ No Agreement", f"{no_agreement_count}/{len(texts)}")
                        
                        # Visualization
                        st.plotly_chart(
                            create_batch_results_chart(results_df),
                            use_container_width=True
                        )
                        
                        # Results table
                        st.subheader("📋 Detailed Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"bias_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ Please enter at least one text to analyze.")
                else:
                    st.warning("⚠️ Please enter some text to analyze.")
        
        with tab2:
            st.subheader("Upload CSV File")
            st.info("Upload a CSV file with a column named 'text' or 'sentence'")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    
                    # Find text column
                    text_col = None
                    for col in ['text', 'Text', 'sentence', 'Sentence']:
                        if col in df_upload.columns:
                            text_col = col
                            break
                    
                    if text_col is None:
                        st.error("❌ CSV must have a column named 'text' or 'sentence'")
                    else:
                        st.success(f"✅ Found {len(df_upload)} texts in column '{text_col}'")
                        
                        # Preview
                        st.subheader("Preview")
                        st.dataframe(df_upload.head(5))
                        
                        if st.button("🎯 Analyze CSV", type="primary"):
                            texts = df_upload[text_col].astype(str).tolist()
                            
                            with st.spinner(f"Analyzing {len(texts)} texts..."):
                                results = pipeline.predict_batch(texts)
                            
                            # Add results to DataFrame
                            df_upload['Prediction'] = [r['prediction'] for r in results]
                            df_upload['Confidence'] = [r.get('confidence', None) for r in results]
                            
                            for label in ['Biased', 'Non-biased', 'No agreement']:
                                df_upload[f'Prob_{label.replace(" ", "_")}'] = [
                                    r['probabilities'].get(label, None) if 'probabilities' in r else None
                                    for r in results
                                ]
                            
                            st.success(f"✅ Analyzed {len(texts)} texts successfully!")
                            
                            # Summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                biased_count = (df_upload['Prediction'] == 'Biased').sum()
                                st.metric("🔴 Biased", f"{biased_count}/{len(texts)}")
                            with col2:
                                non_biased_count = (df_upload['Prediction'] == 'Non-biased').sum()
                                st.metric("🟢 Non-biased", f"{non_biased_count}/{len(texts)}")
                            with col3:
                                no_agreement_count = (df_upload['Prediction'] == 'No agreement').sum()
                                st.metric("⚪ No Agreement", f"{no_agreement_count}/{len(texts)}")
                            
                            # Visualization
                            st.plotly_chart(
                                create_batch_results_chart(df_upload),
                                use_container_width=True
                            )
                            
                            # Results table
                            st.subheader("📋 Results")
                            st.dataframe(df_upload, use_container_width=True)
                            
                            # Download
                            csv = df_upload.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results (CSV)",
                                data=csv,
                                file_name=f"bias_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"❌ Error processing CSV: {str(e)}")
    
    else:  # Model Info
        st.header("ℹ️ Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Performance")
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
                'Value': ['74.59%', '0.6655', '0.6780', '0.6850']
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            st.subheader("🎯 Classes")
            classes_df = pd.DataFrame({
                'Class': artifacts['label_encoder'].classes_,
                'Description': [
                    'Text contains bias',
                    'Text is objective/neutral',
                    'Ambiguous/mixed signals'
                ]
            })
            st.dataframe(classes_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🔧 Model Details")
            st.markdown("""
            - **Model Type:** XGBoost Classifier
            - **Embedding Model:** all-mpnet-base-v2
            - **Embedding Dimension:** 768
            - **Training Time:** ~2 hours
            - **Cross-Validation:** 5-fold CV
            - **Explainability:** LIME
            """)
            
            st.subheader("⚠️ Known Limitations")
            st.markdown("""
            - Struggles with minority class ('No agreement')
            - Bias toward 'Biased' predictions
            - Context-dependent performance
            - Limited to English language
            """)
        
        st.divider()
        
        st.subheader("📚 How It Works")
        
        with st.expander("1️⃣ Text Embedding"):
            st.markdown("""
            The input text is converted to a 768-dimensional vector using the 
            **all-mpnet-base-v2** sentence transformer model. This captures semantic 
            meaning and context.
            """)
        
        with st.expander("2️⃣ Classification"):
            st.markdown("""
            The text embedding is passed to an **XGBoost classifier** trained on 
            labeled examples of biased and non-biased text. The model outputs 
            probabilities for each class.
            """)
        
        with st.expander("3️⃣ LIME Explanation"):
            st.markdown("""
            **LIME (Local Interpretable Model-agnostic Explanations)** identifies 
            which words contributed most to the prediction by testing variations 
            of the input text and observing how predictions change.
            """)
        
        st.divider()
        
        st.subheader("💡 Tips for Best Results")
        st.markdown("""
        - Provide complete sentences or paragraphs
        - Avoid very short texts (< 5 words)
        - Results are most reliable for English text
        - Review LIME explanations to understand predictions
        - Use batch processing for large datasets
        """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()