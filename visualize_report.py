import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from pathlib import Path
import zipfile
import io
import json
from datetime import datetime
import time
import json
import numpy as np
import os
import logging
from typing import Dict, List, Any, Optional

# Forensic analysis imports
try:
    from forensic_deepfake_detection import ForensicConfig, ForensicSegment, AudioForensicsEngine, IdentityClusterer, EvidenceGenerator
    from forensic_pipeline import ForensicPipeline
    from forensic_visualizations import ForensicVisualizer
    FORENSIC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Forensic modules not available: {e}")
    FORENSIC_AVAILABLE = False

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None
import matplotlib.pyplot as plt

# PyThaiNLP for Thai word cloud
try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus.common import thai_stopwords
    from pythainlp.util import normalize as thai_normalize
except ImportError:
    word_tokenize = None
    thai_stopwords = None
    thai_normalize = None

# Set Page Config with better styling
st.set_page_config(
    page_title="Speech Sentiment Analysis Report",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: bold !important;
        color: #1f77b4 !important;
        text-align: center !important;
        padding: 1rem 0 !important;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .stDataFrame {
        font-size: 14px !important;
    }
    
    .stPlotlyChart {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
    }
    
    .streamlit-container {
        max-width: 100%;
    }
    
    h1 {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
    }
    
    h2 {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 600;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #34495e;
        font-size: 1.4rem;
        font-weight: 500;
    }
    
    .stAlert {
        border-radius: 10px;
    }
    
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Speech Sentiment Analysis Report")
st.markdown("### Advanced Visualization Dashboard for Speaker & Emotion Tracking")

# Sidebar - Configuration
st.sidebar.header("📊 Dashboard Settings")
csv_path = st.sidebar.text_input("CSV Log Path", value="analysis_log.csv")
refresh_rate = st.sidebar.slider("Auto-Refresh (seconds)", 1, 60, 10)
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)

# Reset Data Section
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Data Management")

# Reset cache button
if st.sidebar.button("🗑️ Clear All Cache", type="secondary"):
    st.cache_data.clear()
    st.success("Cache cleared! Dashboard will refresh.")
    st.rerun()

# Reset filters button  
if st.sidebar.button("🔄 Reset Filters", type="secondary"):
    # Reset to default values
    st.session_state.min_conf = 0.0
    st.session_state.selected_speakers = []
    st.success("Filters reset! Dashboard will refresh.")
    st.rerun()

# Data validation button
if st.sidebar.button("🔍 Validate Data", type="secondary"):
    if Path(csv_path).exists():
        try:
            test_df = pd.read_csv(csv_path)
            st.sidebar.success(f"✅ CSV Valid: {len(test_df)} rows, {len(test_df.columns)} columns")
            st.sidebar.write(f"Date range: {test_df['ts_utc'].min()} to {test_df['ts_utc'].max()}")
        except Exception as e:
            st.sidebar.error(f"❌ CSV Error: {str(e)}")
    else:
        st.sidebar.error(f"❌ File not found: {csv_path}")

# Export Data Section
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Export Analysis Data")

def create_export_zip(export_name="analysis_export"):
    """Create a comprehensive ZIP export of all analysis visualizations as JPG"""
    
    # Create in-memory ZIP
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Generate and save visualizations as JPG
            
            # 1. Timeline Analysis
            if df is not None and len(df) > 0:
                try:
                    timeline_fig = px.line(df, x='ts_utc', y='confidence', color='speaker',
                                          title='Timeline Analysis', template='plotly_white')
                    timeline_bytes = timeline_fig.to_image(format="jpg", width=1200, height=600)
                    zip_file.writestr("1_Timeline_Analysis.jpg", timeline_bytes)
                except Exception as e:
                    # Fallback: save as HTML if image export fails
                    timeline_fig = px.line(df, x='ts_utc', y='confidence', color='speaker',
                                          title='Timeline Analysis', template='plotly_white')
                    zip_file.writestr("1_Timeline_Analysis.html", timeline_fig.to_html(include_plotlyjs='cdn'))
                
                # 2. Confidence & Uncertainty Trends
                try:
                    confidence_fig = px.line(df, x='ts_utc', y=['confidence', 'entropy'],
                                             title='Confidence & Uncertainty Trends', template='plotly_white')
                    confidence_bytes = confidence_fig.to_image(format="jpg", width=1200, height=600)
                    zip_file.writestr("2_Confidence_Uncertainty_Trends.jpg", confidence_bytes)
                except Exception as e:
                    confidence_fig = px.line(df, x='ts_utc', y=['confidence', 'entropy'],
                                             title='Confidence & Uncertainty Trends', template='plotly_white')
                    zip_file.writestr("2_Confidence_Uncertainty_Trends.html", confidence_fig.to_html(include_plotlyjs='cdn'))
                
                # 3. Emotion Breakdown
                try:
                    emotion_counts = df['pred_class'].value_counts()
                    emotion_fig = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                                         title='Emotion Breakdown', template='plotly_white')
                    emotion_bytes = emotion_fig.to_image(format="jpg", width=800, height=600)
                    zip_file.writestr("3_Emotion_Breakdown.jpg", emotion_bytes)
                except Exception as e:
                    emotion_counts = df['pred_class'].value_counts()
                    emotion_fig = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                                         title='Emotion Breakdown', template='plotly_white')
                    zip_file.writestr("3_Emotion_Breakdown.html", emotion_fig.to_html(include_plotlyjs='cdn'))
                
                # 4. Emotion Probability Heatmap
                prob_cols = [c for c in df.columns if c.startswith('prob_')]
                if prob_cols:
                    try:
                        prob_data = df[prob_cols].copy()
                        prob_data.columns = [c.replace('prob_', '') for c in prob_data.columns]
                        prob_data.index = df['ts_utc'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        heatmap_fig = px.imshow(prob_data.T, title='Emotion Probability Heatmap',
                                                color_continuous_scale='Viridis', template='plotly_white')
                        heatmap_bytes = heatmap_fig.to_image(format="jpg", width=1200, height=800)
                        zip_file.writestr("4_Emotion_Probability_Heatmap.jpg", heatmap_bytes)
                    except Exception as e:
                        prob_data = df[prob_cols].copy()
                        prob_data.columns = [c.replace('prob_', '') for c in prob_data.columns]
                        prob_data.index = df['ts_utc'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        heatmap_fig = px.imshow(prob_data.T, title='Emotion Probability Heatmap',
                                                color_continuous_scale='Viridis', template='plotly_white')
                        zip_file.writestr("4_Emotion_Probability_Heatmap.html", heatmap_fig.to_html(include_plotlyjs='cdn'))
                
                # 5. Speaker Talk Time (Segments)
                try:
                    speaker_stats = df['speaker'].value_counts()
                    talktime_fig = px.bar(x=speaker_stats.values, y=speaker_stats.index,
                                         orientation='h', title='Speaker Talk Time (Segments)',
                                         template='plotly_white')
                    talktime_bytes = talktime_fig.to_image(format="jpg", width=1000, height=800)
                    zip_file.writestr("5_Speaker_Talk_Time.jpg", talktime_bytes)
                except Exception as e:
                    speaker_stats = df['speaker'].value_counts()
                    talktime_fig = px.bar(x=speaker_stats.values, y=speaker_stats.index,
                                         orientation='h', title='Speaker Talk Time (Segments)',
                                         template='plotly_white')
                    zip_file.writestr("5_Speaker_Talk_Time.html", talktime_fig.to_html(include_plotlyjs='cdn'))
                
                # 6. Emotion Matrix per Speaker
                try:
                    speaker_emotion = df.groupby(['speaker', 'pred_class']).size().unstack(fill_value=0)
                    matrix_fig = px.imshow(speaker_emotion, title='Emotion Matrix per Speaker',
                                           color_continuous_scale='Blues', template='plotly_white')
                    matrix_bytes = matrix_fig.to_image(format="jpg", width=1000, height=800)
                    zip_file.writestr("6_Emotion_Matrix.jpg", matrix_bytes)
                except Exception as e:
                    speaker_emotion = df.groupby(['speaker', 'pred_class']).size().unstack(fill_value=0)
                    matrix_fig = px.imshow(speaker_emotion, title='Emotion Matrix per Speaker',
                                           color_continuous_scale='Blues', template='plotly_white')
                    zip_file.writestr("6_Emotion_Matrix.html", matrix_fig.to_html(include_plotlyjs='cdn'))
                
                # 7. Sentiment Volatility
                try:
                    volatility_data = df.groupby('speaker')['pred_class'].apply(lambda x: (x != x.shift()).mean()).reset_index()
                    volatility_data.columns = ['speaker', 'volatility_score']
                    volatility_fig = px.bar(volatility_data, x='speaker', y='volatility_score',
                                            title='Sentiment Volatility', template='plotly_white')
                    volatility_bytes = volatility_fig.to_image(format="jpg", width=1000, height=600)
                    zip_file.writestr("7_Sentiment_Volatility.jpg", volatility_bytes)
                except Exception as e:
                    volatility_data = df.groupby('speaker')['pred_class'].apply(lambda x: (x != x.shift()).mean()).reset_index()
                    volatility_data.columns = ['speaker', 'volatility_score']
                    volatility_fig = px.bar(volatility_data, x='speaker', y='volatility_score',
                                            title='Sentiment Volatility', template='plotly_white')
                    zip_file.writestr("7_Sentiment_Volatility.html", volatility_fig.to_html(include_plotlyjs='cdn'))
                
                # 8. Speaker Interaction (Transition Matrix)
                if len(df) > 1:
                    try:
                        df['next_speaker'] = df['speaker'].shift(-1)
                        trans_df = df.dropna(subset=['next_speaker'])
                        matrix = pd.crosstab(trans_df['speaker'], trans_df['next_speaker'])
                        
                        interaction_fig = px.imshow(matrix, text_auto=True, color_continuous_scale='Blues',
                                                 title='Speaker Interaction (Transition Matrix)',
                                                 template='plotly_white')
                        interaction_bytes = interaction_fig.to_image(format="jpg", width=1000, height=800)
                        zip_file.writestr("8_Speaker_Interaction_Matrix.jpg", interaction_bytes)
                    except Exception as e:
                        df['next_speaker'] = df['speaker'].shift(-1)
                        trans_df = df.dropna(subset=['next_speaker'])
                        matrix = pd.crosstab(trans_df['speaker'], trans_df['next_speaker'])
                        
                        interaction_fig = px.imshow(matrix, text_auto=True, color_continuous_scale='Blues',
                                                 title='Speaker Interaction (Transition Matrix)',
                                                 template='plotly_white')
                        zip_file.writestr("8_Speaker_Interaction_Matrix.html", interaction_fig.to_html(include_plotlyjs='cdn'))
                
                # 9. Speaker Identity Separation (t-SNE)
                if Path("speaker_tsne_plot.png").exists():
                    with open("speaker_tsne_plot.png", 'rb') as f:
                        zip_file.writestr("9_t_SNE_Visualization.png", f.read())
                else:
                    # Create a placeholder if t-SNE doesn't exist
                    try:
                        tsne_fig = go.Figure()
                        tsne_fig.add_annotation(text="t-SNE plot not available", x=0.5, y=0.5, 
                                                 xref="paper", yref="paper", showarrow=False)
                        tsne_bytes = tsne_fig.to_image(format="jpg", width=800, height=600)
                        zip_file.writestr("9_t_SNE_Visualization.jpg", tsne_bytes)
                    except Exception as e:
                        tsne_fig = go.Figure()
                        tsne_fig.add_annotation(text="t-SNE plot not available", x=0.5, y=0.5, 
                                                 xref="paper", yref="paper", showarrow=False)
                        zip_file.writestr("9_t_SNE_Visualization.html", tsne_fig.to_html(include_plotlyjs='cdn'))
                
                # 10. Emotion Radar Comparison
                if prob_cols:
                    try:
                        radar_data = df.groupby('speaker')[prob_cols].mean().reset_index()
                        radar_data.columns = ['speaker'] + [c.replace('prob_', '') for c in prob_cols]
                        
                        fig_radar = go.Figure()
                        for _, row in radar_data.iterrows():
                            fig_radar.add_trace(go.Scatterpolar(
                                r=row[1:].values,
                                theta=radar_data.columns[1:],
                                fill='toself',
                                name=row['speaker']
                            ))
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 1])
                            ),
                            title="Emotion Radar Comparison",
                            template='plotly_white'
                        )
                        radar_bytes = fig_radar.to_image(format="jpg", width=800, height=800)
                        zip_file.writestr("10_Emotion_Radar_Comparison.jpg", radar_bytes)
                    except Exception as e:
                        radar_data = df.groupby('speaker')[prob_cols].mean().reset_index()
                        radar_data.columns = ['speaker'] + [c.replace('prob_', '') for c in prob_cols]
                        
                        fig_radar = go.Figure()
                        for _, row in radar_data.iterrows():
                            fig_radar.add_trace(go.Scatterpolar(
                                r=row[1:].values,
                                theta=radar_data.columns[1:],
                                fill='toself',
                                name=row['speaker']
                            ))
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 1])
                            ),
                            title="Emotion Radar Comparison",
                            template='plotly_white'
                        )
                        zip_file.writestr("10_Emotion_Radar_Comparison.html", fig_radar.to_html(include_plotlyjs='cdn'))
                
                # 11. Audio X-Ray RPCA Analysis
                xray_files = [
                    ("audio_xray_test/csv_rpca_analysis.png", "11_Audio_Xray_Analysis.png"),
                    ("audio_xray_test/test_sample_rpca_analysis.png", "11_Audio_Xray_Test.png")
                ]
                
                xray_found = False
                for source_file, dest_name in xray_files:
                    source_path = Path(source_file)
                    if source_path.exists():
                        with open(source_path, 'rb') as f:
                            zip_file.writestr(dest_name, f.read())
                        xray_found = True
                
                if not xray_found:
                    # Create placeholder if Audio X-Ray doesn't exist
                    try:
                        xray_fig = go.Figure()
                        xray_fig.add_annotation(text="Audio X-Ray analysis not available", x=0.5, y=0.5,
                                                  xref="paper", yref="paper", showarrow=False)
                        xray_bytes = xray_fig.to_image(format="jpg", width=800, height=600)
                        zip_file.writestr("11_Audio_Xray_Analysis.jpg", xray_bytes)
                    except Exception as e:
                        xray_fig = go.Figure()
                        xray_fig.add_annotation(text="Audio X-Ray analysis not available", x=0.5, y=0.5,
                                                  xref="paper", yref="paper", showarrow=False)
                        zip_file.writestr("11_Audio_Xray_Analysis.html", xray_fig.to_html(include_plotlyjs='cdn'))
                
                # 12. Sentiment vs. Acoustic Correlation
                try:
                    correlation_fig = px.scatter(df, x='rms', y='confidence', color='pred_class',
                                               title='Sentiment vs. Acoustic Correlation',
                                               template='plotly_white')
                    correlation_bytes = correlation_fig.to_image(format="jpg", width=800, height=600)
                    zip_file.writestr("12_Sentiment_Acoustic_Correlation.jpg", correlation_bytes)
                except Exception as e:
                    correlation_fig = px.scatter(df, x='rms', y='confidence', color='pred_class',
                                               title='Sentiment vs. Acoustic Correlation',
                                               template='plotly_white')
                    zip_file.writestr("12_Sentiment_Acoustic_Correlation.html", correlation_fig.to_html(include_plotlyjs='cdn'))
                
                # 13. Confidence vs Volume
                try:
                    volume_fig = px.scatter(df, x='rms', y='confidence', 
                                           title='Confidence vs Volume', template='plotly_white')
                    volume_bytes = volume_fig.to_image(format="jpg", width=800, height=600)
                    zip_file.writestr("13_Confidence_Volume.jpg", volume_bytes)
                except Exception as e:
                    volume_fig = px.scatter(df, x='rms', y='confidence', 
                                           title='Confidence vs Volume', template='plotly_white')
                    zip_file.writestr("13_Confidence_Volume.html", volume_fig.to_html(include_plotlyjs='cdn'))
                
                # Add summary statistics
                summary_stats = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_visualizations": 13,
                    "total_segments": len(df) if df is not None else 0,
                    "unique_speakers": df['speaker'].nunique() if df is not None else 0,
                    "date_range": {
                        "start": df['ts_utc'].min().isoformat() if df is not None else None,
                        "end": df['ts_utc'].max().isoformat() if df is not None else None
                    },
                    "kaleido_installed": True,
                    "export_format": "JPG (with HTML fallback if needed)"
                }
                zip_file.writestr("summary_statistics.json", json.dumps(summary_stats, indent=2))
    
    except Exception as e:
        # If there's an error, create a minimal ZIP with error info
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            error_info = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "message": "Export failed, but some visualizations may be available",
                "suggestion": "Please install kaleido: pip install --upgrade kaleido"
            }
            zip_file.writestr("export_error.json", json.dumps(error_info, indent=2))
    
    zip_buffer.seek(0)
    return zip_buffer

# Individual Export Functions
def export_timeline_data():
    """Export Timeline Analysis data"""
    timeline_data = df[['ts_utc', 'speaker', 'pred_class', 'confidence']].copy()
    timeline_data['timestamp'] = timeline_data['ts_utc'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return timeline_data.to_csv(index=False).encode('utf-8')

def export_confidence_trends():
    """Export Confidence & Uncertainty Trends"""
    confidence_data = df[['ts_utc', 'confidence', 'entropy']].copy()
    confidence_data['timestamp'] = confidence_data['ts_utc'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return confidence_data.to_csv(index=False).encode('utf-8')

def export_emotion_breakdown():
    """Export Emotion Breakdown data"""
    emotion_counts = df['pred_class'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']
    emotion_counts['percentage'] = (emotion_counts['count'] / len(df) * 100).round(2)
    return emotion_counts.to_csv(index=False).encode('utf-8')

def export_emotion_heatmap():
    """Export Emotion Probability Heatmap data"""
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if prob_cols:
        prob_data = df[prob_cols].copy()
        prob_data.columns = [c.replace('prob_', '') for c in prob_data.columns]
        prob_data.index = df['ts_utc'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return prob_data.to_csv().encode('utf-8')
    return b"No probability data available"

def export_speaker_talk_time():
    """Export Speaker Talk Time data"""
    speaker_stats = df['speaker'].value_counts().reset_index()
    speaker_stats.columns = ['speaker', 'segments']
    speaker_stats['percentage'] = (speaker_stats['segments'] / len(df) * 100).round(2)
    return speaker_stats.to_csv(index=False).encode('utf-8')

def export_emotion_matrix():
    """Export Emotion Matrix per Speaker"""
    speaker_emotion = df.groupby(['speaker', 'pred_class']).size().unstack(fill_value=0)
    return speaker_emotion.to_csv().encode('utf-8')

def export_volatility():
    """Export Sentiment Volatility data"""
    volatility_data = df.groupby('speaker')['pred_class'].apply(lambda x: (x != x.shift()).mean()).reset_index()
    volatility_data.columns = ['speaker', 'volatility_score']
    return volatility_data.to_csv(index=False).encode('utf-8')

def export_interaction_matrix():
    """Export Speaker Interaction Matrix"""
    if len(df) > 1:
        df['next_speaker'] = df['speaker'].shift(-1)
        trans_df = df.dropna(subset=['next_speaker'])
        matrix = pd.crosstab(trans_df['speaker'], trans_df['next_speaker'])
        return matrix.to_csv().encode('utf-8')
    return b"Not enough data for interaction matrix"

def export_t_sne():
    """Export t-SNE data"""
    if Path("experiment_data.npz").exists():
        try:
            exp_data = np.load("experiment_data.npz")
            if 'embeddings' in exp_data and 'labels' in exp_data:
                t_sne_data = pd.DataFrame({
                    'speaker': exp_data['labels'],
                    'embedding_x': exp_data['embeddings'][:, 0],
                    'embedding_y': exp_data['embeddings'][:, 1]
                })
                return t_sne_data.to_csv(index=False).encode('utf-8')
        except:
            pass
    return b"No t-SNE data available"

def export_radar_comparison():
    """Export Emotion Radar Comparison data"""
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if prob_cols:
        radar_data = df.groupby('speaker')[prob_cols].mean().reset_index()
        radar_data.columns = ['speaker'] + [c.replace('prob_', '') for c in prob_cols]
        return radar_data.to_csv(index=False).encode('utf-8')
    return b"No probability data for radar comparison"

def export_audio_xray():
    """Export Audio X-Ray Analysis"""
    xray_report = Path("audio_xray_test/csv_analysis_report.txt")
    if xray_report.exists():
        with open(xray_report, 'r', encoding='utf-8') as f:
            return f.read().encode('utf-8')
    return b"No Audio X-Ray analysis available"

def export_correlation():
    """Export Correlation data"""
    correlation_data = df[['confidence', 'rms', 'entropy']].copy()
    correlation_data['timestamp'] = df['ts_utc'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return correlation_data.to_csv(index=False).encode('utf-8')

# Export buttons will be added after data loading

st.sidebar.markdown("---")
st.sidebar.header("🔍 Global Filters")

# Confidence Slider
min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.05)

# Speaker Filter
@st.cache_data(ttl=5)
def get_unique_speakers(path):
    if not Path(path).exists():
        return []
    try:
        # Try to read with header first
        try:
            df_temp = pd.read_csv(path)
        except:
            # If that fails, try reading without header and assign column names
            df_temp = pd.read_csv(path, header=None, names=[
                "ts_utc", "file", "speaker", "pred_class", "confidence", 
                "entropy", "rms", "text", "prob_happy", "prob_angry", 
                "prob_fear", "prob_calm", "prob_sad", "prob_surprise"
            ])
        
        if 'speaker' in df_temp.columns:
            return sorted(df_temp['speaker'].fillna('Unknown').unique().tolist())
    except:
        pass
    return []

unique_speakers = get_unique_speakers(csv_path)
selected_speakers = st.sidebar.multiselect("Filter by Speaker", options=unique_speakers, default=unique_speakers)

@st.cache_data(ttl=5)
def load_data(path, min_conf_filter, speaker_filter):
    if not Path(path).exists():
        return None
    try:
        # Try to read with header first
        try:
            df = pd.read_csv(path)
        except:
            # If that fails, try reading without header and assign column names
            df = pd.read_csv(path, header=None, names=[
                "ts_utc", "file", "speaker", "pred_class", "confidence", 
                "entropy", "rms", "text", "prob_happy", "prob_angry", 
                "prob_fear", "prob_calm", "prob_sad", "prob_surprise"
            ])
        
        # Ensure timestamp is datetime - handle extra spaces and ISO format
        df['ts_utc'] = pd.to_datetime(df['ts_utc'].str.strip(), format='ISO8601')
        # Sort by timestamp
        df = df.sort_values('ts_utc')
        # Clean speaker names
        df['speaker'] = df['speaker'].fillna('Unknown').replace('', 'Unknown')
        
        # Apply Filters
        df = df[df['confidence'] >= min_conf_filter]
        if speaker_filter:
            df = df[df['speaker'].isin(speaker_filter)]
        
        # Auto-generate experiment data if new data detected
        if len(df) > 0:
            auto_generate_experiment_data(df)
            
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def auto_generate_experiment_data(df):
    """Auto-generate experiment data for t-SNE when new CSV data arrives"""
    try:
        # Get current experiment data timestamp
        exp_file = Path("experiment_data.npz")
        current_mtime = exp_file.stat().st_mtime if exp_file.exists() else 0
        
        # Get latest CSV data timestamp
        latest_csv_time = df['ts_utc'].max().timestamp() if len(df) > 0 else 0
        
        # If CSV data is newer, regenerate experiment data
        if latest_csv_time > current_mtime:
            import subprocess
            result = subprocess.run(
                [".venv311/bin/python", "generate_experiment_data.py"],
                capture_output=True,
                text=True,
                cwd="."
            )
            # Don't show messages here to avoid cluttering the UI
    except:
        pass  # Silent fail for auto-generation

# Load Data with Filters
df = load_data(csv_path, min_conf, selected_speakers)

# Export buttons (now that df is defined)
if df is not None and len(df) > 0:
    # Comprehensive Export
    st.sidebar.markdown("**📦 Complete Export**")
    
    # Generate ZIP data once and provide download
    zip_data = create_export_zip()
    st.sidebar.download_button(
        label="⬇️ Download Complete Analysis (ZIP)",
        data=zip_data.getvalue(),
        file_name=f"speech_analysis_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        type="primary"
    )
    
    st.sidebar.markdown("**📊 Individual Exports**")
    
    # Individual export buttons
    exports = [
        ("📈 Timeline Analysis", export_timeline_data, "timeline_analysis.csv"),
        ("📊 Confidence Trends", export_confidence_trends, "confidence_trends.csv"),
        ("😊 Emotion Breakdown", export_emotion_breakdown, "emotion_breakdown.csv"),
        ("🔥 Emotion Heatmap", export_emotion_heatmap, "emotion_heatmap.csv"),
        ("👤 Speaker Talk Time", export_speaker_talk_time, "speaker_talk_time.csv"),
        ("🎭 Emotion Matrix", export_emotion_matrix, "emotion_matrix.csv"),
        ("📈 Sentiment Volatility", export_volatility, "sentiment_volatility.csv"),
        ("🔄 Interaction Matrix", export_interaction_matrix, "interaction_matrix.csv"),
        ("🔬 t-SNE Data", export_t_sne, "t_sne_data.csv"),
        ("🕸️ Radar Comparison", export_radar_comparison, "radar_comparison.csv"),
        ("🔊 Audio X-Ray", export_audio_xray, "audio_xray_analysis.txt"),
        ("📉 Correlation Data", export_correlation, "correlation_data.csv")
    ]
    
    for label, export_func, filename in exports:
        data = export_func()
        st.sidebar.download_button(
            label=f"⬇️ {label}",
            data=data,
            file_name=filename,
            mime="text/csv" if filename.endswith('.csv') else "text/plain",
            key=f"download_{label.replace(' ', '_').lower()}"
        )

# Debug information
st.sidebar.write(f"**Debug Info:**")
st.sidebar.write(f"• CSV path: {csv_path}")
st.sidebar.write(f"• File exists: {Path(csv_path).exists()}")
st.sidebar.write(f"• df is None: {df is None}")
if df is not None:
    st.sidebar.write(f"• df shape: {df.shape}")
    st.sidebar.write(f"• df empty: {df.empty}")
    st.sidebar.write(f"• Min confidence: {min_conf}")
    st.sidebar.write(f"• Selected speakers: {len(selected_speakers)}")

if df is None:
    st.info(f"Waiting for CSV data at `{csv_path}`... Please run the analysis script with `--log-csv {csv_path}`")
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
elif df.empty:
    st.warning("⚠️ Data loaded but empty after filtering. Try adjusting confidence threshold or speaker filters.")
    st.write(f"**Current filters:**")
    st.write(f"• Minimum confidence: {min_conf}")
    st.write(f"• Selected speakers: {selected_speakers}")
else:
    # --- 1. Top Metrics ---
    st.markdown("---")
    st.markdown('<h1 class="main-header">📊 Analysis Overview</h1>', unsafe_allow_html=True)
    
    total_segments = len(df)
    
    # Clean speaker names
    df['speaker'] = df['speaker'].fillna('Unknown').replace('', 'Unknown')
    speakers = df['speaker'].unique()
    valid_speakers = [s for s in speakers if s not in ['Unknown', 'N/A', '']]
    num_speakers = len(valid_speakers)
    
    top_emotion = df['pred_class'].mode()[0] if not df.empty else "N/A"
    avg_conf = df['confidence'].mean()
    
    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #1f77b4;">{total_segments:,}</h3>
            <p style="margin: 0; color: #666;">Total Segments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #27ae60;">{num_speakers}</h3>
            <p style="margin: 0; color: #666;">Unique Speakers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #e74c3c;">{top_emotion}</h3>
            <p style="margin: 0; color: #666;">Top Emotion</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #f39c12;">{avg_conf:.1%}</h3>
            <p style="margin: 0; color: #666;">Avg Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    # --- 2. Main Visualizations ---
    if FORENSIC_AVAILABLE:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🕒 Timeline Analysis", 
            "📊 Emotion Breakdown", 
            "👥 Speaker Analytics", 
            "🕸️ Radar Comparison",
            "🔤 Keyword Analysis",
            "📈 Correlation",
            "📝 Detailed Transcript",
            "🔬 Forensic Analysis"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🕒 Timeline Analysis", 
            "📊 Emotion Breakdown", 
            "👥 Speaker Analytics", 
            "🕸️ Radar Comparison",
            "🔤 Keyword Analysis",
            "📈 Correlation",
            "📝 Detailed Transcript"
        ])

    with tab1:
        st.subheader("🕒 Timeline Analysis")
        st.markdown("Track sentiment changes over time with confidence filtering.")
        
        # Multi-speaker timeline with better styling
        fig_timeline = px.scatter(
            df, 
            x="ts_utc", 
            y="speaker", 
            color="pred_class",
            size="confidence",
            hover_data=["text", "confidence", "rms"],
            title="📈 Emotional Shifts Over Time (per Speaker)",
            labels={"ts_utc": "Time (UTC)", "speaker": "Speaker ID", "pred_class": "Emotion"},
            height=500,
            color_discrete_map={
                "happy": "#FFD700", "angry": "#FF4B4B", "fear": "#9370DB", 
                "calm": "#90EE90", "sad": "#4682B4", "surprise": "#FFA500"
            },
            template='plotly_white'
        )
        fig_timeline.update_layout(
            font=dict(size=12),
            title_font=dict(size=16, color='#2c3e50'),
            xaxis_title="Time (UTC)",
            yaxis_title="Speaker",
            legend_title="Emotion",
            hovermode='closest'
        )
        fig_timeline.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig_timeline, width="stretch", use_container_width=True)

        # Emotion Stacked Area Chart
        st.subheader("Emotion Distribution over Time")
        # Group by time bucket if data is too dense, but for streaming we can use raw or rolling average

        # Confidence/Entropy Chart
        st.subheader("📊 Confidence & Uncertainty Trends")
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            x=df['ts_utc'], 
            y=df['confidence'], 
            name="Confidence", 
            line=dict(color="#00CC96", width=2)
        ))
        fig_conf.add_trace(go.Scatter(
            x=df['ts_utc'], 
            y=df['entropy'], 
            name="Entropy (Uncertainty)", 
            line=dict(color="#EF553B", width=2, dash='dash')
        ))
        fig_conf.update_layout(
            title="Model Confidence vs Entropy Over Time",
            xaxis_title="Time",
            yaxis_title="Score",
            template='plotly_white',
            font=dict(size=12),
            title_font=dict(size=16, color='#2c3e50'),
            legend=dict(x=0, y=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig_conf, width="stretch", use_container_width=True)

    with tab2:
        st.subheader("📊 Emotion Breakdown")
        st.markdown("Analyze emotion distribution and patterns across all segments.")
        
        col_pie, col_bar = st.columns(2)
        
        with col_pie:
            st.markdown("**🥧 Total Emotion Share**")
            fig_pie = px.pie(
                df, 
                names='pred_class', 
                hole=0.4, 
                color='pred_class',
                color_discrete_map={
                    "happy": "#FFD700", "angry": "#FF4B4B", "fear": "#9370DB", 
                    "calm": "#90EE90", "sad": "#4682B4", "surprise": "#FFA500"
                },
                template='plotly_white'
            )
            fig_pie.update_layout(
                font=dict(size=12),
                title_font=dict(size=14, color='#2c3e50'),
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01)
            )
            st.plotly_chart(fig_pie, width="stretch", use_container_width=True)
        
        with col_bar:
            st.markdown("**📈 Emotion Frequency**")
            emotion_counts = df['pred_class'].value_counts().reset_index()
            emotion_counts.columns = ['emotion', 'count']
            fig_bar = px.bar(
                emotion_counts, 
                x='emotion', 
                y='count', 
                color='emotion',
                color_discrete_map={
                    "happy": "#FFD700", "angry": "#FF4B4B", "fear": "#9370DB", 
                    "calm": "#90EE90", "sad": "#4682B4", "surprise": "#FFA500"
                },
                template='plotly_white'
            )
            fig_bar.update_layout(
                font=dict(size=12),
                title_font=dict(size=14, color='#2c3e50'),
                xaxis_title="Emotion",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig_bar, width="stretch", use_container_width=True)

        # Probabilities Heatmap (Top Emotions)
        st.subheader("🔥 Emotion Probability Heatmap")
        prob_cols = [c for c in df.columns if c.startswith('prob_')]
        if prob_cols:
            heatmap_data = df[prob_cols].T
            heatmap_data.index = [c.replace('prob_', '').capitalize() for c in heatmap_data.index]
            fig_heat = px.imshow(
                heatmap_data, 
                labels=dict(x="Segment Index", y="Emotion", color="Probability"),
                x=df.index, 
                color_continuous_scale='Viridis',
                template='plotly_white'
            )
            fig_heat.update_layout(
                font=dict(size=12),
                title_font=dict(size=14, color='#2c3e50'),
                height=400
            )
            st.plotly_chart(fig_heat, width="stretch", use_container_width=True)

    with tab3:
        st.subheader("Speaker Dominance & Sentiment Matrix")
        
        col_spk1, col_spk2 = st.columns(2)
        
        with col_spk1:
            st.markdown("**Speaker Talk Time (Segments)**")
            spk_counts = df['speaker'].value_counts().reset_index()
            spk_counts.columns = ['speaker', 'segments']
            fig_spk_pie = px.pie(spk_counts, values='segments', names='speaker', title="Speaker Participation")
            st.plotly_chart(fig_spk_pie, width="stretch")
            
        with col_spk2:
            st.markdown("**Emotion Matrix per Speaker**")
            speaker_emotion = df.groupby(['speaker', 'pred_class']).size().unstack(fill_value=0)
            st.dataframe(speaker_emotion, width="stretch")

        # Sentiment Volatility (How much emotions change)
        st.subheader("Sentiment Volatility")
        # Ensure we only calculate for speakers with > 1 segment
        df['emo_change'] = df.groupby('speaker')['pred_class'].transform(lambda x: x != x.shift()).astype(int)
        volatility = df.groupby('speaker')['emo_change'].mean().reset_index()
        volatility.columns = ['speaker', 'volatility_score']
        fig_vol = px.bar(volatility, x='speaker', y='volatility_score', 
                        title="Speaker Emotional Volatility (Higher = More Shifts)",
                        labels={'volatility_score': 'Volatility Index'})
        st.plotly_chart(fig_vol, width="stretch")

        # Speaker Transition Matrix
        st.subheader("🔄 Speaker Interaction (Transition Matrix)")
        st.markdown("Shows who spoke after whom. Darker cells indicate more frequent turn-taking.")
        if len(df) > 1:
            df['next_speaker'] = df['speaker'].shift(-1)
            # Remove last row where next_speaker is NaN
            trans_df = df.dropna(subset=['next_speaker'])
            matrix = pd.crosstab(trans_df['speaker'], trans_df['next_speaker'])
            
            # Create larger, more visible heatmap
            fig_trans = px.imshow(
                matrix, 
                text_auto=True, 
                color_continuous_scale='Blues',
                labels=dict(x="Spoke Next", y="Spoke First", color="Turns"),
                title="Speaker Turn-Taking Frequency",
                template='plotly_white',
                height=800,  # Increased height significantly
                width=1000    # Increased width significantly
            )
            
            fig_trans.update_layout(
                font=dict(size=16),  # Increased font size for larger matrix
                title_font=dict(size=22, color='#2c3e50'),  # Larger title for larger matrix
                xaxis_title="Spoke Next",
                yaxis_title="Spoke First"
            )
            
            # Add annotations for better readability
            fig_trans.update_traces(
                textfont={"size": 14},  # Larger text in cells for bigger matrix
                hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>Transitions: %{z}<extra></extra>"
            )
            
            st.plotly_chart(fig_trans, width="stretch", use_container_width=True)
        else:
            st.info("Not enough segments to calculate transitions.")
        
        # t-SNE Speaker Identity Visualization
        st.markdown("---")
        st.subheader("🔬 Speaker Identity Separation (t-SNE)")
        
        # Check if t-SNE plot exists or should be shown
        tsne_plot_file = Path("speaker_tsne_plot.png")
        
        # Auto-generate t-SNE if data has changed
        if auto_refresh and tsne_plot_file.exists():
            # Check if experiment data is newer than t-SNE plot
            exp_file = Path("experiment_data.npz")
            if exp_file.exists():
                exp_mtime = exp_file.stat().st_mtime
                tsne_mtime = tsne_plot_file.stat().st_mtime
                
                # If experiment data is newer, regenerate t-SNE
                if exp_mtime > tsne_mtime:
                    st.info("🔄 New data detected. Regenerating t-SNE...")
                    try:
                        import subprocess
                        result = subprocess.run(
                            [".venv311/bin/python", "visualize_paper.py", "--input", "experiment_data.npz", "--output", "speaker_tsne_plot.png"],
                            capture_output=True,
                            text=True,
                            cwd="."
                        )
                        
                        if result.returncode == 0 and tsne_plot_file.exists():
                            st.success("✅ t-SNE updated automatically!")
                        else:
                            st.warning("⚠️ Auto-update failed. Click generate button manually.")
                    except Exception as e:
                        st.warning(f"⚠️ Auto-update error: {str(e)}")
        
        # Simple check - if file exists, show it
        if tsne_plot_file.exists():
            st.success("🎯 t-SNE plot found! Displaying visualization...")
            
            try:
                # Read and display the image
                with open(tsne_plot_file, "rb") as image_file:
                    image_bytes = image_file.read()
                    st.image(image_bytes, caption="Speaker Identity Separation (t-SNE Visualization)", width="stretch")
                
                st.info("📊 This t-SNE plot shows how well speakers are separated in the embedding space. Tight clusters indicate consistent speaker identity, while scattered points may indicate voice cloning or synthesis.")
                
                # Show file info
                st.write(f"**File Info:**")
                st.write(f"• File size: {tsne_plot_file.stat().st_size / 1024:.1f} KB")
                st.write(f"• Last updated: {tsne_plot_file.stat().st_mtime}")
                
                # Auto-refresh indicator
                if auto_refresh:
                    st.write("🔄 Auto-refresh enabled - t-SNE will update when new data arrives")
                
            except Exception as e:
                st.error(f"❌ Error displaying plot: {str(e)}")
                st.write(f"Debug: File path = {tsne_plot_file.absolute()}")
        else:
            st.info("🔬 Generate t-SNE plot to visualize speaker identity separation.")
            
            # Show experiment data info if available
            if Path("experiment_data.npz").exists():
                try:
                    experiment_data = np.load("experiment_data.npz")
                    if 'embeddings' in experiment_data:
                        embeddings = experiment_data['embeddings']
                        speakers = experiment_data.get('speakers', ['Unknown'] * len(embeddings))
                        
                        st.write(f"**Available Research Data:**")
                        st.write(f"• Embeddings: {len(embeddings)} samples")
                        st.write(f"• Unique Speakers: {len(set(speakers))}")
                        st.write(f"• Embedding Dimension: {embeddings.shape[1] if len(embeddings.shape) > 1 else 'N/A'}")
                        
                        # Generate t-SNE button
                        if st.button("🔍 Generate t-SNE Visualization", key="main_tsne_btn"):
                            st.info("Computing t-SNE... Please wait.")
                            
                            import subprocess
                            try:
                                result = subprocess.run(
                                    [".venv311/bin/python", "visualize_paper.py", "--input", "experiment_data.npz", "--output", "speaker_tsne_plot.png"],
                                    capture_output=True,
                                    text=True,
                                    cwd="."
                                )
                                
                                if result.returncode == 0 and tsne_plot_file.exists():
                                    st.success("t-SNE plot generated successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to generate t-SNE: {result.stderr}")
                            except Exception as e:
                                st.error(f"Error running t-SNE generation: {str(e)}")
                except Exception as e:
                    st.warning(f"Could not load experiment data: {str(e)}")
            else:
                st.info("📝 No research data found. Run analysis with `--diarization` to generate speaker embeddings for t-SNE visualization.")

    with tab4:
        st.subheader("Emotion Radar Comparison")
        st.markdown("Compare the 'Emotional Fingerprint' of different speakers.")
        
        prob_cols = [c for c in df.columns if c.startswith('prob_')]
        if prob_cols:
            # Aggregate probabilities per speaker
            radar_df = df.groupby('speaker')[prob_cols].mean().reset_index()
            
            # Select speakers to compare
            selected_radars = st.multiselect("Select Speakers to Compare", options=radar_df['speaker'].unique(), default=radar_df['speaker'].unique()[:3])
            
            if selected_radars:
                fig_radar = go.Figure()
                categories = [c.replace('prob_', '').capitalize() for c in prob_cols]
                
                for spk in selected_radars:
                    row = radar_df[radar_df['speaker'] == spk].iloc[0]
                    values = [row[c] for c in prob_cols]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=spk
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Average Emotional Intensity per Speaker"
                )
                st.plotly_chart(fig_radar, width="stretch")
            else:
                st.warning("Please select at least one speaker to display radar chart.")
        else:
            st.info("Probability data (prob_*) not found in CSV. Run analysis with full class logging.")
        
        # Audio X-Ray Test Analysis
        st.markdown("---")
        st.subheader("🔬 Audio X-Ray RPCA Analysis")
        st.markdown("Forensic audio analysis using Robust PCA to detect synthetic voice characteristics.")
        
        # Check for Audio X-Ray test results
        xray_test_dir = Path("audio_xray_test")
        rpca_analysis_file = xray_test_dir / "test_sample_rpca_analysis.png"
        artifact_detail_file = xray_test_dir / "test_sample_artifact_detail.png"
        report_file = xray_test_dir / "test_sample_report.txt"
        
        # Auto-generate Audio X-Ray if data has changed
        if auto_refresh and not rpca_analysis_file.exists():
            st.info("🔄 Running Audio X-Ray analysis on new CSV data...")
            try:
                import subprocess
                result = subprocess.run(
                    [".venv311/bin/python", "run_xray_csv.py", csv_path],
                    capture_output=True,
                    text=True,
                    cwd="."
                )
                
                if result.returncode == 0 and rpca_analysis_file.exists():
                    st.success("✅ Audio X-Ray analysis completed automatically!")
                else:
                    st.warning("⚠️ Auto-analysis failed. Run manually if needed.")
            except Exception as e:
                st.warning(f"⚠️ Auto-analysis error: {str(e)}")
        
        # Check for both test and CSV analysis files
        csv_analysis_file = xray_test_dir / "csv_rpca_analysis.png"
        csv_report_file = xray_test_dir / "csv_analysis_report.txt"
        
        # Use CSV analysis if available, otherwise use test analysis
        analysis_file = csv_analysis_file if csv_analysis_file.exists() else rpca_analysis_file
        report_file_to_use = csv_report_file if csv_report_file.exists() else report_file
        
        if analysis_file.exists():
            # Display Audio X-Ray results
            col_xray1, col_xray2 = st.columns(2)
            
            with col_xray1:
                st.markdown("**📊 RPCA Spectrogram Analysis**")
                st.image(str(analysis_file), caption="RPCA Decomposition: Low-rank (voice) vs Sparse (artifacts)", width="stretch")
            
            with col_xray2:
                if artifact_detail_file.exists():
                    st.markdown("**🔍 Artifact Detail View**")
                    st.image(str(artifact_detail_file), caption="Sparse Component Analysis - Artifact Detection", width="stretch")
            
            # Display analysis report
            if report_file_to_use.exists():
                st.markdown("**📋 Forensic Analysis Report**")
                try:
                    with open(report_file_to_use, 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    
                    # Parse and display key metrics
                    lines = report_content.split('\n')
                    metrics_data = {}
                    
                    for line in lines:
                        if 'Average Deepfake Likelihood:' in line:
                            metrics_data['Deepfake Likelihood'] = line.split(':')[1].strip()
                        elif 'Average Rhythmicity Score:' in line:
                            metrics_data['Rhythmicity Score'] = line.split(':')[1].strip()
                        elif 'Average Silence Ratio:' in line:
                            metrics_data['Silence Ratio'] = line.split(':')[1].strip()
                        elif 'HIGH RISK:' in line:
                            metrics_data['Risk Level'] = 'HIGH'
                        elif 'MEDIUM RISK:' in line:
                            metrics_data['Risk Level'] = 'MEDIUM'
                        elif 'LOW RISK:' in line:
                            metrics_data['Risk Level'] = 'LOW'
                    
                    # Display metrics in columns
                    if metrics_data:
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        
                        with metrics_col1:
                            if 'Deepfake Likelihood' in metrics_data:
                                likelihood_str = metrics_data['Deepfake Likelihood'].replace('%', '')
                                try:
                                    likelihood = float(likelihood_str)
                                    color = "🔴" if likelihood > 50 else "🟡" if likelihood > 25 else "🟢"
                                    st.metric(f"Deepfake Risk {color}", f"{likelihood:.1f}%")
                                except:
                                    st.metric("Deepfake Risk", metrics_data['Deepfake Likelihood'])
                        
                        with metrics_col2:
                            if 'Silence Ratio' in metrics_data:
                                silence_str = metrics_data['Silence Ratio'].replace('%', '')
                                try:
                                    silence = float(silence_str)
                                    st.metric("Silence Ratio", f"{silence:.1f}%")
                                except:
                                    st.metric("Silence Ratio", metrics_data['Silence Ratio'])
                        
                        with metrics_col3:
                            if 'Risk Level' in metrics_data:
                                risk = metrics_data['Risk Level']
                                if risk == 'HIGH':
                                    st.error(f"🔴 **Risk Level:** {risk}")
                                elif risk == 'MEDIUM':
                                    st.warning(f"🟡 **Risk Level:** {risk}")
                                else:
                                    st.success(f"🟢 **Risk Level:** {risk}")
                    
                    # Show detailed report in expander
                    with st.expander("📄 View Full Analysis Report"):
                        st.text(report_content)
                    
                    # Auto-refresh indicator
                    if auto_refresh:
                        st.write("🔄 Auto-refresh enabled - Audio X-Ray will update when new data arrives")
                        
                except Exception as e:
                    st.error(f"Error loading Audio X-Ray report: {str(e)}")
            
            # Add Audio X-Ray info
            with st.expander("ℹ️ About Audio X-Ray Analysis"):
                st.markdown("""
                **Audio X-Ray RPCA** (Robust Principal Component Analysis) decomposes audio spectrograms into:
                
                - **Low-rank component**: Natural voice patterns and harmonic structure
                - **Sparse component**: Synthetic artifacts, compression artifacts, and anomalies
                
                **Detection Indicators:**
                - High deepfake likelihood percentages
                - Unusual rhythmicity patterns
                - Excessive silence (missing natural breaths)
                - Abnormal artifact density in sparse component
                
                **Applications:**
                - Deepfake detection
                - Voice cloning identification  
                - Audio authenticity verification
                - Forensic evidence analysis
                
                **Current Analysis:** Based on your live CSV data with synthetic audio generation
                """)
            
            # Download Audio X-Ray files
            st.markdown("**📥 Download Analysis Files**")
            download_xray_col1, download_xray_col2 = st.columns(2)
            
            with download_xray_col1:
                if analysis_file.exists():
                    with open(analysis_file, "rb") as f:
                        st.download_button(
                            label="📊 RPCA Analysis Chart",
                            data=f.read(),
                            file_name="audio_xray_rpca_analysis.png",
                            mime="image/png"
                        )
            
            with download_xray_col2:
                if report_file_to_use.exists():
                    with open(report_file_to_use, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="📋 Analysis Report",
                            data=f.read(),
                            file_name="audio_xray_report.txt",
                            mime="text/plain"
                        )
        
        else:
            st.info("🔬 Audio X-Ray analysis results not found. Running analysis...")
            if auto_refresh:
                st.write("🔄 Auto-refresh is enabled - Audio X-Ray will run automatically")
            else:
                st.code("python run_xray_csv.py analysis_log.csv", language="bash")
            
            # Manual run button
            if st.button("🔍 Run Audio X-Ray Analysis", key="manual_xray_btn"):
                st.info("Running Audio X-Ray analysis on CSV data...")
                try:
                    import subprocess
                    result = subprocess.run(
                        [".venv311/bin/python", "run_xray_csv.py", csv_path],
                        capture_output=True,
                        text=True,
                        cwd="."
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ Audio X-Ray analysis completed!")
                        st.rerun()
                    else:
                        st.error(f"Analysis failed: {result.stderr}")
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")

    with tab5:
        st.subheader("Keyword Analysis (ASR WordCloud)")
        
        # Combine all text
        all_text = " ".join(df['text'].dropna().astype(str))
        if all_text.strip():
            # Filter by speaker if needed
            spk_filter = st.selectbox("Select Speaker for WordCloud", ["All"] + list(df['speaker'].unique()))
            if spk_filter != "All":
                all_text = " ".join(df[df['speaker'] == spk_filter]['text'].dropna().astype(str))
            
            if all_text.strip():
                if WordCloud:
                    if word_tokenize:
                        # Thai normalization and tokenization
                        if thai_normalize:
                            all_text = thai_normalize(all_text)
                        tokens = word_tokenize(all_text, engine="newmm")
                        # Filter out stopwords if available
                        stop_list = list(thai_stopwords()) if thai_stopwords else []
                        filtered = [t for t in tokens if len(t) > 1 and t not in stop_list]
                        text_for_cloud = " ".join(filtered)
                    else:
                        text_for_cloud = all_text

                    # Generate WordCloud
                    try:
                        wc = WordCloud(
                            font_path=None, # System default or provide a Thai font path if needed
                            width=800, height=400, 
                            background_color="white",
                            colormap="viridis"
                        ).generate(text_for_cloud)
                        
                        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                    except Exception as e:
                        st.warning(f"Could not generate WordCloud: {e}")
                else:
                    st.warning("`wordcloud` library is not installed. Please run `pip install wordcloud` to see keywords.")
            else:
                st.info("No text found for selected speaker.")
        else:
            st.info("No ASR transcription data found in CSV. Use `--asr thai` during analysis.")

    with tab6:
        st.subheader("Sentiment vs. Acoustic Correlation")
        st.markdown("Analyze how voice volume (RMS) correlates with specific emotions.")
        
        if 'rms' in df.columns:
            # Box plot of RMS by Emotion
            fig_corr = px.box(df, x="pred_class", y="rms", color="pred_class",
                             points="all", title="Volume (RMS) Distribution by Emotion",
                             color_discrete_map={
                                 "happy": "#FFD700", "angry": "#FF4B4B", "fear": "#9370DB", 
                                 "calm": "#90EE90", "sad": "#4682B4", "surprise": "#FFA500"
                             })
            st.plotly_chart(fig_corr, width="stretch")
            
            # Confidence vs RMS Scatter
            st.subheader("Confidence vs Volume")
            fig_conf_rms = px.scatter(df, x="rms", y="confidence", color="pred_class", 
                                     hover_data=["speaker", "text"],
                                     title="Model Confidence relative to Vocal Intensity")
            st.plotly_chart(fig_conf_rms, width="stretch")
        else:
            st.info("RMS data not found in CSV.")

    with tab7:
        st.subheader("Transcription & Sentiment Log")
        
        # Display as a searchable table
        st.dataframe(
            df[['ts_utc', 'speaker', 'pred_class', 'confidence', 'text', 'rms']].sort_values('ts_utc', ascending=False),
            column_config={
                "ts_utc": "Timestamp",
                "speaker": "Speaker",
                "pred_class": "Emotion",
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                "rms": st.column_config.NumberColumn("Volume (RMS)", format="%.4f"),
                "text": "Transcribed Text"
            },
            width="stretch",
            hide_index=True
        )

    # Add forensic analysis tab if available
    if FORENSIC_AVAILABLE:
        with tab8:
            st.subheader("🔬 Forensic Deepfake Detection Analysis")
            
            # Forensic analysis controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                audio_file = st.text_input("Audio File Path (optional)", value="", 
                                         help="Path to audio file for detailed analysis")
            
            with col2:
                suspicion_threshold = st.slider("Suspicion Threshold", 0.0, 1.0, 0.3, 0.05,
                                              help="Threshold for identifying suspicious segments")
            
            with col3:
                run_forensic = st.button("🔍 Run Forensic Analysis", type="primary")
            
            # Check for existing forensic results
            forensic_output_dir = Path("forensic_output")
            forensic_summary_file = forensic_output_dir / "forensic_summary.json"
            forensic_dossier_file = forensic_output_dir / "forensic_dossier.json"
            
            # Display existing results if available
            if forensic_summary_file.exists():
                st.markdown("---")
                st.subheader("📋 Existing Forensic Analysis Results")
                
                try:
                    with open(forensic_summary_file, 'r', encoding='utf-8') as f:
                        forensic_data = json.load(f)
                    
                    # Analysis Overview
                    overview = forensic_data.get("analysis_overview", {})
                    key_findings = forensic_data.get("key_findings", {})
                    risk_assessment = forensic_data.get("risk_assessment", {})
                    
                    # Key Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Segments", overview.get("total_segments_analyzed", 0))
                    
                    with col2:
                        st.metric("Suspicious Segments", overview.get("suspicious_segments_found", 0))
                    
                    with col3:
                        suspicion_rate = overview.get("overall_suspicion_rate", 0)
                        st.metric("Suspicion Rate", f"{suspicion_rate:.1%}")
                    
                    with col4:
                        risk_level = risk_assessment.get("risk_level", "UNKNOWN")
                        risk_color = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"
                        st.metric("Risk Level", f"{risk_color} {risk_level}")
                    
                    # Risk Assessment
                    st.markdown("---")
                    st.subheader("⚠️ Risk Assessment")
                    
                    risk_col1, risk_col2 = st.columns(2)
                    with risk_col1:
                        st.info(f"**Risk Level:** {risk_level}")
                        st.info(f"**Suspicion Rate:** {suspicion_rate:.1%}")
                        st.info(f"**Confidence:** {risk_assessment.get('confidence', 'unknown').title()}")
                    
                    with risk_col2:
                        st.warning(f"**Description:** {risk_assessment.get('risk_description', 'No description')}")
                    
                    # Key Findings
                    st.markdown("---")
                    st.subheader("🔍 Key Findings")
                    
                    findings_col1, findings_col2 = st.columns(2)
                    
                    with findings_col1:
                        st.write("**Suspicious Speakers:**")
                        suspicious_speakers = key_findings.get("primary_suspicious_speakers", [])
                        if suspicious_speakers:
                            # Show top 10 speakers
                            for speaker in suspicious_speakers[:10]:
                                st.write(f"• {speaker}")
                            if len(suspicious_speakers) > 10:
                                st.write(f"... and {len(suspicious_speakers) - 10} more")
                        else:
                            st.write("None detected")
                    
                    with findings_col2:
                        st.write("**Detection Indicators:**")
                        indicators = key_findings.get("most_common_indicators", {})
                        for indicator, rate in indicators.items():
                            indicator_name = indicator.replace("_", " ").title()
                            st.write(f"• **{indicator_name}**: {rate:.1%}")
                    
                    # Forensic Metrics
                    st.markdown("---")
                    st.subheader("📊 Forensic Metrics Summary")
                    
                    metrics = forensic_data.get("forensic_metrics_summary", {})
                    if metrics:
                        metrics_df = []
                        for metric_name, values in metrics.items():
                            if isinstance(values, dict) and "mean" in values:
                                metrics_df.append({
                                    "Metric": metric_name.replace("_", " ").title(),
                                    "Mean": values.get("mean", 0),
                                    "Std Dev": values.get("std", 0),
                                    "Threshold": values.get("threshold", 0)
                                })
                        
                        if metrics_df:
                            metrics_display_df = pd.DataFrame(metrics_df)
                            st.dataframe(metrics_display_df, use_container_width=True, hide_index=True)
                    
                    # Recommendations
                    st.markdown("---")
                    st.subheader("💡 Recommendations")
                    
                    recommendations = forensic_data.get("recommendations", [])
                    if recommendations:
                        for i, rec in enumerate(recommendations, 1):
                            if rec.startswith("HIGH RISK:"):
                                st.error(f"{i}. {rec}")
                            elif "Recommendation:" in rec:
                                st.warning(f"{i}. {rec}")
                            else:
                                st.info(f"{i}. {rec}")
                    
                    # Visual Evidence
                    st.markdown("---")
                    st.subheader("🖼️ Visual Evidence")
                    
                    # Check for forensic visualization files
                    cluster_plot = forensic_output_dir / "forensic_clusters.png"
                    if cluster_plot.exists():
                        st.image(str(cluster_plot), caption="Forensic Cluster Analysis", use_column_width=True)
                    
                    # Download forensic reports
                    st.markdown("---")
                    st.subheader("📥 Download Forensic Reports")
                    
                    download_col1, download_col2 = st.columns(2)
                    
                    with download_col1:
                        if forensic_summary_file.exists():
                            with open(forensic_summary_file, 'r', encoding='utf-8') as f:
                                summary_data = f.read()
                            st.download_button(
                                label="📄 Download Summary Report",
                                data=summary_data,
                                file_name="forensic_summary.json",
                                mime="application/json"
                            )
                    
                    with download_col2:
                        if forensic_dossier_file.exists():
                            with open(forensic_dossier_file, 'r', encoding='utf-8') as f:
                                dossier_data = f.read()
                            st.download_button(
                                label="📋 Download Detailed Dossier",
                                data=dossier_data,
                                file_name="forensic_dossier.json",
                                mime="application/json"
                            )
                    
                except Exception as e:
                    st.error(f"❌ Error loading forensic results: {str(e)}")
            
            else:
                st.info("📝 No existing forensic analysis results found. Run forensic analysis to generate results.")
            
            if run_forensic:
                with st.spinner("Performing forensic analysis..."):
                    try:
                        # Initialize forensic pipeline
                        config = ForensicConfig()
                        config.create_plots = False  # We'll handle plots in Streamlit
                        pipeline = ForensicPipeline(config)
                        
                        # Run analysis
                        results = pipeline.analyze_csv_data(csv_path, audio_file if audio_file else None)
                        
                        if results["status"] == "completed":
                            st.success("✅ Forensic analysis completed successfully!")
                            
                            # Display summary metrics
                            st.subheader("📊 Analysis Summary")
                            summary = results["summary"]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Segments", summary["analysis_overview"]["total_segments_analyzed"])
                            
                            with col2:
                                st.metric("Suspicious Segments", summary["analysis_overview"]["suspicious_segments_found"])
                            
                            with col3:
                                st.metric("Suspicion Rate", f"{summary['analysis_overview']['overall_suspicion_rate']:.1%}")
                            
                            with col4:
                                risk_level = summary["risk_assessment"]["risk_level"]
                                risk_color = {
                                    "HIGH": "red",
                                    "MEDIUM": "orange", 
                                    "LOW": "green",
                                    "MINIMAL": "blue"
                                }.get(risk_level, "gray")
                                st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
                            
                            # Display key findings
                            st.subheader("🔍 Key Findings")
                            
                            findings = summary["key_findings"]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Suspicious Speakers:**")
                                for speaker in findings["primary_suspicious_speakers"]:
                                    st.write(f"• {speaker}")
                                
                                st.write("**Average Suspicion Score:**")
                                st.write(f"{findings['average_suspicion_score']:.3f}")
                            
                            with col2:
                                st.write("**Most Common Indicators:**")
                                for indicator, rate in findings["most_common_indicators"].items():
                                    indicator_name = indicator.replace('_', ' ').title()
                                    st.write(f"• {indicator_name}: {rate:.1%}")
                                
                                st.write("**Max Suspicion Score:**")
                                st.write(f"{findings['max_suspicion_score']:.3f}")
                            
                            # Speaker Transition Matrix
                            st.subheader("🔄 Speaker Interaction (Transition Matrix)")
                            if len(df) > 1:
                                # Create speaker transition matrix
                                transitions = {}
                                for i in range(1, len(df)):
                                    prev_speaker = df.iloc[i-1]['speaker']
                                    curr_speaker = df.iloc[i]['speaker']
                                    if prev_speaker != curr_speaker:
                                        if prev_speaker not in transitions:
                                            transitions[prev_speaker] = {}
                                        transitions[prev_speaker][curr_speaker] = transitions[prev_speaker].get(curr_speaker, 0) + 1
                                
                                if transitions:
                                    # Create matrix for heatmap
                                    speakers = sorted(set(transitions.keys()).union(*[set(v.keys()) for v in transitions.values()]))
                                    matrix = np.zeros((len(speakers), len(speakers)))
                                    
                                    for i, from_speaker in enumerate(speakers):
                                        for j, to_speaker in enumerate(speakers):
                                            if from_speaker in transitions and to_speaker in transitions[from_speaker]:
                                                matrix[i][j] = transitions[from_speaker][to_speaker]
                                    
                                    # Create larger, more visible heatmap
                                    fig_trans = px.imshow(
                                        matrix,
                                        labels=dict(x="Spoke Next", y="Spoke First", color="Transitions"),
                                        x=speakers,
                                        y=speakers,
                                        color_continuous_scale='Blues',
                                        template='plotly_white',
                                        height=600,  # Increased height
                                        width=800     # Increased width
                                    )
                                    
                                    fig_trans.update_layout(
                                        font=dict(size=14),  # Increased font size
                                        title_font=dict(size=18, color='#2c3e50'),  # Larger title
                                        xaxis_title="Spoke Next",
                                        yaxis_title="Spoke First",
                                        title="Speaker Turn-Taking Frequency"
                                    )
                                    
                                    # Add annotations for better readability
                                    fig_trans.update_traces(
                                        text=matrix,
                                        texttemplate="%{text:.0f}",
                                        textfont={"size": 12},  # Larger text in cells
                                        hovertemplate="<b>%{y}</b> → <b>%{x}</b><br>Transitions: %{z}<extra></extra>"
                                    )
                                    
                                    st.plotly_chart(fig_trans, width="stretch", use_container_width=True)
                                else:
                                    st.info("No speaker transitions found (single speaker or no changes).")
                            else:
                                st.info("Not enough segments to calculate transitions.")
                            
                            # Display forensic metrics
                            st.subheader("📈 Forensic Metrics")
                            metrics = summary.get("forensic_metrics_summary", {})
                            
                            if metrics:
                                metrics_df = pd.DataFrame([
                                    {
                                        "Metric": metric.replace('_', ' ').title(),
                                        "Mean": data["mean"],
                                        "Std": data["std"],
                                        "Threshold": data.get("threshold", "N/A")
                                    }
                                    for metric, data in metrics.items()
                                    if metric != "detection_rates"
                                ])
                                
                                st.dataframe(metrics_df, width="stretch", hide_index=True)
                                
                                # Detection rates
                                if "detection_rates" in metrics:
                                    st.write("**Detection Rates:**")
                                    detection_rates = metrics["detection_rates"]
                                    for detection_type, rate in detection_rates.items():
                                        st.write(f"• {detection_type.replace('_', ' ').title()}: {rate:.1%}")
                            
                            # Display recommendations
                            st.subheader("💡 Recommendations")
                            recommendations = summary.get("recommendations", [])
                            
                            if recommendations:
                                for i, rec in enumerate(recommendations, 1):
                                    st.write(f"{i}. {rec}")
                            else:
                                st.info("No specific recommendations - audio appears authentic.")
                            
                            # Display forensic dossier download
                            st.subheader("📁 Forensic Dossier")
                            
                            dossier_path = Path("forensic_output/forensic_dossier.json")
                            if dossier_path.exists():
                                with open(dossier_path, 'r') as f:
                                    dossier_data = json.load(f)
                                
                                # Convert to downloadable JSON
                                dossier_json = json.dumps(dossier_data, indent=2, default=str)
                                st.download_button(
                                    label="📥 Download Forensic Dossier",
                                    data=dossier_json,
                                    file_name="forensic_dossier.json",
                                    mime="application/json"
                                )
                                
                                # Show dossier summary
                                with st.expander("📋 View Dossier Summary"):
                                    st.json(dossier_data["metadata"])
                            else:
                                st.warning("Forensic dossier file not found.")
                            
                            # Display visualizations if available
                            st.subheader("🖼️ Forensic Visualizations")
                            
                            viz_paths = [
                                ("forensic_output/forensic_clusters.png", "Speaker Identity Clustering"),
                                ("forensic_output/forensic_metrics.png", "Forensic Metrics Distribution"),
                                ("forensic_output/suspicion_analysis.png", "Suspicion Score Analysis"),
                                ("forensic_output/speaker_analysis.png", "Speaker Risk Analysis"),
                                ("forensic_output/forensic_dashboard.png", "Summary Dashboard")
                            ]
                            
                            for viz_path, title in viz_paths:
                                if Path(viz_path).exists():
                                    st.image(viz_path, caption=title, width=800)
                        
                        elif results["status"] == "no_suspicious_segments":
                            st.info("✅ No suspicious segments detected in the analysis.")
                        
                        else:
                            st.error(f"❌ Analysis failed: {results['status']}")
                    
                    except Exception as e:
                        st.error(f"❌ Error during forensic analysis: {str(e)}")
                        st.exception(e)
            
            # Display forensic analysis info
            with st.expander("ℹ️ About Forensic Analysis"):
                st.markdown("""
                **Forensic Deepfake Detection** analyzes audio segments for characteristics that may indicate synthetic or manipulated speech:
                
                **Key Features:**
                - **Spectrogram Analysis**: Detects high-frequency cutoffs common in compressed/synthetic audio
                - **Pitch Contour Tracking**: Identifies unnatural pitch patterns and jitter artifacts
                - **Bispectral Analysis**: Detects glitches and phase discontinuities
                - **Speaker Clustering**: Groups similar segments and identifies outliers
                - **Comprehensive Metrics**: ZCR, spectral flatness, jitter, shimmer, breath events
                
                **Detection Indicators:**
                - High-frequency shelf (cutoff) detection
                - Pitch jitter and step artifacts
                - Unnatural rhythm patterns
                - Abnormal spectral characteristics
                - Clustering-based outlier detection
                
                **Output:**
                - Detailed forensic dossier with all findings
                - Risk assessment and recommendations
                - Visual analysis of suspicious segments
                - Speaker-specific threat evaluation
                """)
    
    # Sidebar - Stats & Exports
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Research Analysis")
    
    experiment_file = "experiment_data.npz"
    if Path(experiment_file).exists():
        if st.sidebar.button("Generate t-SNE Plot"):
            st.sidebar.info("Computing t-SNE... Please wait.")
            
            # Use subprocess for better control
            import subprocess
            try:
                result = subprocess.run(
                    [".venv311/bin/python", "visualize_paper.py", "--input", experiment_file, "--output", "speaker_tsne_plot.png"],
                    capture_output=True,
                    text=True,
                    cwd="."
                )
                
                if result.returncode == 0 and Path("speaker_tsne_plot.png").exists():
                    st.sidebar.success("t-SNE plot generated successfully!")
                    st.session_state.show_tsne = True
                else:
                    st.sidebar.error(f"Failed to generate t-SNE: {result.stderr}")
                    st.session_state.show_tsne = False
            except Exception as e:
                st.sidebar.error(f"Error running t-SNE generation: {str(e)}")
                st.session_state.show_tsne = False
    else:
        st.sidebar.caption("Run analysis with `--diarization` to generate research data.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Report")
    if st.sidebar.button("Download CSV"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Confirm Download",
            data=csv,
            file_name="sentiment_report.csv",
            mime="text/csv",
        )

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
