import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Load the data
df = pd.read_csv('analysis_log.csv')

# Get probability columns
prob_cols = [c for c in df.columns if c.startswith('prob_')]
if prob_cols:
    # Calculate average emotion probabilities per speaker
    radar_data = df.groupby('speaker')[prob_cols].mean().reset_index()
    radar_data.columns = ['speaker'] + [c.replace('prob_', '') for c in prob_cols]
    
    # Select top speakers for clarity (limit to 10 speakers)
    top_speakers = radar_data.head(10)
    
    # Create radar chart
    fig_radar = go.Figure()
    
    # Add traces for each speaker
    colors = px.colors.qualitative.Set1[:10]
    for i, (_, row) in enumerate(top_speakers.iterrows()):
        fig_radar.add_trace(go.Scatterpolar(
            r=row[1:].values,
            theta=radar_data.columns[1:],
            fill='toself',
            name=row['speaker'],
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)]
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            )
        ),
        title="Speaker Emotion Radar Comparison - Thai Speech Analysis",
        title_font=dict(size=16, family="Arial, sans-serif"),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        font=dict(family="Arial, sans-serif", size=12),
        width=800,
        height=800,
        template='plotly_white'
    )
    
    # Save the figure
    fig_radar.write_image("emotion_radar_comparison.png", scale=2, width=800, height=800)
    print("Emotion radar plot saved as 'emotion_radar_comparison.png'")
    
    # Also save as HTML for interactive viewing
    fig_radar.write_html("emotion_radar_comparison.html")
    print("Interactive radar plot saved as 'emotion_radar_comparison.html'")
    
else:
    print("No probability columns found in the data")
