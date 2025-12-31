# Forensic Deepfake Detection System

## Overview

This system implements a comprehensive forensic analysis framework for detecting synthetic voice characteristics and potential deepfake audio segments. It integrates with the existing speech sentiment analysis pipeline to provide advanced signal-level analysis, clustering-based outlier detection, and automated evidence generation.

## Key Features

### 🔬 Signal-Level Analysis
- **Spectrogram Analysis**: Detects high-frequency cutoffs (16kHz shelf) common in compressed/synthetic audio
- **Pitch (F0) Contour Tracking**: Identifies jitter, step artifacts, and unnatural pitch patterns
- **Bispectral Analysis**: Detects glitches and phase discontinuities indicative of synthetic speech
- **Comprehensive Metrics**: Zero-crossing rate, spectral flatness, jitter, shimmer, breath events

### 🎯 Advanced Detection
- **Forensic Metrics**: ZCR, spectral flatness, jitter/shimmer, breath event count, pause ratio
- **Automated Thresholds**: Configurable detection thresholds for each metric
- **Suspicion Scoring**: Composite scoring system combining multiple indicators
- **Speaker-Specific Analysis**: Individual speaker risk assessment

### 📊 Clustering & Visualization
- **t-SNE Embedding**: Dimensionality reduction for speaker identity visualization
- **DBSCAN Clustering**: Outlier detection and speaker grouping
- **Enhanced Visualizations**: Spectrograms, pitch contours, glitch density charts
- **Real vs Clone Color Coding**: Visual distinction between authentic and suspicious segments

### 📁 Evidence Generation
- **Forensic Dossier**: Comprehensive JSON report with all findings
- **Risk Assessment**: Overall threat level evaluation
- **Actionable Recommendations**: Specific guidance based on analysis results
- **Audio Evidence**: Extraction and compilation of suspicious segments

## Architecture

### Core Components

1. **ForensicConfig** - Configuration management with audio and threshold parameters
2. **ForensicSegment** - Data structure for audio segments with forensic metrics
3. **AudioForensicsEngine** - Signal analysis engine for forensic metrics
4. **IdentityClusterer** - Speaker embedding clustering and outlier detection
5. **EvidenceGenerator** - Forensic dossier compilation and export
6. **ForensicPipeline** - Main orchestration pipeline
7. **ForensicVisualizer** - Advanced visualization components

### Detection Methods

#### High-Frequency Shelf Detection
- Analyzes spectral energy distribution above 16kHz threshold
- Identifies compression artifacts common in synthetic audio
- Provides energy ratio metrics for quantitative assessment

#### Pitch Artifact Detection
- Calculates jitter (pitch variation) and shimmer (amplitude variation)
- Detects step-like changes in pitch contours
- Identifies monotonic segments unnatural for speech

#### Glitch Density Analysis
- Bispectral analysis for phase discontinuities
- Temporal glitch density tracking
- Instantaneous phase analysis for artifact detection

#### Clustering-Based Outlier Detection
- t-SNE dimensionality reduction for feature visualization
- DBSCAN clustering for speaker group identification
- Outlier detection based on cluster membership

## Installation & Setup

### Dependencies

```bash
# Core audio processing
pip install librosa numpy scipy pandas

# Machine learning and clustering
pip install scikit-learn torch

# Visualization
pip install matplotlib seaborn plotly

# Streamlit dashboard
pip install streamlit

# Optional: Thai language support
pip install pythainlp wordcloud
```

### File Structure

```
Speech-Sentiment-Analysis/
├── forensic_deepfake_detection.py    # Core forensic analysis engine
├── forensic_pipeline.py              # Main analysis pipeline
├── forensic_visualizations.py        # Advanced visualization components
├── visualize_report.py               # Updated Streamlit dashboard
├── analysis_log.csv                  # Input CSV with sentiment analysis
├── forensic_output/                  # Generated analysis results
│   ├── forensic_dossier.json        # Comprehensive forensic report
│   ├── forensic_clusters.png         # Speaker clustering visualization
│   ├── forensic_metrics.png          # Metrics distribution plots
│   ├── suspicion_analysis.png        # Suspicion score analysis
│   ├── speaker_analysis.png          # Speaker-specific analysis
│   └── forensic_dashboard.png       # Summary dashboard
└── README_FORENSIC.md               # This documentation
```

## Usage

### Command Line Interface

```bash
# Basic forensic analysis
python forensic_pipeline.py --csv analysis_log.csv

# With audio file for detailed analysis
python forensic_pipeline.py --csv analysis_log.csv --audio audio_file.wav

# Custom configuration
python forensic_pipeline.py --csv analysis_log.csv --audio audio_file.wav \
    --config forensic_config.json --output custom_output

# Adjust suspicion threshold
python forensic_pipeline.py --csv analysis_log.csv --suspicion-threshold 0.4

# Disable plot generation
python forensic_pipeline.py --csv analysis_log.csv --no-plots
```

### Streamlit Dashboard Integration

```bash
# Launch the enhanced dashboard
streamlit run visualize_report.py
```

The dashboard now includes a **"🔬 Forensic Analysis"** tab with:
- Real-time forensic analysis controls
- Interactive suspicion threshold adjustment
- Comprehensive results visualization
- Forensic dossier download
- Detailed metric analysis

### Python API

```python
from forensic_pipeline import ForensicPipeline, ForensicConfig
from forensic_deepfake_detection import AudioForensicsEngine

# Initialize with custom configuration
config = ForensicConfig(
    high_freq_cutoff=16000.0,
    jitter_threshold=0.005,
    suspicion_threshold=0.3
)

# Run forensic analysis
pipeline = ForensicPipeline(config)
results = pipeline.analyze_csv_data("analysis_log.csv", "audio_file.wav")

# Access results
if results["status"] == "completed":
    print(f"Suspicious segments: {results['suspicious_segments']}")
    print(f"Risk level: {results['summary']['risk_assessment']['risk_level']}")
    
    # Download forensic dossier
    dossier = results["forensic_dossier"]
```

## Configuration

### Default Parameters

```python
@dataclass
class ForensicConfig:
    # Audio processing
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    
    # Detection thresholds
    high_freq_cutoff: float = 16000.0  # Hz
    jitter_threshold: float = 0.005
    shimmer_threshold: float = 0.05
    zcr_threshold: float = 0.1
    spectral_flatness_threshold: float = 0.3
    min_breath_events_per_minute: float = 2.0
    
    # Clustering parameters
    tsne_perplexity: int = 30
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    
    # Analysis parameters
    segment_duration: float = 2.0  # seconds
    overlap_ratio: float = 0.5
    
    # Output settings
    output_dir: str = "forensic_output"
    create_plots: bool = True
```

### Custom Configuration

Create a JSON configuration file:

```json
{
    "sample_rate": 44100,
    "high_freq_cutoff": 16000.0,
    "jitter_threshold": 0.008,
    "shimmer_threshold": 0.07,
    "suspicion_threshold": 0.4,
    "dbscan_eps": 0.6,
    "create_plots": true
}
```

## Output & Results

### Forensic Dossier Structure

```json
{
    "metadata": {
        "total_segments": 150,
        "suspicious_segments": 23,
        "suspicion_rate": 0.153,
        "analysis_timestamp": "2025-12-31T16:00:00",
        "config": {...}
    },
    "forensic_metrics": {
        "zcr": {"mean": 0.082, "std": 0.034, "threshold": 0.1},
        "spectral_flatness": {"mean": 0.245, "std": 0.089, "threshold": 0.3},
        "jitter": {"mean": 0.0042, "std": 0.0018, "threshold": 0.005},
        "detection_rates": {
            "high_freq_shelf": 0.187,
            "pitch_artifacts": 0.233,
            "unnatural_rhythm": 0.127
        }
    },
    "suspicious_segments": [...],
    "cluster_analysis": {...},
    "recommendations": [
        "HIGH RISK: More than 15% of segments show suspicious characteristics",
        "High-frequency cutoff detected in multiple segments",
        "Recommendation: Immediate manual review required"
    ],
    "risk_assessment": {
        "risk_level": "MEDIUM",
        "risk_description": "Significant portion of segments show suspicious patterns",
        "suspicion_rate": 0.153,
        "confidence": "medium"
    }
}
```

### Risk Levels

- **HIGH**: >50% suspicious segments - Immediate review required
- **MEDIUM**: 20-50% suspicious segments - Enhanced verification needed
- **LOW**: 5-20% suspicious segments - Monitor closely
- **MINIMAL**: <5% suspicious segments - Low risk

## Integration with Existing System

### CSV Analysis Integration

The forensic system automatically analyzes existing CSV data from the sentiment analysis pipeline:

- **Speaker Match Indicators**: Detects "[Match: ...]" patterns in speaker labels
- **Emotional Volatility**: High entropy segments flagged as suspicious
- **Confidence Anomalies**: Unusually high or low confidence scores
- **Volume Spikes**: High RMS energy segments
- **Probability Distribution**: Overly confident emotion predictions

### Audio Processing Integration

- **Segment Extraction**: Automatic extraction of suspicious audio segments
- **Real-time Analysis**: Compatible with live audio processing
- **Batch Processing**: Efficient analysis of large audio datasets
- **Format Support**: WAV, MP3, FLAC and other common audio formats

## Advanced Features

### Bispectral Analysis

The system implements simplified bispectral analysis for glitch detection:

```python
# Glitch density computation
glitch_density = compute_glitch_density_indicator(audio, sample_rate)

# Phase discontinuity detection
phase_jumps = detect_phase_discontinuities(audio)

# Bispectral coherence (simplified)
bispectrum_mag = compute_bispectrum_magnitude(audio)
```

### Enhanced t-SNE Clustering

```python
# Feature extraction
embeddings = extract_forensic_embeddings(segments)

# Dimensionality reduction
tsne_embeddings = TSNE(n_components=2, perplexity=30).fit_transform(embeddings)

# Clustering and outlier detection
cluster_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(embeddings)
```

### Real-time Visualization

The Streamlit dashboard provides:
- Live forensic analysis results
- Interactive metric exploration
- Dynamic threshold adjustment
- Real-time risk assessment updates

## Performance & Optimization

### Computational Requirements

- **CPU**: Multi-core processor recommended for batch processing
- **Memory**: 4GB+ RAM for large audio datasets
- **Storage**: Additional space for forensic output files
- **GPU**: Optional, can accelerate deep learning components

### Optimization Tips

1. **Segment Duration**: Adjust `segment_duration` based on analysis requirements
2. **Batch Processing**: Process multiple segments in parallel for efficiency
3. **Threshold Tuning**: Calibrate thresholds for specific audio characteristics
4. **Output Management**: Regular cleanup of forensic output files

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Audio Loading**: Check audio file paths and formats
3. **Memory Issues**: Reduce segment duration or batch size
4. **Visualization Errors**: Disable plots if display issues occur

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
pipeline = ForensicPipeline(config)
results = pipeline.analyze_csv_data(csv_path, audio_path)
```

## Future Enhancements

### Planned Features

- **Deep Learning Models**: Integration with advanced deepfake detection models
- **Real-time Processing**: Live audio stream analysis capabilities
- **Multi-language Support**: Extended language-specific analysis
- **Cloud Integration**: Distributed processing for large-scale analysis
- **API Endpoints**: RESTful API for integration with other systems

### Research Directions

- **Advanced Bispectral Analysis**: Full bispectral coherence implementation
- **Neural Voice Detection**: Deep learning-based synthetic voice detection
- **Cross-modal Analysis**: Integration with video deepfake detection
- **Adaptive Thresholds**: Machine learning-based threshold optimization

## License & Citation

This forensic analysis system is part of the Speech Sentiment Analysis project. Please cite appropriately if used in research or commercial applications.

## Support

For technical support, feature requests, or bug reports, please refer to the project documentation or create an issue in the project repository.

---

**Note**: This forensic analysis system is designed for research and educational purposes. Results should be validated by human experts for critical applications.
