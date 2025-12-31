# Speech Sentiment Analysis

This project aims to analyze and classify emotions from speech signals using various deep learning models. The datasets used in this project include Crema, Ravdess, Savee, and Tess. Data augmentation techniques such as noise introduction, pitch shifting, and stretching were applied to enhance the dataset. Features such as MFCCs, Energy and Entropy of Energy, Zero Crossing Rate, Mel-Spectrogram, and Spectral Features were extracted for analysis.

## Introduction

Speech sentiment analysis is the task of classifying emotions in spoken language. It has applications in various fields such as customer service, mental health monitoring, and human-computer interaction. This project leverages multiple datasets and a combination of data augmentation and feature extraction techniques to build and evaluate four machine learning models: ANN, CNN, LSTM, and CNN-LSTM.

## Datasets

The following datasets were used in this project:

- **Crema**: Crowd-sourced Emotional Multimodal Actors dataset.
- **Ravdess**: Ryerson Audio-Visual Database of Emotional Speech and Song.
- **Savee**: Surrey Audio-Visual Expressed Emotion dataset.
- **Tess**: Toronto Emotional Speech Set.

## Data Augmentation

To enhance the dataset and improve model robustness, the following data augmentation techniques were applied:

- **Noise Introduction**: Adding background noise to the audio samples.
- **Pitch Shifting**: Altering the pitch of the audio samples.
- **Stretching**: Changing the speed of the audio samples without altering the pitch.

## Feature Extraction

The following features were extracted from the audio data to capture various aspects of the speech signals:

- **MFCCs (Mel-Frequency Cepstral Coefficients)**: Capture detailed spectral information and are widely used in speech and audio processing.
- **Energy and Entropy of Energy**: Provide information about the intensity and variability of speech, which are important for detecting emotions.
- **Zero Crossing Rate**: Useful for distinguishing between different types of speech sounds.
- **Mel-Spectrogram**: Represents the power spectrum in the mel scale. Captures both temporal and spectral features of the audio signal.
- **Spectral Features (Centroid, Spread, Roll-off)**: Provide a comprehensive description of the spectral characteristics of the speech signal.

## Models

Four machine learning models were built and evaluated for this project:

1. **ANN (Artificial Neural Network)**
2. **CNN (Convolutional Neural Network)**
3. **LSTM (Long Short-Term Memory)**
4. **CNN-LSTM (Convolutional Neural Network - Long Short-Term Memory)**

## Results

The models were evaluated based on their accuracy and other relevant metrics. The CNN model showed the highest performance, achieving a validation accuracy of 70%.

![CNN_Loss curve](https://github.com/user-attachments/assets/9fed9d71-2391-4c29-b882-f4e4e599f5e6)

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the dependencies using the following command:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
 
pip install -r requirements.txt
```

## Usage
To use the models for speech sentiment analysis, follow these steps:

1. Clone this repository:
```bash
git clone https://github.com/DevG06/Speech-Sentiment-Analysis.git
```
2. Navigate to project directory:
```bash
cd Speech-Sentiment-Analysis
```

3. Run the Jupyter notebooks provided to preprocess the data, extract features, and train the models.

## Advanced Workflows

### 1. Real-time Monitoring with Calibration & Enhancement
To run the real-time analyzer with auto-bias correction and noise suppression:
```bash
.venv311/bin/python realtime_analysis.py \
  --mode mic \
  --asr thai \
  --diarization \
  --speaker-threshold 0.6 \
  --mic-source-separation \
  --auto-bias \
  --dynamic-calibration \
  --log-csv analysis_log.csv
```
*   **`--mic-source-separation`**: Uses DeepFilterNet to remove background noise and music in real-time.
*   **Speaker Hints**: The console will show `[Match: High/Medium/Low]`. A `Medium` match often indicates a voice clone or a significant tone change.

### 2. Speaker Diarization & Source Separation (Offline)
For maximum accuracy on recordings with music:

**Step A: Clean Audio (Optional but Recommended)**
If the file has loud music, run diarization with source separation:
```bash
.venv311/bin/python realtime_analysis.py \
  --mode watch \
  --source-separation \
  --input-dir path/to/recordings/
```
*   **`--source-separation`**: Uses Facebook Demucs (htdemucs) to isolate vocals before analysis.

**Step B: Run Deep Diarization**
Detects speaker turns using TitaNet-L and Multi-Scale Spectral Clustering.
```bash
.venv311/bin/python diarization_analysis.py path/to/audio.wav --out-dir results/
```

**Step B: Analyze Sentiment per Speaker**
Processes the diarization output (RTTM) to track emotions for each individual speaker.
```bash
.venv311/bin/python diarized_sentiment.py \
  --audio path/to/audio.wav \
  --rttm results/input_manifest.rttm
```
.venv311/bin/python realtime_analysis.py \
  --mode mic \
  --asr thai \
  --diarization \
  --speaker-threshold 0.49 \
  --auto-bias \
  --dynamic-calibration \
  --source-separation
### 3. Advanced Visualization Dashboard (Advanced Reporting)
To view an interactive, real-time "Intelligence Dashboard" of your analysis:

**Step A: Start the Analyzer with Rich Logging**
Ensure you specify a `--log-csv` path. The dashboard reads this file live.
```bash
.venv311/bin/python realtime_analysis.py \
  --mode mic \
  --asr thai \
  --diarization \
 --speaker-threshold 0.49 \
  --log-csv analysis_log.csv \
  --auto-bias \
  --dynamic-calibration
```

**Step B: Launch the Interactive Dashboard**
In a separate terminal window:
```bash
.venv311/bin/streamlit run visualize_report.py
```

**Dashboard Features:**
*   **🕒 Timeline Analysis**: Track emotional shifts per speaker over time.
*   **📊 Emotion Breakdown**: View total sentiment share and distribution.
*   **👥 Speaker Analytics**: Analyze talk-time, emotional participation, and "Mood Volatility".
*   **🕸️ Radar Comparison**: Compare the unique "Emotional Fingerprint" of different speakers.
*   **🔤 Keyword Analysis**: Integrated Thai/English WordCloud to see what people are talking about.
*   **📈 Correlation**: Analyze how vocal intensity (RMS) correlates with confidence and specific emotions.
*   **🔬 Research Mode**: Generate and view publication-ready **t-SNE plots** directly from the sidebar.

### 4. 🔬 Forensic Deepfake Detection System
Advanced forensic analysis for detecting synthetic voice characteristics and AI-generated audio:

**Step A: Run Forensic Analysis**
Analyze existing sentiment data for deepfake indicators:
```bash
.venv311/bin/python forensic_pipeline.py --csv analysis_log.csv
```

**Step B: With Audio File (Enhanced Analysis)**
For detailed signal-level analysis with spectrograms and pitch tracking:
```bash
.venv311/bin/python forensic_pipeline.py --csv analysis_log.csv --audio audio_file.wav
```

**Step C: Interactive Forensic Dashboard**
Access forensic analysis through the Streamlit interface:
```bash
.venv311/bin/streamlit run visualize_report.py
# Navigate to "🔬 Forensic Analysis" tab
```

**Forensic Features:**
*   **🎵 Spectrogram Analysis**: High-frequency cutoff detection (16kHz threshold)
*   **📈 Pitch Contour Tracking**: Jitter and step artifact identification
*   **🔍 Bispectral Analysis**: Glitch density visualization for artifact isolation
*   **📊 Advanced Metrics**: ZCR, spectral flatness, jitter/shimmer, breath events
*   **🎯 t-SNE Clustering**: Real vs suspected clone color coding with outlier detection
*   **📋 Forensic Dossier**: Comprehensive JSON reports replacing CSV exports
*   **⚠️ Risk Assessment**: HIGH/MEDIUM/LOW/MINIMAL threat classifications
*   **🔧 Automated Evidence Generation**: Suspicious segment compilation and recommendations

**Detection Indicators:**
- **Synthetic Voice Signs**: Unnatural pitch patterns, spectral anomalies
- **Voice Clone Detection**: Speaker embedding outliers and identity inconsistencies  
- **Artifact Isolation**: Bispectral glitch detection and high-frequency shelf analysis
- **Breathing Patterns**: Abnormal breath event counts and pause ratios
- **Rhythm Analysis**: Unnatural speech timing and cadence detection

**Output Files:**
- `forensic_dossier.json` - Complete forensic analysis report
- `forensic_clusters.png` - t-SNE speaker clustering visualization
- `forensic_metrics.png` - Forensic metrics distribution analysis
- `suspicion_analysis.png` - Suspicion score breakdown
- `speaker_analysis.png` - Speaker-specific risk assessment

## License
This project is licensed under the [MIT License](LICENSE).