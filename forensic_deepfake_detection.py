#!/usr/bin/env python3
"""
Forensic Deepfake Detection System
===================================

This module implements a comprehensive forensic analysis system for detecting
synthetic voice characteristics in audio segments. It provides signal-level
analysis, clustering-based outlier detection, and automated evidence generation.

Key Features:
- Spectrogram analysis with high-frequency cutoff detection
- Pitch (F0) contour tracking for jitter and step artifacts
- Forensic metrics: ZCR, spectral flatness, jitter/shimmer, breath events
- Bispectral analysis for glitch density detection
- Enhanced t-SNE embedding clustering with real vs clone identification
- Comprehensive forensic dossier generation

Dependencies:
- librosa: Audio signal processing
- numpy: Numerical computations
- scipy: Signal processing and statistical analysis
- scikit-learn: Machine learning and clustering
- matplotlib/seaborn: Visualization
- torch: Deep learning operations (if available)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import librosa
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import entropy
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Import Audio X-Ray RPCA module
try:
    from audio_xray_rpca import AudioXRay, AudioXRayConfig
    AUDIO_XRAY_AVAILABLE = True
    XRAY_IMPORT_ERROR = None
except ImportError as e:
    AUDIO_XRAY_AVAILABLE = False
    XRAY_IMPORT_ERROR = str(e)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log Audio X-Ray availability after logger is defined
if AUDIO_XRAY_AVAILABLE:
    logger.info("Audio X-Ray RPCA module loaded successfully")
else:
    logger.warning(f"Audio X-Ray RPCA module not available: {XRAY_IMPORT_ERROR}")


@dataclass
class ForensicConfig:
    """Configuration parameters for forensic analysis."""
    
    # Audio processing parameters
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 22050.0
    
    # Forensic thresholds
    high_freq_cutoff: float = 16000.0  # Hz - threshold for high-frequency shelf detection
    jitter_threshold: float = 0.005  # Jitter threshold for synthetic voice detection
    shimmer_threshold: float = 0.05  # Shimmer threshold for synthetic voice detection
    zcr_threshold: float = 0.1  # Zero-crossing rate threshold
    spectral_flatness_threshold: float = 0.3  # Spectral flatness threshold
    min_breath_events_per_minute: float = 2.0  # Minimum breath events for natural speech
    
    # Clustering parameters
    tsne_perplexity: int = 30
    tsne_n_iter: int = 1000
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    
    # Analysis parameters
    segment_duration: float = 2.0  # Duration of audio segments for analysis (seconds)
    overlap_ratio: float = 0.5  # Overlap ratio between segments
    
    # Output parameters
    output_dir: str = "forensic_output"
    create_plots: bool = True
    save_audio_segments: bool = True


@dataclass
class ForensicSegment:
    """Data structure for forensic analysis of audio segments."""
    
    # Basic segment information
    segment_id: str
    timestamp: str
    speaker: str
    duration: float
    
    # Audio analysis results
    spectrogram_cutoff_freq: Optional[float] = None
    jitter: Optional[float] = None
    shimmer: Optional[float] = None
    zcr: Optional[float] = None
    spectral_flatness: Optional[float] = None
    breath_events: Optional[int] = None
    pause_ratio: Optional[float] = None
    high_freq_shelf: Optional[bool] = None
    pitch_artifacts: Optional[bool] = None
    unnatural_rhythm: Optional[bool] = None
    
    # RPCA Audio X-Ray results
    rpca_deepfake_likelihood: Optional[float] = None
    rpca_rhythmicity_score: Optional[float] = None
    rpca_phase_blast_score: Optional[float] = None
    rpca_silence_ratio: Optional[float] = None
    rpca_artifact_density: Optional[float] = None
    rpca_periodicity_score: Optional[float] = None
    rpca_indicators: Optional[List[str]] = None
    rpca_artifact_audio_path: Optional[str] = None
    rpca_visualizations: Optional[Dict[str, str]] = None
    
    # Analysis metadata
    suspicion_score: float = 0.0
    risk_level: str = "MINIMAL"
    flags: List[str] = None
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.flags is None:
            self.flags = []
        if self.rpca_indicators is None:
            self.rpca_indicators = []
        if self.rpca_visualizations is None:
            self.rpca_visualizations = {}
    
    # Embedding and clustering
    embedding: Optional[np.ndarray] = None
    cluster_label: Optional[int] = None
    is_outlier: bool = False
    
    # Audio data (optional)
    audio_data: Optional[np.ndarray] = None


class AudioForensicsEngine:
    """Engine for signal-level audio forensic analysis."""
    
    def __init__(self, config: ForensicConfig):
        self.config = config
        logger.info("Initialized AudioForensicsEngine")
    
    def analyze_segment(self, audio: np.ndarray, sr: int, 
                       start_time: float, speaker_label: str) -> ForensicSegment:
        """Perform comprehensive forensic analysis on an audio segment."""
        
        segment_id = f"{speaker_label}_{start_time:.2f}"
        duration = len(audio) / sr
        
        # Basic audio features
        rms_energy = np.sqrt(np.mean(audio**2))
        zcr = librosa.feature.zero_crossing_rate(audio)[0].mean()
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0].mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0].mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0].mean()
        
        # Pitch analysis
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, sr=sr, fmin=50, fmax=400, frame_length=2048, hop_length=512
        )
        
        # Remove NaN values from pitch
        valid_f0 = f0[~np.isnan(f0)]
        f0_mean = np.mean(valid_f0) if len(valid_f0) > 0 else 0.0
        f0_std = np.std(valid_f0) if len(valid_f0) > 0 else 0.0
        
        # Calculate jitter and shimmer
        jitter = self._calculate_jitter(valid_f0)
        shimmer = self._calculate_shimmer(audio, sr)
        
        # Breath detection
        breath_events = self._detect_breath_events(audio, sr)
        pause_ratio = self._calculate_pause_ratio(audio, sr)
        
        # High-frequency shelf detection
        high_freq_shelf = self._detect_high_freq_shelf(audio, sr)
        
        # Pitch artifact detection
        pitch_artifacts = self._detect_pitch_artifacts(valid_f0)
        
        # Unnatural rhythm detection
        unnatural_rhythm = self._detect_unnatural_rhythm(audio, sr)
        
        # Calculate suspicion score
        suspicion_score = self._calculate_suspicion_score(
            zcr, spectral_flatness, jitter, shimmer, 
            breath_events, pause_ratio, high_freq_shelf, pitch_artifacts
        )
        
        is_suspicious = suspicion_score > 0.5
        
        segment = ForensicSegment(
            segment_id=segment_id,
            timestamp=str(start_time),
            speaker=speaker_label,
            duration=duration,
            zcr=zcr,
            spectral_flatness=spectral_flatness,
            jitter=jitter,
            shimmer=shimmer,
            breath_events=breath_events,
            pause_ratio=pause_ratio,
            high_freq_shelf=high_freq_shelf,
            pitch_artifacts=pitch_artifacts,
            unnatural_rhythm=unnatural_rhythm,
            suspicion_score=suspicion_score,
            audio_data=audio
        )
        
        # Apply Audio X-Ray RPCA analysis if available
        segment = self.apply_audio_xray(segment, audio, sr)
        
        # Recalculate suspicion score with RPCA results
        if segment.rpca_deepfake_likelihood is not None:
            segment.suspicion_score = self._calculate_suspicion_score(
                zcr, spectral_flatness, jitter, shimmer, 
                breath_events, pause_ratio, high_freq_shelf, pitch_artifacts,
                segment.rpca_deepfake_likelihood
            )
        
        # Update risk level based on final suspicion score
        if segment.suspicion_score > 0.7:
            segment.risk_level = "HIGH"
        elif segment.suspicion_score > 0.4:
            segment.risk_level = "MEDIUM"
        elif segment.suspicion_score > 0.2:
            segment.risk_level = "LOW"
        else:
            segment.risk_level = "MINIMAL"
        
        # Add flags based on analysis
        if high_freq_shelf:
            segment.flags.append("High-frequency shelf detected")
        if pitch_artifacts:
            segment.flags.append("Pitch artifacts detected")
        if unnatural_rhythm:
            segment.flags.append("Unnatural rhythm detected")
        if segment.rpca_deepfake_likelihood and segment.rpca_deepfake_likelihood > 0.5:
            segment.flags.append("RPCA indicates synthetic voice")
        
        logger.info(f"Completed forensic analysis for segment {segment_id}: suspicion_score={segment.suspicion_score:.3f}")
        
        return segment
    
    def _calculate_jitter(self, f0_values: np.ndarray) -> float:
        """Calculate pitch jitter (variation in fundamental frequency)."""
        if len(f0_values) < 2:
            return 0.0
        
        # Calculate relative jitter
        f0_diff = np.diff(f0_values)
        jitter = np.std(f0_diff) / np.mean(f0_values) if np.mean(f0_values) > 0 else 0.0
        return jitter
    
    def _calculate_shimmer(self, audio: np.ndarray, sr: int) -> float:
        """Calculate amplitude shimmer (variation in amplitude)."""
        # Calculate RMS energy in short frames
        frame_length = int(0.01 * sr)  # 10ms frames
        hop_length = frame_length // 2
        
        rms_frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            rms_frames.append(np.sqrt(np.mean(frame**2)))
        
        rms_frames = np.array(rms_frames)
        
        if len(rms_frames) < 2:
            return 0.0
        
        # Calculate relative shimmer
        rms_diff = np.diff(rms_frames)
        shimmer = np.std(rms_diff) / np.mean(rms_frames) if np.mean(rms_frames) > 0 else 0.0
        return shimmer
    
    def _detect_breath_events(self, audio: np.ndarray, sr: int) -> int:
        """Detect breath events in audio segment."""
        # Simple breath detection based on energy and spectral characteristics
        frame_length = int(0.02 * sr)  # 20ms frames
        hop_length = frame_length // 2
        
        breath_count = 0
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            
            # Calculate spectral characteristics
            fft = np.fft.fft(frame)
            freqs = np.fft.fftfreq(len(frame), 1/sr)
            
            # Breath typically has energy in low frequencies and high spectral flatness
            low_freq_energy = np.sum(np.abs(fft[:len(fft)//4]))
            total_energy = np.sum(np.abs(fft))
            
            if total_energy > 0:
                low_freq_ratio = low_freq_energy / total_energy
                flatness = librosa.feature.spectral_flatness(y=frame)[0].mean()
                
                # Simple heuristic for breath detection
                if low_freq_ratio > 0.7 and flatness > 0.4:
                    breath_count += 1
        
        return breath_count
    
    def _calculate_pause_ratio(self, audio: np.ndarray, sr: int) -> float:
        """Calculate the ratio of silence/pauses in the audio segment."""
        # Define silence threshold (very low energy)
        silence_threshold = 0.01 * np.max(np.abs(audio))
        
        frame_length = int(0.01 * sr)  # 10ms frames
        hop_length = frame_length // 2
        
        silence_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            frame_energy = np.sqrt(np.mean(frame**2))
            
            if frame_energy < silence_threshold:
                silence_frames += 1
            total_frames += 1
        
        return silence_frames / total_frames if total_frames > 0 else 0.0
    
    def _detect_high_freq_shelf(self, audio: np.ndarray, sr: int) -> bool:
        """Detect if there's a high-frequency shelf (cutoff) in the spectrum."""
        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=self.config.n_fft, hop_length=self.config.hop_length)
        magnitude = np.abs(stft)
        
        # Average magnitude spectrum
        avg_spectrum = np.mean(magnitude, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.config.n_fft)
        
        # Check energy ratio above and below cutoff frequency
        cutoff_idx = np.argmin(np.abs(freqs - self.config.high_freq_cutoff))
        
        low_freq_energy = np.sum(avg_spectrum[:cutoff_idx])
        high_freq_energy = np.sum(avg_spectrum[cutoff_idx:])
        
        if low_freq_energy > 0:
            high_freq_ratio = high_freq_energy / low_freq_energy
            # High-frequency shelf detected if ratio is very low
            return high_freq_ratio < 0.1
        
        return False
    
    def _detect_pitch_artifacts(self, f0_values: np.ndarray) -> bool:
        """Detect pitch artifacts indicative of synthetic speech."""
        if len(f0_values) < 10:
            return False
        
        # Check for unusual pitch patterns
        jitter = self._calculate_jitter(f0_values)
        
        # Check for step-like changes (non-smooth transitions)
        f0_diff = np.diff(f0_values)
        large_changes = np.sum(np.abs(f0_diff) > 2 * np.std(f0_diff))
        
        # Check for monotonic segments (unnatural for speech)
        monotonic_segments = 0
        for i in range(len(f0_values) - 5):
            segment = f0_values[i:i+5]
            if (np.all(np.diff(segment) >= 0) or np.all(np.diff(segment) <= 0)):
                monotonic_segments += 1
        
        return (jitter > self.config.jitter_threshold or 
                large_changes > len(f0_diff) * 0.1 or
                monotonic_segments > len(f0_values) * 0.2)
    
    def _detect_unnatural_rhythm(self, audio: np.ndarray, sr: int) -> bool:
        """Detect unnatural rhythm patterns in speech."""
        # Calculate onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=512)
        
        if len(onset_frames) < 2:
            return False
        
        # Calculate inter-onset intervals
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        intervals = np.diff(onset_times)
        
        # Check for overly regular intervals (unnatural for natural speech)
        if len(intervals) > 0:
            interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
            # Low coefficient of variation suggests overly regular rhythm
            return interval_cv < 0.3
        
        return False
    
    def _calculate_suspicion_score(self, zcr: float, spectral_flatness: float, 
                                jitter: float, shimmer: float, breath_events: int,
                                pause_ratio: float, high_freq_shelf: bool, 
                                pitch_artifacts: bool, rpca_likelihood: Optional[float] = None) -> float:
        """Calculate overall suspicion score based on all forensic metrics."""
        
        score = 0.0
        
        # Zero-crossing rate contribution
        if zcr > self.config.zcr_threshold:
            score += 0.15
        
        # Spectral flatness contribution
        if spectral_flatness > self.config.spectral_flatness_threshold:
            score += 0.15
        
        # Jitter contribution
        if jitter > self.config.jitter_threshold:
            score += 0.2
        
        # Shimmer contribution
        if shimmer > self.config.shimmer_threshold:
            score += 0.2
        
        # Breath events contribution (too few breaths is suspicious)
        expected_breaths = self.config.min_breath_events_per_minute * 2.0 / 60.0  # 2 second segment
        if breath_events < max(1, expected_breaths * 0.5):
            score += 0.1
        
        # Pause ratio contribution
        if pause_ratio > 0.3:  # Too much silence
            score += 0.1
        
        # High-frequency shelf contribution
        if high_freq_shelf:
            score += 0.25
        
        # Pitch artifacts contribution
        if pitch_artifacts:
            score += 0.25
        
        # RPCA Audio X-Ray contribution (if available)
        if rpca_likelihood is not None and AUDIO_XRAY_AVAILABLE:
            score += rpca_likelihood * 0.3  # Weight RPCA results significantly
        
        return min(score, 1.0)  # Cap at 1.0
    
    def apply_audio_xray(self, segment: ForensicSegment, audio: np.ndarray, sr: int) -> ForensicSegment:
        """Apply Audio X-Ray RPCA analysis to segment if available."""
        if not AUDIO_XRAY_AVAILABLE:
            logger.warning("Audio X-Ray RPCA not available, skipping analysis")
            return segment
        
        try:
            # Create temporary audio file for RPCA analysis
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                import soundfile as sf
                sf.write(tmp_file.name, audio, sr)
                
                # Configure Audio X-Ray with minimal visualizations for speed
                xray_config = AudioXRayConfig(
                    output_dir=self.config.output_dir,
                    save_visualizations=False,  # Disable visualizations for speed
                    save_audio=True
                )
                
                # Run Audio X-Ray analysis
                xray = AudioXRay(xray_config)
                results = xray.run_audio_xray(tmp_file.name)
                
                # Update segment with RPCA results
                segment.rpca_deepfake_likelihood = results['artifact_analysis']['deepfake_likelihood']
                segment.rpca_rhythmicity_score = results['artifact_analysis']['rhythmicity_score']
                segment.rpca_phase_blast_score = results['artifact_analysis']['phase_blast_score']
                segment.rpca_silence_ratio = results['artifact_analysis']['silence_ratio']
                segment.rpca_artifact_density = results['artifact_analysis']['artifact_density']
                segment.rpca_periodicity_score = results['artifact_analysis']['periodicity_score']
                segment.rpca_indicators = results['artifact_analysis']['deepfake_indicators']
                segment.rpca_artifact_audio_path = results['artifact_audio_path']
                segment.rpca_visualizations = results['visualizations']
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
                
                logger.info(f"Audio X-Ray analysis completed for segment {segment.segment_id}")
                
        except Exception as e:
            logger.error(f"Error in Audio X-Ray analysis for segment {segment.segment_id}: {str(e)}")
        
        return segment


class IdentityClusterer:
    """Clustering engine for speaker identity analysis and outlier detection."""
    
    def __init__(self, config: ForensicConfig):
        self.config = config
        self.scaler = StandardScaler()
        logger.info("Initialized IdentityClusterer")
    
    def extract_embeddings(self, segments: List[ForensicSegment]) -> np.ndarray:
        """Extract feature embeddings from forensic segments."""
        features = []
        
        for segment in segments:
            # Create feature vector from forensic metrics
            feature_vector = [
                segment.zcr,
                segment.spectral_flatness,
                segment.jitter,
                segment.shimmer,
                segment.breath_events,
                segment.pause_ratio,
                float(segment.high_freq_shelf),
                float(segment.pitch_artifacts),
                float(segment.unnatural_rhythm)
            ]
            
            # Add RPCA features if available
            if segment.rpca_deepfake_likelihood is not None:
                feature_vector.extend([
                    segment.rpca_deepfake_likelihood,
                    segment.rpca_rhythmicity_score if segment.rpca_rhythmicity_score is not None else 0.0,
                    segment.rpca_phase_blast_score if segment.rpca_phase_blast_score is not None else 0.0,
                    segment.rpca_silence_ratio,
                    segment.rpca_artifact_density,
                    segment.rpca_periodicity_score if segment.rpca_periodicity_score is not None else 0.0
                ])
            
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled
    
    def cluster_embeddings(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform t-SNE dimensionality reduction and DBSCAN clustering."""
        
        # t-SNE for visualization
        logger.info("Performing t-SNE dimensionality reduction...")
        
        # Handle small datasets - skip t-SNE if too few samples
        if len(embeddings) < 6:
            logger.warning(f"Too few samples ({len(embeddings)}) for t-SNE, using simple projection")
            # Use simple 2D projection instead
            if embeddings.shape[1] >= 2:
                tsne_embeddings = embeddings[:, :2]
            else:
                # Pad with zeros if needed
                tsne_embeddings = np.zeros((len(embeddings), 2))
                tsne_embeddings[:, :embeddings.shape[1]] = embeddings
        else:
            # Adjust perplexity for small datasets
            perplexity = min(self.config.tsne_perplexity, len(embeddings) - 1, 30)
            tsne = TSNE(n_components=2, perplexity=max(perplexity, 5), 
                       max_iter=self.config.tsne_n_iter, random_state=42)
            tsne_embeddings = tsne.fit_transform(embeddings)
        
        # DBSCAN for clustering
        logger.info("Performing DBSCAN clustering...")
        dbscan = DBSCAN(eps=self.config.dbscan_eps, min_samples=self.config.dbscan_min_samples)
        cluster_labels = dbscan.fit_predict(embeddings)
        
        return tsne_embeddings, cluster_labels
    
    def identify_outliers(self, segments: List[ForensicSegment], 
                         cluster_labels: np.ndarray) -> List[bool]:
        """Identify outliers based on clustering results."""
        
        # Count segments per cluster
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        # Mark small clusters as potential outliers
        outlier_threshold = max(3, len(segments) * 0.05)  # 5% or at least 3 segments
        
        is_outlier = []
        for label in cluster_labels:
            cluster_size = counts[unique_labels == label][0] if label in unique_labels else 0
            is_outlier.append(cluster_size < outlier_threshold or label == -1)  # -1 is noise in DBSCAN
        
        return is_outlier
    
    def visualize_clusters(self, segments: List[ForensicSegment], 
                          tsne_embeddings: np.ndarray, 
                          cluster_labels: np.ndarray,
                          output_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of t-SNE clusters with color coding."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create color map for clusters
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            cluster_points = tsne_embeddings[mask]
            
            if label == -1:  # Noise points
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          c=[colors[i]], s=50, alpha=0.7, label=f'Cluster {label}')
        
        # Highlight suspicious segments
        suspicious_mask = [seg.suspicion_score > 0.5 for seg in segments]
        suspicious_points = tsne_embeddings[suspicious_mask]
        if len(suspicious_points) > 0:
            ax.scatter(suspicious_points[:, 0], suspicious_points[:, 1], 
                      c='red', marker='*', s=100, alpha=0.8, 
                      edgecolors='darkred', linewidth=2, label='Suspicious')
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('Speaker Identity Clustering with Forensic Analysis')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster visualization saved to {output_path}")
        
        return fig


class EvidenceGenerator:
    """Generator for compiling and exporting forensic evidence."""
    
    def __init__(self, config: ForensicConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized EvidenceGenerator with output dir: {self.output_dir}")
    
    def generate_forensic_dossier(self, segments: List[ForensicSegment], 
                                 tsne_embeddings: Optional[np.ndarray] = None,
                                 cluster_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Generate comprehensive forensic dossier."""
        
        logger.info("Generating forensic dossier...")
        
        # Filter suspicious segments
        suspicious_segments = [seg for seg in segments if seg.suspicion_score > 0.5]
        
        dossier = {
            "metadata": {
                "total_segments": len(segments),
                "suspicious_segments": len(suspicious_segments),
                "suspicion_rate": len(suspicious_segments) / len(segments) if segments else 0,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "config": asdict(self.config)
            },
            "forensic_metrics": self._calculate_forensic_metrics_summary(segments),
            "suspicious_segments": self._serialize_segments(suspicious_segments),
            "cluster_analysis": self._analyze_clusters(segments, cluster_labels) if cluster_labels is not None else None,
            "recommendations": self._generate_recommendations(suspicious_segments)
        }
        
        # Save dossier
        dossier_path = self.output_dir / "forensic_dossier.json"
        with open(dossier_path, 'w') as f:
            json.dump(dossier, f, indent=2, default=str)
        
        logger.info(f"Forensic dossier saved to {dossier_path}")
        
        return dossier
    
    def _calculate_forensic_metrics_summary(self, segments: List[ForensicSegment]) -> Dict[str, Any]:
        """Calculate summary statistics for forensic metrics."""
        
        if not segments:
            return {}
        
        metrics = {
            "zcr": {"mean": np.mean([s.zcr for s in segments]),
                   "std": np.std([s.zcr for s in segments]),
                   "threshold": self.config.zcr_threshold},
            "spectral_flatness": {"mean": np.mean([s.spectral_flatness for s in segments]),
                                 "std": np.std([s.spectral_flatness for s in segments]),
                                 "threshold": self.config.spectral_flatness_threshold},
            "jitter": {"mean": np.mean([s.jitter for s in segments]),
                      "std": np.std([s.jitter for s in segments]),
                      "threshold": self.config.jitter_threshold},
            "shimmer": {"mean": np.mean([s.shimmer for s in segments]),
                       "std": np.std([s.shimmer for s in segments]),
                       "threshold": self.config.shimmer_threshold},
            "breath_events": {"mean": np.mean([s.breath_events for s in segments]),
                             "std": np.std([s.breath_events for s in segments]),
                             "threshold": self.config.min_breath_events_per_minute},
            "suspicion_scores": {"mean": np.mean([s.suspicion_score for s in segments]),
                                "std": np.std([s.suspicion_score for s in segments]),
                                "max": np.max([s.suspicion_score for s in segments])}
        }
        
        # Add detection rates
        metrics["detection_rates"] = {
            "high_freq_shelf": np.mean([s.high_freq_shelf for s in segments]),
            "pitch_artifacts": np.mean([s.pitch_artifacts for s in segments]),
            "unnatural_rhythm": np.mean([s.unnatural_rhythm for s in segments])
        }
        
        return metrics
    
    def _serialize_segments(self, segments: List[ForensicSegment]) -> List[Dict[str, Any]]:
        """Serialize segments to dictionary format."""
        serialized = []
        
        for seg in segments:
            seg_dict = asdict(seg)
            # Convert numpy arrays to lists for JSON serialization
            if seg.embedding is not None:
                seg_dict['embedding'] = seg.embedding.tolist()
            if seg.audio_data is not None:
                seg_dict['audio_data'] = seg.audio_data.tolist()
            serialized.append(seg_dict)
        
        return serialized
    
    def _analyze_clusters(self, segments: List[ForensicSegment], 
                         cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze clustering results."""
        
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        cluster_analysis = {
            "num_clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
            "noise_points": counts[unique_labels == -1][0] if -1 in unique_labels else 0,
            "cluster_sizes": {str(label): int(count) for label, count in zip(unique_labels, counts) if label != -1},
            "suspicious_by_cluster": {}
        }
        
        # Analyze suspicious segments by cluster
        for label in unique_labels:
            if label == -1:
                continue
            
            cluster_mask = cluster_labels == label
            cluster_segments = [segments[i] for i in range(len(segments)) if cluster_mask[i]]
            suspicious_in_cluster = [seg for seg in cluster_segments if seg.suspicion_score > 0.5]
            
            cluster_analysis["suspicious_by_cluster"][str(label)] = {
                "total": len(cluster_segments),
                "suspicious": len(suspicious_in_cluster),
                "suspicion_rate": len(suspicious_in_cluster) / len(cluster_segments) if cluster_segments else 0
            }
        
        return cluster_analysis
    
    def _generate_recommendations(self, suspicious_segments: List[ForensicSegment]) -> List[str]:
        """Generate recommendations based on forensic analysis."""
        
        recommendations = []
        
        if not suspicious_segments:
            recommendations.append("No suspicious segments detected. Audio appears to be authentic.")
            return recommendations
        
        # Analyze common patterns in suspicious segments
        high_freq_count = sum(1 for s in suspicious_segments if s.high_freq_shelf)
        pitch_artifact_count = sum(1 for s in suspicious_segments if s.pitch_artifacts)
        rhythm_count = sum(1 for s in suspicious_segments if s.unnatural_rhythm)
        
        suspicion_rate = len(suspicious_segments)
        
        if suspicion_rate > 0.5:
            recommendations.append("HIGH RISK: More than 50% of segments show suspicious characteristics.")
            recommendations.append("Recommendation: Immediate manual review required.")
        elif suspicion_rate > 0.2:
            recommendations.append("MODERATE RISK: Significant portion of segments show suspicious patterns.")
            recommendations.append("Recommendation: Enhanced verification procedures recommended.")
        
        if high_freq_count > len(suspicious_segments) * 0.5:
            recommendations.append("High-frequency cutoff detected in multiple segments - possible compression artifact.")
        
        if pitch_artifact_count > len(suspicious_segments) * 0.5:
            recommendations.append("Pitch artifacts detected - possible voice synthesis indicators.")
        
        if rhythm_count > len(suspicious_segments) * 0.3:
            recommendations.append("Unnatural rhythm patterns detected - possible synthetic speech markers.")
        
        # Speaker-specific recommendations
        suspicious_speakers = set(s.speaker for s in suspicious_segments)
        if len(suspicious_speakers) > 1:
            recommendations.append(f"Multiple speakers show suspicious patterns: {', '.join(suspicious_speakers)}")
            recommendations.append("Recommendation: Investigate potential widespread synthesis or cloning.")
        else:
            recommendations.append(f"Suspicious patterns concentrated in speaker: {list(suspicious_speakers)[0]}")
            recommendations.append("Recommendation: Focus verification on this speaker's segments.")
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    config = ForensicConfig()
    engine = AudioForensicsEngine(config)
    clusterer = IdentityClusterer(config)
    generator = EvidenceGenerator(config)
    
    logger.info("Forensic Deepfake Detection System initialized successfully")
    logger.info("Ready to analyze audio segments for synthetic voice detection")
