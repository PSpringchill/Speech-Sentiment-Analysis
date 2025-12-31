#!/usr/bin/env python3
"""
Audio X-Ray: RPCA-based Forensic Analysis for Deepfake Detection

This module implements Robust Principal Component Analysis (RPCA) to separate
audio into Low-Rank (smooth voice) and Sparse (artifacts/glitches) components,
enabling detection of synthetic voice characteristics that are otherwise
hidden under the main voice signal.

The Concept: Audio X-Ray
- Standard analysis looks at the whole sound
- RPCA mathematically splits audio into two layers:
  * Low-Rank Matrix (L): Smooth, repetitive parts - the "perfect" human voice
  * Sparse Matrix (S): Non-repetitive, spiky events - metallic clicks, breath artifacts

By listening only to the Sparse Matrix (S), we remove the distracting voice
and hear only the "digital errors" that indicate AI-generated audio.
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sporco.admm.rpca import RobustPCA, RobustPCAOptions
from sporco import util
from typing import Tuple, Optional, Dict, Any
import os
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioXRayConfig:
    """Configuration for Audio X-Ray RPCA analysis"""
    # RPCA parameters
    lambda_parameter: float = 1.0  # RPCA regularization parameter
    rho: float = 1.0  # ADMM penalty parameter
    max_iter: int = 1000  # Maximum ADMM iterations
    tol: float = 1e-6  # Convergence tolerance
    
    # Audio processing parameters
    sample_rate: int = 44100  # Analysis sample rate
    n_fft: int = 2048  # FFT window size
    hop_length: int = 512  # STFT hop length
    win_length: Optional[int] = None  # STFT window length
    
    # Artifact detection parameters
    artifact_threshold_multiplier: float = 3.0  # Threshold for artifact detection
    min_artifact_duration: float = 0.01  # Minimum duration for valid artifacts (seconds)
    
    # Output parameters
    output_dir: str = "audio_xray_output"  # Output directory for results
    save_visualizations: bool = True  # Whether to save spectrogram plots
    save_audio: bool = True  # Whether to save artifact audio


class AudioXRay:
    """
    Audio X-Ray: RPCA-based forensic analysis for deepfake detection
    
    This class implements the core functionality to separate audio into
    Low-Rank (voice) and Sparse (artifacts) components using Robust PCA.
    """
    
    def __init__(self, config: Optional[AudioXRayConfig] = None):
        """
        Initialize Audio X-Ray analyzer
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or AudioXRayConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file for analysis
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        logger.info(f"Loading audio from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Load audio at specified sample rate
        y, sr = librosa.load(file_path, sr=self.config.sample_rate)
        
        logger.info(f"Audio loaded: {len(y)} samples, {sr} Hz, {len(y)/sr:.2f} seconds")
        
        return y, sr
    
    def compute_spectrogram(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude spectrogram for RPCA analysis
        
        Args:
            audio: Audio time series
            sr: Sample rate
            
        Returns:
            Tuple of (magnitude_spectrogram, phase_spectrogram)
        """
        logger.info("Computing spectrogram for RPCA analysis")
        
        # Compute STFT
        S_full = librosa.stft(
            audio, 
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        
        # Separate magnitude and phase
        S_magnitude, S_phase = librosa.magphase(S_full)
        
        logger.info(f"Spectrogram computed: {S_magnitude.shape[0]} freq bins, {S_magnitude.shape[1]} time frames")
        
        return S_magnitude, S_phase
    
    def apply_rpca(self, magnitude_spectrogram: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Robust PCA to separate Low-Rank and Sparse components
        
        Args:
            magnitude_spectrogram: Magnitude spectrogram matrix
            
        Returns:
            Tuple of (low_rank_matrix, sparse_matrix)
        """
        logger.info("Applying Robust PCA (Audio X-Ray)")
        
        # Configure RPCA solver with default options
        opt = RobustPCAOptions()
        
        # Apply RPCA
        rpca_solver = RobustPCA(
            magnitude_spectrogram, 
            lmbda=self.config.lambda_parameter,
            opt=opt
        )
        
        # Run RPCA decomposition
        low_rank, sparse = rpca_solver.solve()
        
        # Convert to numpy arrays if needed
        if hasattr(low_rank, 'asarray'):
            low_rank = low_rank.asarray()
        if hasattr(sparse, 'asarray'):
            sparse = sparse.asarray()
        
        logger.info(f"RPCA completed: Low-rank shape {low_rank.shape}, Sparse shape {sparse.shape}")
        
        return low_rank, sparse
    
    def filter_artifacts(self, sparse_matrix: np.ndarray) -> np.ndarray:
        """
        Apply hard threshold filtering to extract strong artifacts
        
        Args:
            sparse_matrix: Sparse component from RPCA
            
        Returns:
            Filtered sparse matrix with only strong artifacts
        """
        logger.info("Filtering artifacts with hard threshold")
        
        # Calculate threshold
        mean_value = np.mean(sparse_matrix)
        threshold = mean_value * self.config.artifact_threshold_multiplier
        
        # Apply hard threshold
        filtered_sparse = np.where(sparse_matrix > threshold, sparse_matrix, 0)
        
        # Count artifacts
        artifact_count = np.sum(filtered_sparse > 0)
        total_elements = filtered_sparse.size
        artifact_percentage = (artifact_count / total_elements) * 100
        
        logger.info(f"Artifact filtering: {artifact_count} artifacts ({artifact_percentage:.2f}% of spectrogram)")
        logger.info(f"Threshold: {threshold:.6f} (mean: {mean_value:.6f})")
        
        return filtered_sparse
    
    def reconstruct_artifact_audio(self, 
                                 filtered_sparse: np.ndarray, 
                                 original_phase: np.ndarray, 
                                 sr: int) -> np.ndarray:
        """
        Reconstruct audio time series from filtered sparse component
        
        Args:
            filtered_sparse: Filtered sparse spectrogram
            original_phase: Original phase information
            sr: Sample rate
            
        Returns:
            Reconstructed audio time series containing only artifacts
        """
        logger.info("Reconstructing artifact audio")
        
        # Combine filtered sparse with original phase
        S_artifacts = filtered_sparse * original_phase
        
        # Inverse STFT to reconstruct audio
        y_artifacts = librosa.istft(
            S_artifacts,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        
        logger.info(f"Artifact audio reconstructed: {len(y_artifacts)} samples")
        
        return y_artifacts
    
    def analyze_artifacts(self, 
                        artifact_audio: np.ndarray, 
                        sr: int,
                        sparse_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analyze reconstructed artifacts for deepfake indicators
        
        Args:
            artifact_audio: Reconstructed artifact audio
            sr: Sample rate
            sparse_matrix: Filtered sparse spectrogram
            
        Returns:
            Dictionary containing artifact analysis results
        """
        logger.info("Analyzing artifacts for deepfake indicators")
        
        analysis = {}
        
        # 1. Rhythmicity Analysis (for metallic clicking)
        # Compute zero-crossing rate to detect rhythmic patterns
        zcr = librosa.feature.zero_crossing_rate(artifact_audio)[0]
        rhythmicity_score = np.std(zcr)  # Higher std = more rhythmic
        
        # 2. Spectral Analysis (for phase blasts)
        # Compute spectral centroid to detect high-frequency bursts
        spectral_centroids = librosa.feature.spectral_centroid(y=artifact_audio, sr=sr)[0]
        phase_blast_score = np.max(spectral_centroids) / np.mean(spectral_centroids)
        
        # 3. Silence Analysis (for missing breaths)
        # Compute energy in silent regions
        energy = np.sum(artifact_audio ** 2)
        silence_ratio = np.sum(np.abs(artifact_audio) < 0.001) / len(artifact_audio)
        
        # 4. Artifact Density
        artifact_density = np.sum(sparse_matrix > 0) / sparse_matrix.size
        
        # 5. Temporal Distribution
        # Check if artifacts are periodic (indicating vocoder artifacts)
        temporal_frames = np.sum(sparse_matrix > 0, axis=0)
        periodicity_score = np.max(temporal_frames) / (np.mean(temporal_frames) + 1e-8)
        
        analysis.update({
            'rhythmicity_score': float(rhythmicity_score),
            'phase_blast_score': float(phase_blast_score),
            'silence_ratio': float(silence_ratio),
            'artifact_density': float(artifact_density),
            'periodicity_score': float(periodicity_score),
            'total_energy': float(energy),
            'zcr_mean': float(np.mean(zcr)),
            'spectral_centroid_mean': float(np.mean(spectral_centroids))
        })
        
        # Determine deepfake likelihood
        deepfake_indicators = []
        
        if rhythmicity_score > 0.1:
            deepfake_indicators.append("Rhythmic metallic clicking detected")
        
        if phase_blast_score > 3.0:
            deepfake_indicators.append("Phase blast events detected")
        
        if silence_ratio > 0.8:
            deepfake_indicators.append("Excessive silence (missing breaths)")
        
        if periodicity_score > 2.0:
            deepfake_indicators.append("Periodic artifact patterns (vocoder signature)")
        
        analysis['deepfake_indicators'] = deepfake_indicators
        analysis['deepfake_likelihood'] = len(deepfake_indicators) / 4.0  # Normalize to 0-1
        
        logger.info(f"Artifact analysis completed: {len(deepfake_indicators)} indicators found")
        
        return analysis
    
    def create_visualizations(self, 
                            original_magnitude: np.ndarray,
                            low_rank: np.ndarray,
                            sparse: np.ndarray,
                            filtered_sparse: np.ndarray,
                            file_name: str) -> Dict[str, str]:
        """
        Create visualization plots for RPCA analysis
        
        Args:
            original_magnitude: Original magnitude spectrogram
            low_rank: Low-rank component
            sparse: Sparse component
            filtered_sparse: Filtered sparse component
            file_name: Base name for output files
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if not self.config.save_visualizations:
            return {}
        
        logger.info("Creating RPCA visualizations")
        
        # Convert to dB for better visualization
        orig_db = librosa.amplitude_to_db(original_magnitude, ref=np.max)
        low_rank_db = librosa.amplitude_to_db(low_rank, ref=np.max)
        sparse_db = librosa.amplitude_to_db(sparse + 1e-8, ref=np.max)  # Add small value to avoid log(0)
        filtered_sparse_db = librosa.amplitude_to_db(filtered_sparse + 1e-8, ref=np.max)
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Audio X-Ray RPCA Analysis: {file_name}', fontsize=16)
        
        # Original spectrogram
        im1 = axes[0, 0].imshow(orig_db, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('Original Spectrogram')
        axes[0, 0].set_xlabel('Time Frames')
        axes[0, 0].set_ylabel('Frequency Bins')
        plt.colorbar(im1, ax=axes[0, 0], format='%+2.0f dB')
        
        # Low-rank component (the voice)
        im2 = axes[0, 1].imshow(low_rank_db, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title('Low-Rank Component (L) - The Voice')
        axes[0, 1].set_xlabel('Time Frames')
        axes[0, 1].set_ylabel('Frequency Bins')
        plt.colorbar(im2, ax=axes[0, 1], format='%+2.0f dB')
        
        # Sparse component (artifacts)
        im3 = axes[1, 0].imshow(sparse_db, aspect='auto', origin='lower', cmap='hot')
        axes[1, 0].set_title('Sparse Component (S) - All Artifacts')
        axes[1, 0].set_xlabel('Time Frames')
        axes[1, 0].set_ylabel('Frequency Bins')
        plt.colorbar(im3, ax=axes[1, 0], format='%+2.0f dB')
        
        # Filtered sparse component (strong artifacts only)
        im4 = axes[1, 1].imshow(filtered_sparse_db, aspect='auto', origin='lower', cmap='hot')
        axes[1, 1].set_title('Filtered Sparse (S) - Strong Artifacts Only')
        axes[1, 1].set_xlabel('Time Frames')
        axes[1, 1].set_ylabel('Frequency Bins')
        plt.colorbar(im4, ax=axes[1, 1], format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / f"{file_name}_rpca_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create artifact detail view
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Artifact Detail Analysis: {file_name}', fontsize=16)
        
        # Sparse component detail
        im1 = axes[0].imshow(sparse_db, aspect='auto', origin='lower', cmap='hot')
        axes[0].set_title('Sparse Component - All Artifacts')
        axes[0].set_xlabel('Time Frames')
        axes[0].set_ylabel('Frequency Bins')
        plt.colorbar(im1, ax=axes[0], format='%+2.0f dB')
        
        # Filtered sparse detail
        im2 = axes[1].imshow(filtered_sparse_db, aspect='auto', origin='lower', cmap='hot')
        axes[1].set_title('Filtered Sparse - Strong Artifacts')
        axes[1].set_xlabel('Time Frames')
        axes[1].set_ylabel('Frequency Bins')
        plt.colorbar(im2, ax=axes[1], format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Save detail view
        detail_path = self.output_dir / f"{file_name}_artifact_detail.png"
        plt.savefig(detail_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved: {viz_path}, {detail_path}")
        
        return {
            'full_analysis': str(viz_path),
            'artifact_detail': str(detail_path)
        }
    
    def run_audio_xray(self, file_path: str) -> Dict[str, Any]:
        """
        Run complete Audio X-Ray analysis on audio file
        
        Args:
            file_path: Path to audio file to analyze
            
        Returns:
            Dictionary containing complete analysis results
        """
        logger.info(f"Starting Audio X-Ray analysis for: {file_path}")
        
        # Extract base filename for outputs
        base_name = Path(file_path).stem
        
        try:
            # 1. Load audio
            audio, sr = self.load_audio(file_path)
            
            # 2. Compute spectrogram
            magnitude_spectrogram, phase_spectrogram = self.compute_spectrogram(audio, sr)
            
            # 3. Apply RPCA
            low_rank, sparse = self.apply_rpca(magnitude_spectrogram)
            
            # 4. Filter artifacts
            filtered_sparse = self.filter_artifacts(sparse)
            
            # 5. Reconstruct artifact audio
            artifact_audio = self.reconstruct_artifact_audio(filtered_sparse, phase_spectrogram, sr)
            
            # 6. Analyze artifacts
            artifact_analysis = self.analyze_artifacts(artifact_audio, sr, filtered_sparse)
            
            # 7. Create visualizations
            visualizations = self.create_visualizations(
                magnitude_spectrogram, low_rank, sparse, filtered_sparse, base_name
            )
            
            # 8. Save artifact audio
            artifact_audio_path = None
            if self.config.save_audio:
                artifact_audio_path = self.output_dir / f"{base_name}_artifacts_only.wav"
                sf.write(artifact_audio_path, artifact_audio, sr)
                logger.info(f"Artifact audio saved: {artifact_audio_path}")
            
            # Compile results
            results = {
                'file_path': file_path,
                'base_name': base_name,
                'audio_info': {
                    'duration': len(audio) / sr,
                    'sample_rate': sr,
                    'samples': len(audio)
                },
                'rpca_results': {
                    'low_rank_shape': low_rank.shape,
                    'sparse_shape': sparse.shape,
                    'lambda_parameter': self.config.lambda_parameter
                },
                'artifact_analysis': artifact_analysis,
                'visualizations': visualizations,
                'artifact_audio_path': str(artifact_audio_path) if artifact_audio_path else None,
                'config': self.config.__dict__
            }
            
            self.results[base_name] = results
            
            logger.info(f"Audio X-Ray analysis completed for: {file_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error in Audio X-Ray analysis: {str(e)}")
            raise
    
    def generate_report(self, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate forensic report from Audio X-Ray analysis
        
        Args:
            results: Analysis results. If None, uses stored results.
            
        Returns:
            Formatted report string
        """
        if results is None:
            if not self.results:
                return "No analysis results available."
            # Use the most recent result
            results = list(self.results.values())[-1]
        
        report = []
        report.append("=" * 60)
        report.append("AUDIO X-RAY FORENSIC REPORT")
        report.append("=" * 60)
        report.append(f"File: {results['file_path']}")
        report.append(f"Duration: {results['audio_info']['duration']:.2f} seconds")
        report.append(f"Sample Rate: {results['audio_info']['sample_rate']} Hz")
        report.append("")
        
        # RPCA Analysis Summary
        report.append("RPCA DECOMPOSITION:")
        report.append(f"  Low-Rank Matrix Shape: {results['rpca_results']['low_rank_shape']}")
        report.append(f"  Sparse Matrix Shape: {results['rpca_results']['sparse_shape']}")
        report.append(f"  Lambda Parameter: {results['rpca_results']['lambda_parameter']}")
        report.append("")
        
        # Artifact Analysis
        analysis = results['artifact_analysis']
        report.append("ARTIFACT ANALYSIS:")
        report.append(f"  Deepfake Likelihood: {analysis['deepfake_likelihood']:.2%}")
        report.append(f"  Rhythmicity Score: {analysis['rhythmicity_score']:.4f}")
        report.append(f"  Phase Blast Score: {analysis['phase_blast_score']:.4f}")
        report.append(f"  Silence Ratio: {analysis['silence_ratio']:.2%}")
        report.append(f"  Artifact Density: {analysis['artifact_density']:.2%}")
        report.append(f"  Periodicity Score: {analysis['periodicity_score']:.4f}")
        report.append("")
        
        # Deepfake Indicators
        if analysis['deepfake_indicators']:
            report.append("DEEPFAKE INDICATORS DETECTED:")
            for indicator in analysis['deepfake_indicators']:
                report.append(f"  ⚠️  {indicator}")
            report.append("")
        else:
            report.append("DEEPFAKE INDICATORS: None detected")
            report.append("")
        
        # Evidence Assessment
        report.append("EVIDENCE ASSESSMENT:")
        if analysis['deepfake_likelihood'] > 0.5:
            report.append("  🔴 HIGH PROBABILITY of synthetic voice")
            report.append("  Recommendation: Further forensic analysis recommended")
        elif analysis['deepfake_likelihood'] > 0.25:
            report.append("  🟡 MEDIUM PROBABILITY of synthetic voice")
            report.append("  Recommendation: Additional verification needed")
        else:
            report.append("  🟢 LOW PROBABILITY of synthetic voice")
            report.append("  Recommendation: Likely authentic voice")
        report.append("")
        
        # Output Files
        report.append("GENERATED FILES:")
        if results['artifact_audio_path']:
            report.append(f"  Artifact Audio: {results['artifact_audio_path']}")
        for viz_name, viz_path in results['visualizations'].items():
            report.append(f"  {viz_name.replace('_', ' ').title()}: {viz_path}")
        report.append("")
        
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """
    Main function for standalone Audio X-Ray analysis
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio X-Ray: RPCA-based forensic analysis")
    parser.add_argument("input_audio", help="Input audio file for analysis")
    parser.add_argument("--output-dir", default="audio_xray_output", help="Output directory")
    parser.add_argument("--lambda-param", type=float, default=1.0, help="RPCA lambda parameter")
    parser.add_argument("--threshold", type=float, default=3.0, help="Artifact threshold multiplier")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    parser.add_argument("--no-audio", action="store_true", help="Skip artifact audio export")
    
    args = parser.parse_args()
    
    # Configure Audio X-Ray
    config = AudioXRayConfig(
        lambda_parameter=args.lambda_param,
        artifact_threshold_multiplier=args.threshold,
        output_dir=args.output_dir,
        save_visualizations=not args.no_viz,
        save_audio=not args.no_audio
    )
    
    # Run analysis
    xray = AudioXRay(config)
    
    print("🔬 Starting Audio X-Ray Analysis...")
    print(f"📁 Input: {args.input_audio}")
    print(f"📁 Output: {args.output_dir}")
    print()
    
    try:
        results = xray.run_audio_xray(args.input_audio)
        
        # Print report
        report = xray.generate_report(results)
        print(report)
        
        # Save report
        report_path = Path(args.output_dir) / f"{Path(args.input_audio).stem}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📋 Report saved: {report_path}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
