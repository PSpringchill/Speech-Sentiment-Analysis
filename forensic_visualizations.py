#!/usr/bin/env python3
"""
Forensic Visualization Components
=================================

This module provides advanced visualization components for forensic audio analysis,
including spectrogram analysis with high-frequency cutoff detection, pitch contour
tracking, and bispectral analysis for glitch detection.

Features:
- Spectrogram visualization with high-frequency shelf detection
- Pitch (F0) contour tracking with jitter and step artifact highlighting
- Bispectral analysis for glitch density visualization
- Enhanced forensic dashboard integration
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import librosa
import librosa.display
from scipy import signal
from scipy.stats import entropy
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from forensic_deepfake_detection import ForensicConfig, ForensicSegment


class ForensicVisualizer:
    """Advanced visualization components for forensic audio analysis."""
    
    def __init__(self, config: ForensicConfig):
        self.config = config
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_spectrogram_with_cutoff_analysis(self, audio: np.ndarray, sr: int, 
                                            segment: ForensicSegment,
                                            output_path: Optional[str] = None) -> plt.Figure:
        """Create spectrogram visualization with high-frequency cutoff detection."""
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Forensic Spectrogram Analysis - {segment.segment_id}', fontsize=16)
        
        # 1. Main spectrogram
        ax1 = axes[0]
        
        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=self.config.n_fft, hop_length=self.config.hop_length)
        magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.config.n_fft)
        times = librosa.frames_to_time(np.arange(magnitude_db.shape[1]), sr=sr, hop_length=self.config.hop_length)
        
        # Plot spectrogram
        img = librosa.display.specshow(magnitude_db, sr=sr, hop_length=self.config.hop_length,
                                      x_axis='time', y_axis='hz', ax=ax1, cmap='viridis')
        
        # Add high-frequency cutoff line
        cutoff_line = ax1.axhline(y=self.config.high_freq_cutoff, color='red', 
                                 linestyle='--', linewidth=2, label=f'Cutoff ({self.config.high_freq_cutoff}Hz)')
        
        # Highlight cutoff region if detected
        if segment.high_freq_shelf_detected:
            cutoff_idx = np.argmin(np.abs(freqs - self.config.high_freq_cutoff))
            cutoff_region = patches.Rectangle((0, self.config.high_freq_cutoff), 
                                            times[-1], freqs[-1] - self.config.high_freq_cutoff,
                                            linewidth=2, edgecolor='red', facecolor='red', alpha=0.2)
            ax1.add_patch(cutoff_region)
            ax1.text(times[-1]*0.7, self.config.high_freq_cutoff + 1000, 
                    'HIGH-FREQ SHELF DETECTED', color='red', fontweight='bold', fontsize=12)
        
        ax1.set_title('Spectrogram with High-Frequency Analysis')
        ax1.legend()
        plt.colorbar(img, ax=ax1, format='%+2.0f dB')
        
        # 2. Frequency spectrum analysis
        ax2 = axes[1]
        
        # Compute average spectrum
        avg_spectrum = np.mean(np.abs(stft), axis=1)
        avg_spectrum_db = librosa.amplitude_to_db(avg_spectrum)
        
        # Plot spectrum
        ax2.plot(freqs, avg_spectrum_db, color='blue', linewidth=2, label='Average Spectrum')
        
        # Add cutoff line
        ax2.axvline(x=self.config.high_freq_cutoff, color='red', linestyle='--', 
                   linewidth=2, label=f'Cutoff ({self.config.high_freq_cutoff}Hz)')
        
        # Highlight low-frequency vs high-frequency energy ratio
        cutoff_idx = np.argmin(np.abs(freqs - self.config.high_freq_cutoff))
        low_freq_energy = np.sum(avg_spectrum[:cutoff_idx])
        high_freq_energy = np.sum(avg_spectrum[cutoff_idx:])
        
        if low_freq_energy > 0:
            ratio = high_freq_energy / low_freq_energy
            ax2.text(self.config.high_freq_cutoff + 1000, np.max(avg_spectrum_db) - 10,
                    f'HF/LF Ratio: {ratio:.3f}', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow" if ratio < 0.1 else "white"))
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_title('Frequency Spectrum Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(20000, freqs[-1]))
        
        # 3. Spectral characteristics over time
        ax3 = axes[2]
        
        # Compute spectral features over time
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.config.hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.config.hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=self.config.hop_length)[0]
        
        time_frames = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=sr, hop_length=self.config.hop_length)
        
        # Plot spectral features
        ax3.plot(time_frames, spectral_centroids, color='blue', label='Spectral Centroid', linewidth=2)
        ax3.plot(time_frames, spectral_rolloff, color='green', label='Spectral Rolloff', linewidth=2)
        ax3.plot(time_frames, spectral_bandwidth, color='orange', label='Spectral Bandwidth', linewidth=2)
        
        # Add cutoff line
        ax3.axhline(y=self.config.high_freq_cutoff, color='red', linestyle='--', 
                   linewidth=2, label=f'Cutoff ({self.config.high_freq_cutoff}Hz)')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_title('Spectral Characteristics Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_pitch_contour_analysis(self, audio: np.ndarray, sr: int, 
                                  segment: ForensicSegment,
                                  output_path: Optional[str] = None) -> plt.Figure:
        """Create pitch contour visualization with jitter and step artifact analysis."""
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Pitch Contour Analysis - {segment.segment_id}', fontsize=16)
        
        # Extract pitch using PYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, sr=sr, fmin=50, fmax=400, frame_length=2048, hop_length=512
        )
        
        time_frames = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
        
        # 1. Raw pitch contour
        ax1 = axes[0]
        
        # Plot pitch contour
        valid_mask = ~np.isnan(f0)
        ax1.plot(time_frames[valid_mask], f0[valid_mask], 'b-', linewidth=2, label='F0 Contour')
        ax1.scatter(time_frames[valid_mask], f0[valid_mask], c='blue', s=10, alpha=0.5)
        
        # Highlight unvoiced regions
        unvoiced_mask = np.isnan(f0)
        if np.any(unvoiced_mask):
            ax1.scatter(time_frames[unvoiced_mask], np.ones(np.sum(unvoiced_mask)) * 25, 
                       c='lightgray', s=5, alpha=0.5, label='Unvoiced')
        
        # Add jitter threshold lines
        if len(f0[valid_mask]) > 0:
            f0_mean = np.mean(f0[valid_mask])
            jitter_threshold = f0_mean * self.config.jitter_threshold
            ax1.axhline(y=f0_mean + jitter_threshold, color='red', linestyle='--', 
                      alpha=0.5, label='Jitter Threshold')
            ax1.axhline(y=f0_mean - jitter_threshold, color='red', linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Fundamental Frequency (Hz)')
        ax1.set_title('Pitch (F0) Contour')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 400)
        
        # 2. Pitch derivative and step detection
        ax2 = axes[1]
        
        # Calculate pitch derivative
        valid_f0 = f0[valid_mask]
        if len(valid_f0) > 1:
            f0_diff = np.diff(valid_f0)
            f0_second_diff = np.diff(f0_diff)
            
            time_diff = time_frames[valid_mask][1:]
            time_second_diff = time_frames[valid_mask][2:]
            
            # Plot first derivative
            ax2.plot(time_diff, f0_diff, 'g-', linewidth=2, label='F0 First Derivative', alpha=0.7)
            
            # Plot second derivative
            ax2_2 = ax2.twinx()
            ax2_2.plot(time_second_diff, f0_second_diff, 'r-', linewidth=2, label='F0 Second Derivative', alpha=0.7)
            ax2_2.set_ylabel('Second Derivative (Hz/s²)', color='red')
            ax2_2.tick_params(axis='y', labelcolor='red')
            
            # Highlight large jumps (step artifacts)
            jump_threshold = 2 * np.std(f0_diff)
            large_jumps = np.abs(f0_diff) > jump_threshold
            
            if np.any(large_jumps):
                ax2.scatter(time_diff[large_jumps], f0_diff[large_jumps], 
                          c='red', s=50, marker='o', label='Step Artifacts', zorder=5)
                
                # Add annotations for large jumps
                for i, (t, diff) in enumerate(zip(time_diff[large_jumps], f0_diff[large_jumps])):
                    if i < 5:  # Limit annotations to avoid clutter
                        ax2.annotate(f'Jump: {diff:.1f}Hz', (t, diff), 
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=8, color='red',
                                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('First Derivative (Hz/s)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_title('Pitch Derivative Analysis - Step Artifact Detection')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Voicing probability and confidence
        ax3 = axes[2]
        
        # Plot voicing probability
        ax3.plot(time_frames, voiced_probs, 'purple', linewidth=2, label='Voicing Probability')
        ax3.fill_between(time_frames, 0, voiced_probs, alpha=0.3, color='purple')
        
        # Add voicing threshold
        ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Voicing Threshold')
        
        # Highlight low confidence regions
        low_confidence = voiced_probs < 0.5
        if np.any(low_confidence):
            ax3.fill_between(time_frames[low_confidence], 0, 1, alpha=0.2, color='red', 
                           label='Low Confidence')
        
        # Add jitter and shimmer indicators
        jitter_text = f'Jitter: {segment.jitter:.4f} {"(HIGH)" if segment.jitter > self.config.jitter_threshold else "(OK)"}'
        shimmer_text = f'Shimmer: {segment.shimmer:.4f} {"(HIGH)" if segment.shimmer > self.config.shimmer_threshold else "(OK)"}'
        
        ax3.text(0.02, 0.95, jitter_text, transform=ax3.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow" if segment.jitter > self.config.jitter_threshold else "lightgreen"))
        ax3.text(0.02, 0.85, shimmer_text, transform=ax3.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow" if segment.shimmer > self.config.shimmer_threshold else "lightgreen"))
        
        # Add pitch artifact detection status
        if segment.pitch_artifacts_detected:
            ax3.text(0.98, 0.95, 'PITCH ARTIFACTS DETECTED', transform=ax3.transAxes, 
                    fontsize=12, fontweight='bold', color='red', ha='right',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Voicing Probability')
        ax3.set_title('Voicing Confidence Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_bispectral_glitch_analysis(self, audio: np.ndarray, sr: int,
                                       segment: ForensicSegment,
                                       output_path: Optional[str] = None) -> plt.Figure:
        """Create bispectral analysis visualization for glitch detection."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Bispectral Glitch Analysis - {segment.segment_id}', fontsize=16)
        
        # 1. Bispectrum magnitude
        ax1 = axes[0, 0]
        
        # Compute bispectrum (simplified version for visualization)
        # Note: Full bispectral analysis is computationally intensive
        # This is a simplified version for demonstration
        
        # Use shorter segment for bispectral analysis
        bisect_audio = audio[:min(len(audio), sr*2)]  # Use max 2 seconds
        
        # Compute STFT for bispectral analysis
        f, t, Zxx = signal.stft(bisect_audio, fs=sr, nperseg=1024, noverlap=512)
        
        # Compute bispectrum magnitude (simplified)
        # In practice, this would involve more complex bispectral computation
        bispectrum_mag = np.abs(Zxx)**2
        
        # Plot bispectrum magnitude
        im1 = ax1.imshow(10 * np.log10(bispectrum_mag + 1e-10), aspect='auto', 
                        origin='lower', cmap='hot', extent=[t[0], t[-1], f[0], f[-1]/2])
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_title('Bispectrum Magnitude')
        plt.colorbar(im1, ax=ax1, format='%+2.0f dB')
        
        # 2. Glitch density over time
        ax2 = axes[0, 1]
        
        # Compute glitch density indicator (simplified)
        # This would normally involve bispectral coherence analysis
        glitch_density = self._compute_glitch_density_indicator(audio, sr)
        
        time_glitch = np.linspace(0, len(audio)/sr, len(glitch_density))
        
        ax2.plot(time_glitch, glitch_density, 'r-', linewidth=2, label='Glitch Density')
        ax2.fill_between(time_glitch, 0, glitch_density, alpha=0.3, color='red')
        
        # Add threshold line
        glitch_threshold = 0.3  # Simplified threshold
        ax2.axhline(y=glitch_threshold, color='black', linestyle='--', 
                   linewidth=2, label='Glitch Threshold')
        
        # Highlight high glitch regions
        high_glitch = glitch_density > glitch_threshold
        if np.any(high_glitch):
            ax2.fill_between(time_glitch[high_glitch], 0, glitch_density[high_glitch], 
                           alpha=0.5, color='red', label='High Glitch Regions')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Glitch Density')
        ax2.set_title('Temporal Glitch Density Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Phase analysis
        ax3 = axes[1, 0]
        
        # Compute instantaneous phase
        analytic_signal = signal.hilbert(audio)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        time_phase = np.linspace(0, len(audio)/sr, len(instantaneous_phase))
        
        # Plot phase
        ax3.plot(time_phase, instantaneous_phase, 'b-', linewidth=1, alpha=0.7)
        
        # Highlight phase discontinuities (potential glitches)
        phase_diff = np.diff(instantaneous_phase)
        phase_jumps = np.abs(phase_diff) > np.pi  # Large phase jumps
        
        if np.any(phase_jumps):
            jump_times = time_phase[1:][phase_jumps]
            ax3.scatter(jump_times, instantaneous_phase[1:][phase_jumps], 
                       c='red', s=30, marker='o', label='Phase Discontinuities', zorder=5)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Phase (radians)')
        ax3.set_title('Instantaneous Phase Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        
        # Create summary statistics visualization
        stats_data = {
            'Glitch Density': np.mean(glitch_density),
            'Phase Jumps': np.sum(phase_jumps) / len(audio) * sr,  # jumps per second
            'Spectral Flatness': segment.spectral_flatness,
            'ZCR': segment.zcr
        }
        
        # Create bar plot
        items = list(stats_data.keys())
        values = list(stats_data.values())
        colors = ['red' if v > 0.3 else 'orange' if v > 0.1 else 'green' for v in values]
        
        bars = ax4.bar(items, values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax4.set_ylabel('Value')
        ax4.set_title('Forensic Indicators Summary')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add overall glitch assessment
        overall_glitch_score = np.mean(glitch_density)
        if overall_glitch_score > 0.5:
            assessment = "HIGH GLITCH ACTIVITY"
            color = "red"
        elif overall_glitch_score > 0.2:
            assessment = "MODERATE GLITCH ACTIVITY"
            color = "orange"
        else:
            assessment = "LOW GLITCH ACTIVITY"
            color = "green"
        
        ax4.text(0.5, 0.95, assessment, transform=ax4.transAxes, 
                fontsize=12, fontweight='bold', color=color, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _compute_glitch_density_indicator(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute simplified glitch density indicator."""
        
        # This is a simplified version of glitch density detection
        # In practice, this would involve more sophisticated bispectral analysis
        
        frame_length = int(0.05 * sr)  # 50ms frames
        hop_length = frame_length // 2
        
        glitch_density = []
        
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            
            # Compute spectral features
            fft = np.fft.fft(frame)
            magnitude = np.abs(fft)
            
            # Compute spectral flatness (indicator of noise-like content)
            if np.sum(magnitude) > 0:
                geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
                arithmetic_mean = np.mean(magnitude)
                flatness = geometric_mean / arithmetic_mean
            else:
                flatness = 0
            
            # Compute zero-crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            
            # Compute energy variation
            energy = np.sum(frame**2)
            
            # Combine indicators (simplified)
            glitch_indicator = flatness * 0.4 + zcr * 0.3 + (1 - energy / np.max(frame**2 + 1e-10)) * 0.3
            
            glitch_density.append(glitch_indicator)
        
        return np.array(glitch_density)
    
    def create_forensic_dashboard(self, segments: List[ForensicSegment], 
                                output_dir: str):
        """Create comprehensive forensic dashboard with all visualizations."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating forensic dashboard in {output_path}")
        
        # Create visualizations for suspicious segments
        suspicious_segments = [s for s in segments if s.is_suspicious]
        
        if not suspicious_segments:
            logger.warning("No suspicious segments found for detailed visualization")
            return
        
        # Limit visualizations to top suspicious segments to avoid too many files
        top_suspicious = sorted(suspicious_segments, key=lambda s: s.suspicion_score, reverse=True)[:5]
        
        for i, segment in enumerate(top_suspicious):
            if segment.audio_data is not None:
                # Spectrogram analysis
                spec_path = output_path / f"spectrogram_{segment.segment_id}.png"
                self.plot_spectrogram_with_cutoff_analysis(
                    segment.audio_data, self.config.sample_rate, segment, spec_path
                )
                
                # Pitch contour analysis
                pitch_path = output_path / f"pitch_contour_{segment.segment_id}.png"
                self.plot_pitch_contour_analysis(
                    segment.audio_data, self.config.sample_rate, segment, pitch_path
                )
                
                # Bispectral glitch analysis
                bispectral_path = output_path / f"bispectral_{segment.segment_id}.png"
                self.plot_bispectral_glitch_analysis(
                    segment.audio_data, self.config.sample_rate, segment, bispectral_path
                )
                
                logger.info(f"Created visualizations for segment {segment.segment_id}")
        
        # Create summary dashboard
        self._create_summary_dashboard(segments, output_path)
        
        logger.info("Forensic dashboard creation completed")
    
    def _create_summary_dashboard(self, segments: List[ForensicSegment], output_path: Path):
        """Create summary dashboard with overview of all forensic analysis."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Forensic Analysis Summary Dashboard', fontsize=16)
        
        suspicious_segments = [s for s in segments if s.is_suspicious]
        
        # 1. Suspicion score distribution
        ax1 = axes[0, 0]
        suspicion_scores = [s.suspicion_score for s in segments]
        ax1.hist(suspicion_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax1.set_title('Suspicion Score Distribution')
        ax1.set_xlabel('Suspicion Score')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Detection rates
        ax2 = axes[0, 1]
        detection_types = ['High-Freq\nShelf', 'Pitch\nArtifacts', 'Unnatural\nRhythm']
        detection_rates = [
            np.mean([s.high_freq_shelf_detected for s in segments]),
            np.mean([s.pitch_artifacts_detected for s in segments]),
            np.mean([s.unnatural_rhythm_detected for s in segments])
        ]
        
        colors = ['red' if rate > 0.3 else 'orange' if rate > 0.1 else 'green' for rate in detection_rates]
        bars = ax2.bar(detection_types, detection_rates, color=colors, alpha=0.7)
        ax2.set_title('Forensic Detection Rates')
        ax2.set_ylabel('Detection Rate')
        ax2.set_ylim(0, 1)
        
        # Add percentage labels
        for bar, rate in zip(bars, detection_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Speaker risk assessment
        ax3 = axes[0, 2]
        speakers = list(set(s.speaker_label for s in segments))
        speaker_risks = []
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s.speaker_label == speaker]
            suspicion_rate = len([s for s in speaker_segments if s.is_suspicious]) / len(speaker_segments)
            speaker_risks.append(suspicion_rate)
        
        colors = ['red' if risk > 0.5 else 'orange' if risk > 0.2 else 'green' for risk in speaker_risks]
        bars = ax3.bar(speakers, speaker_risks, color=colors, alpha=0.7)
        ax3.set_title('Speaker Risk Assessment')
        ax3.set_ylabel('Suspicion Rate')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for bar, risk in zip(bars, speaker_risks):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{risk:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Forensic metrics correlation
        ax4 = axes[1, 0]
        
        # Create correlation matrix of key metrics
        metrics_data = []
        for s in segments:
            metrics_data.append([
                s.zcr, s.spectral_flatness, s.jitter, s.shimmer, 
                s.breath_events, s.suspicion_score
            ])
        
        metrics_array = np.array(metrics_data)
        metric_names = ['ZCR', 'Flatness', 'Jitter', 'Shimmer', 'Breath', 'Suspicion']
        correlation_matrix = np.corrcoef(metrics_array.T)
        
        im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(metric_names)))
        ax4.set_yticks(range(len(metric_names)))
        ax4.set_xticklabels(metric_names, rotation=45)
        ax4.set_yticklabels(metric_names)
        ax4.set_title('Forensic Metrics Correlation')
        
        # Add correlation values
        for i in range(len(metric_names)):
            for j in range(len(metric_names)):
                text = ax4.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax4)
        
        # 5. Risk level summary
        ax5 = axes[1, 1]
        
        risk_levels = {'Low': 0, 'Medium': 0, 'High': 0}
        for segment in segments:
            if segment.suspicion_score > 0.5:
                risk_levels['High'] += 1
            elif segment.suspicion_score > 0.2:
                risk_levels['Medium'] += 1
            else:
                risk_levels['Low'] += 1
        
        colors = ['green', 'orange', 'red']
        wedges, texts, autotexts = ax5.pie(risk_levels.values(), labels=risk_levels.keys(),
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Segment Risk Distribution')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 6. Key statistics summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create text summary
        stats_text = f"""
        FORENSIC ANALYSIS SUMMARY
        
        Total Segments: {len(segments)}
        Suspicious Segments: {len(suspicious_segments)}
        Overall Suspicion Rate: {len(suspicious_segments)/len(segments):.1%}
        
        Avg Suspicion Score: {np.mean([s.suspicion_score for s in segments]):.3f}
        Max Suspicion Score: {np.max([s.suspicion_score for s in segments]):.3f}
        
        Key Findings:
        • High-Freq Shelf: {np.mean([s.high_freq_shelf_detected for s in segments]):.1%}
        • Pitch Artifacts: {np.mean([s.pitch_artifacts_detected for s in segments]):.1%}
        • Unnatural Rhythm: {np.mean([s.unnatural_rhythm_detected for s in segments]):.1%}
        
        Risk Assessment: {'HIGH' if len(suspicious_segments)/len(segments) > 0.5 else 'MEDIUM' if len(suspicious_segments)/len(segments) > 0.2 else 'LOW'}
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save summary dashboard
        dashboard_path = output_path / "forensic_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary dashboard saved to {dashboard_path}")


if __name__ == "__main__":
    # Example usage
    config = ForensicConfig()
    visualizer = ForensicVisualizer(config)
    
    # Generate synthetic audio for testing
    duration = 3.0
    sr = config.sample_rate
    samples = int(duration * sr)
    t = np.linspace(0, duration, samples)
    
    # Create synthetic speech-like signal
    f0 = 150 + 30 * np.sin(2 * np.pi * 2 * t)
    audio = np.sin(2 * np.pi * f0 * t)
    audio += 0.1 * np.random.randn(samples)
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Create test segment
    segment = ForensicSegment(
        segment_id="test_segment",
        timestamp="0.0",
        speaker="Test_Speaker",
        duration=duration,
        zcr=0.05,
        spectral_flatness=0.2,
        jitter=0.003,
        shimmer=0.02,
        breath_events=2,
        pause_ratio=0.1,
        high_freq_shelf=False,
        pitch_artifacts=True,
        unnatural_rhythm=False,
        suspicion_score=0.6,
        audio_data=audio
    )
    
    # Test visualizations
    output_dir = "test_forensic_output"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer.plot_spectrogram_with_cutoff_analysis(audio, sr, segment, 
                                                   f"{output_dir}/test_spectrogram.png")
    visualizer.plot_pitch_contour_analysis(audio, sr, segment, 
                                          f"{output_dir}/test_pitch.png")
    visualizer.plot_bispectral_glitch_analysis(audio, sr, segment, 
                                             f"{output_dir}/test_bispectral.png")
    
    print("Test visualizations created successfully!")
