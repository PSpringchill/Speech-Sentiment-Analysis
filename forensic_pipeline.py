#!/usr/bin/env python3
"""
Forensic Analysis Pipeline
===========================

This module implements the main forensic analysis pipeline that integrates
with the existing speech sentiment analysis system. It processes CSV data,
extracts audio segments, performs forensic analysis, and generates comprehensive
reports.

Usage:
    python forensic_pipeline.py --csv analysis_log.csv --audio audio_file.wav
    python forensic_pipeline.py --csv analysis_log.csv --audio-dir ./audio_segments/
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our forensic components
from forensic_deepfake_detection import (
    ForensicConfig, ForensicSegment, AudioForensicsEngine, 
    IdentityClusterer, EvidenceGenerator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ForensicPipeline:
    """Main pipeline for forensic deepfake detection analysis."""
    
    def __init__(self, config: Optional[ForensicConfig] = None):
        self.config = config or ForensicConfig()
        self.audio_engine = AudioForensicsEngine(self.config)
        self.clusterer = IdentityClusterer(self.config)
        self.evidence_generator = EvidenceGenerator(self.config)
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("Forensic Pipeline initialized")
    
    def analyze_csv_data(self, csv_path: str, audio_path: Optional[str] = None, max_segments: Optional[int] = None) -> Dict[str, Any]:
        """Analyze CSV data and perform forensic analysis on suspicious segments."""
        
        logger.info(f"Starting forensic analysis of CSV: {csv_path}")
        
        # Load and validate CSV data
        df = self._load_csv_data(csv_path)
        
        # Identify suspicious segments based on existing analysis
        suspicious_segments = self._identify_suspicious_segments(df)
        
        # Limit segments if specified
        if max_segments is not None and len(suspicious_segments) > max_segments:
            suspicious_segments = suspicious_segments[:max_segments]
            logger.info(f"Limited analysis to {max_segments} segments")
        
        if not suspicious_segments:
            logger.info("No suspicious segments identified in CSV data")
            return {"status": "no_suspicious_segments", "segments_analyzed": 0}
        
        logger.info(f"Identified {len(suspicious_segments)} suspicious segments")
        
        # Extract audio segments if audio path provided
        audio_segments = []
        if audio_path:
            audio_segments = self._extract_audio_segments(suspicious_segments, audio_path)
        else:
            logger.warning("No audio path provided. Performing analysis on CSV data only.")
            # Create synthetic segments for analysis
            audio_segments = self._create_synthetic_segments(suspicious_segments)
        
        # Perform forensic analysis on each segment
        forensic_segments = []
        for i, (csv_segment, audio_data) in enumerate(zip(suspicious_segments, audio_segments)):
            try:
                segment = self.audio_engine.analyze_segment(
                    audio_data, self.config.sample_rate, 
                    csv_segment['start_time'], csv_segment['speaker']
                )
                forensic_segments.append(segment)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Analyzed {i + 1}/{len(suspicious_segments)} segments")
                    
            except Exception as e:
                logger.error(f"Error analyzing segment {i}: {str(e)}")
                continue
        
        if not forensic_segments:
            logger.error("No segments successfully analyzed")
            return {"status": "analysis_failed", "segments_analyzed": 0}
        
        logger.info(f"Successfully analyzed {len(forensic_segments)} segments")
        
        # Perform clustering analysis
        embeddings = self.clusterer.extract_embeddings(forensic_segments)
        tsne_embeddings, cluster_labels = self.clusterer.cluster_embeddings(embeddings)
        
        # Update segments with clustering information
        for i, segment in enumerate(forensic_segments):
            segment.embedding = embeddings[i]
            segment.cluster_label = cluster_labels[i]
            segment.is_outlier = cluster_labels[i] == -1
        
        # Generate visualizations
        if self.config.create_plots:
            self._generate_visualizations(forensic_segments, tsne_embeddings, cluster_labels)
        
        # Generate forensic dossier
        dossier = self.evidence_generator.generate_forensic_dossier(
            forensic_segments, tsne_embeddings, cluster_labels
        )
        
        # Generate summary report
        summary = self._generate_summary_report(forensic_segments, dossier)
        
        logger.info("Forensic analysis completed successfully")
        
        return {
            "status": "completed",
            "segments_analyzed": len(forensic_segments),
            "suspicious_segments": len([s for s in forensic_segments if s.suspicion_score > 0.5]),
            "forensic_dossier": dossier,
            "summary": summary
        }
    
    def _load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load and validate CSV data."""
        
        try:
            # Try to read with header first
            df = pd.read_csv(csv_path)
            
            # If no header, use standard column names
            if df.columns[0].startswith('2025-'):  # Timestamp pattern
                df = pd.read_csv(csv_path, header=None, names=[
                    'ts_utc', 'file', 'speaker', 'pred_class', 'confidence', 
                    'entropy', 'rms', 'text', 'prob_angry', 'prob_calm', 
                    'prob_fear', 'prob_happy', 'prob_sad', 'prob_surprise'
                ])
            
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    def _identify_suspicious_segments(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify suspicious segments based on CSV analysis."""
        
        suspicious_segments = []
        
        for idx, row in df.iterrows():
            suspicion_score = 0.0
            reasons = []
            
            # Check for speaker match indicators
            speaker = str(row.get('speaker', ''))
            if 'Match:' in speaker or 'Clone' in speaker:
                suspicion_score += 0.3
                reasons.append("Speaker match/clone indicator")
            
            # Check for high entropy (emotional volatility)
            entropy = row.get('entropy', 0)
            if pd.notna(entropy) and entropy > 0.5:
                suspicion_score += 0.2
                reasons.append("High emotional entropy")
            
            # Check for unusual confidence patterns
            confidence = row.get('confidence', 0)
            if pd.notna(confidence) and (confidence > 0.99 or confidence < 0.5):
                suspicion_score += 0.1
                reasons.append("Unusual confidence score")
            
            # Check for RMS volume spikes
            rms = row.get('rms', 0)
            if pd.notna(rms) and rms > 0.02:
                suspicion_score += 0.15
                reasons.append("High RMS volume")
            
            # Check for probability distribution anomalies
            prob_cols = [col for col in df.columns if col.startswith('prob_')]
            if prob_cols:
                probs = [row.get(col, 0) for col in prob_cols if pd.notna(row.get(col, 0))]
                if probs:
                    max_prob = max(probs)
                    if max_prob > 0.95:  # Overly confident predictions
                        suspicion_score += 0.15
                        reasons.append("Overly confident emotion prediction")
            
            # Include segment if suspicion score is high enough
            if suspicion_score >= 0.3:  # Threshold for suspicious segments
                # Extract timestamp and calculate start time
                timestamp = row.get('ts_utc', str(idx))
                try:
                    # Parse timestamp to get seconds from start
                    if isinstance(timestamp, str) and 'T' in timestamp:
                        time_part = timestamp.split('T')[1].split('+')[0]
                        h, m, s = map(float, time_part.split(':'))
                        start_time = h * 3600 + m * 60 + s
                    else:
                        start_time = float(idx)  # Fallback to index
                except:
                    start_time = float(idx)
                
                suspicious_segments.append({
                    'index': idx,
                    'start_time': start_time,
                    'speaker': speaker.split('[')[0].strip() if '[' in speaker else speaker,
                    'suspicion_score': suspicion_score,
                    'reasons': reasons,
                    'row_data': row.to_dict()
                })
        
        return suspicious_segments
    
    def _extract_audio_segments(self, suspicious_segments: List[Dict[str, Any]], 
                               audio_path: str) -> List[np.ndarray]:
        """Extract audio segments from audio file."""
        
        audio_segments = []
        
        try:
            # Load audio file
            if os.path.isfile(audio_path):
                audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
                logger.info(f"Loaded audio file: {audio_path} (duration: {len(audio)/sr:.2f}s)")
            else:
                logger.error(f"Audio file not found: {audio_path}")
                return self._create_synthetic_segments(suspicious_segments)
            
            # Extract segments
            for segment_info in suspicious_segments:
                start_time = segment_info['start_time']
                start_sample = int(start_time * sr)
                end_sample = min(start_sample + int(self.config.segment_duration * sr), len(audio))
                
                if start_sample < len(audio):
                    segment_audio = audio[start_sample:end_sample]
                    
                    # Pad if necessary
                    if len(segment_audio) < int(self.config.segment_duration * sr):
                        padding = int(self.config.segment_duration * sr) - len(segment_audio)
                        segment_audio = np.pad(segment_audio, (0, padding), mode='constant')
                    
                    audio_segments.append(segment_audio)
                else:
                    # Create synthetic segment if start time is beyond audio length
                    synthetic_segment = self._generate_synthetic_audio()
                    audio_segments.append(synthetic_segment)
        
        except Exception as e:
            logger.error(f"Error extracting audio segments: {str(e)}")
            return self._create_synthetic_segments(suspicious_segments)
        
        return audio_segments
    
    def _create_synthetic_segments(self, suspicious_segments: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Create synthetic audio segments for analysis when no audio is available."""
        
        logger.warning("Creating synthetic audio segments for analysis")
        synthetic_segments = []
        
        for segment_info in suspicious_segments:
            # Generate synthetic audio based on segment characteristics
            synthetic_audio = self._generate_synthetic_audio()
            synthetic_segments.append(synthetic_audio)
        
        return synthetic_segments
    
    def _generate_synthetic_audio(self) -> np.ndarray:
        """Generate synthetic audio for testing purposes."""
        
        duration = self.config.segment_duration
        sr = self.config.sample_rate
        samples = int(duration * sr)
        
        # Generate synthetic speech-like signal
        t = np.linspace(0, duration, samples)
        
        # Create harmonic content (fundamental + harmonics)
        f0 = 150 + 50 * np.sin(2 * np.pi * 0.5 * t)  # Varying pitch
        signal = np.sin(2 * np.pi * f0 * t)
        
        # Add harmonics
        for harmonic in [2, 3, 4]:
            signal += 0.3 / harmonic * np.sin(2 * np.pi * f0 * harmonic * t)
        
        # Add formant-like characteristics
        formant_freq = 1000 + 200 * np.sin(2 * np.pi * 0.3 * t)
        signal += 0.2 * np.sin(2 * np.pi * formant_freq * t)
        
        # Add noise
        noise = 0.1 * np.random.randn(samples)
        signal += noise
        
        # Apply envelope
        envelope = 0.5 * (1 + np.sin(np.pi * t / duration))
        signal *= envelope
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        return signal.astype(np.float32)
    
    def _generate_visualizations(self, segments: List[ForensicSegment], 
                                tsne_embeddings: np.ndarray, 
                                cluster_labels: np.ndarray):
        """Generate forensic visualizations."""
        
        logger.info("Generating forensic visualizations...")
        
        # 1. t-SNE Cluster Plot
        cluster_fig = self.clusterer.visualize_clusters(
            segments, tsne_embeddings, cluster_labels,
            output_path=self.output_dir / "forensic_clusters.png"
        )
        plt.close(cluster_fig)
        
        # 2. Forensic Metrics Distribution
        self._plot_forensic_metrics(segments)
        
        # 3. Suspicion Score Analysis
        self._plot_suspicion_analysis(segments)
        
        # 4. Speaker Analysis
        self._plot_speaker_analysis(segments)
        
        logger.info("Visualizations saved to output directory")
    
    def _plot_forensic_metrics(self, segments: List[ForensicSegment]):
        """Plot distribution of forensic metrics."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Forensic Metrics Distribution', fontsize=16)
        
        metrics = [
            ('zcr', 'Zero-Crossing Rate', self.config.zcr_threshold),
            ('spectral_flatness', 'Spectral Flatness', self.config.spectral_flatness_threshold),
            ('jitter', 'Pitch Jitter', self.config.jitter_threshold),
            ('shimmer', 'Amplitude Shimmer', self.config.shimmer_threshold),
            ('breath_events', 'Breath Events', self.config.min_breath_events_per_minute),
            ('suspicion_score', 'Suspicion Score', 0.5)
        ]
        
        for idx, (metric, title, threshold) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            values = [getattr(seg, metric) for seg in segments]
            suspicious_values = [getattr(seg, metric) for seg in segments if seg.suspicion_score > 0.5]
            normal_values = [getattr(seg, metric) for seg in segments if seg.suspicion_score <= 0.5]
            
            # Plot histograms
            ax.hist(normal_values, alpha=0.7, label='Normal', bins=20, color='green')
            ax.hist(suspicious_values, alpha=0.7, label='Suspicious', bins=20, color='red')
            
            # Add threshold line
            if metric != 'suspicion_score':
                ax.axvline(threshold, color='orange', linestyle='--', label='Threshold')
            else:
                ax.axvline(threshold, color='orange', linestyle='--', label='Decision Threshold')
            
            ax.set_title(title)
            ax.set_xlabel(title)
            ax.set_ylabel('Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "forensic_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_suspicion_analysis(self, segments: List[ForensicSegment]):
        """Plot suspicion score analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Suspicion Score Analysis', fontsize=16)
        
        # 1. Suspicion score distribution
        ax = axes[0, 0]
        suspicion_scores = [seg.suspicion_score for seg in segments]
        ax.hist(suspicion_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(0.5, color='red', linestyle='--', label='Suspicion Threshold')
        ax.set_title('Suspicion Score Distribution')
        ax.set_xlabel('Suspicion Score')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Suspicion by speaker
        ax = axes[0, 1]
        speakers = list(set(seg.speaker for seg in segments))
        speaker_suspicion = []
        for speaker in speakers:
            speaker_segments = [seg for seg in segments if seg.speaker == speaker]
            avg_suspicion = np.mean([seg.suspicion_score for seg in speaker_segments])
            speaker_suspicion.append(avg_suspicion)
        
        colors = ['red' if s > 0.5 else 'green' for s in speaker_suspicion]
        ax.bar(speakers, speaker_suspicion, color=colors, alpha=0.7)
        ax.axhline(0.5, color='black', linestyle='--', label='Threshold')
        ax.set_title('Average Suspicion Score by Speaker')
        ax.set_xlabel('Speaker')
        ax.set_ylabel('Average Suspicion Score')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Detection rates
        ax = axes[1, 0]
        detection_types = ['High-Freq Shelf', 'Pitch Artifacts', 'Unnatural Rhythm']
        detection_rates = [
            np.mean([seg.high_freq_shelf for seg in segments]),
            np.mean([seg.pitch_artifacts for seg in segments]),
            np.mean([seg.unnatural_rhythm for seg in segments])
        ]
        
        bars = ax.bar(detection_types, detection_rates, color=['orange', 'purple', 'brown'], alpha=0.7)
        ax.set_title('Forensic Detection Rates')
        ax.set_ylabel('Detection Rate')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, rate in zip(bars, detection_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom')
        
        # 4. Suspicion vs other metrics scatter plot
        ax = axes[1, 1]
        jitter_values = [seg.jitter for seg in segments]
        shimmer_values = [seg.shimmer for seg in segments]
        colors = ['red' if seg.suspicion_score > 0.5 else 'green' for seg in segments]
        
        scatter = ax.scatter(jitter_values, shimmer_values, c=colors, alpha=0.6, s=50)
        ax.set_xlabel('Jitter')
        ax.set_ylabel('Shimmer')
        ax.set_title('Jitter vs Shimmer (colored by suspicion)')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.6, label='Suspicious'),
                          Patch(facecolor='green', alpha=0.6, label='Normal')]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "suspicion_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_speaker_analysis(self, segments: List[ForensicSegment]):
        """Plot speaker-specific forensic analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Speaker Forensic Analysis', fontsize=16)
        
        # Group by speaker
        speakers = list(set(seg.speaker for seg in segments))
        speaker_data = {}
        
        for speaker in speakers:
            speaker_segments = [seg for seg in segments if seg.speaker == speaker]
            speaker_data[speaker] = speaker_segments
        
        # 1. Segment count by speaker
        ax = axes[0, 0]
        segment_counts = [len(speaker_data[speaker]) for speaker in speakers]
        suspicious_counts = [len([s for s in speaker_data[speaker] if s.suspicion_score > 0.5]) 
                           for speaker in speakers]
        
        x = np.arange(len(speakers))
        width = 0.35
        
        ax.bar(x - width/2, segment_counts, width, label='Total', alpha=0.7, color='blue')
        ax.bar(x + width/2, suspicious_counts, width, label='Suspicious', alpha=0.7, color='red')
        
        ax.set_xlabel('Speaker')
        ax.set_ylabel('Segment Count')
        ax.set_title('Segments by Speaker')
        ax.set_xticks(x)
        ax.set_xticklabels(speakers, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Average metrics by speaker
        ax = axes[0, 1]
        metrics_to_plot = ['zcr', 'spectral_flatness', 'jitter', 'shimmer']
        
        x = np.arange(len(speakers))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            values = [np.mean([getattr(seg, metric) for seg in speaker_data[speaker]]) 
                     for speaker in speakers]
            ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), alpha=0.7)
        
        ax.set_xlabel('Speaker')
        ax.set_ylabel('Average Value')
        ax.set_title('Average Forensic Metrics by Speaker')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(speakers, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Suspicion rate by speaker
        ax = axes[1, 0]
        suspicion_rates = [len([s for s in speaker_data[speaker] if s.suspicion_score > 0.5]) / len(speaker_data[speaker]) 
                          for speaker in speakers]
        
        colors = ['red' if rate > 0.5 else 'orange' if rate > 0.2 else 'green' for rate in suspicion_rates]
        bars = ax.bar(speakers, suspicion_rates, color=colors, alpha=0.7)
        ax.axhline(0.5, color='black', linestyle='--', label='High Risk')
        ax.axhline(0.2, color='orange', linestyle='--', label='Medium Risk')
        ax.set_xlabel('Speaker')
        ax.set_ylabel('Suspicion Rate')
        ax.set_title('Suspicion Rate by Speaker')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, rate in zip(bars, suspicion_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom')
        
        # 4. Speaker risk classification
        ax = axes[1, 1]
        risk_categories = {'Low Risk': 0, 'Medium Risk': 0, 'High Risk': 0}
        
        for rate in suspicion_rates:
            if rate > 0.5:
                risk_categories['High Risk'] += 1
            elif rate > 0.2:
                risk_categories['Medium Risk'] += 1
            else:
                risk_categories['Low Risk'] += 1
        
        colors = ['green', 'orange', 'red']
        ax.pie(risk_categories.values(), labels=risk_categories.keys(), colors=colors, 
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Speaker Risk Classification')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "speaker_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, segments: List[ForensicSegment], 
                               dossier: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report of forensic analysis."""
        
        total_segments = len(segments)
        suspicious_segments = [s for s in segments if s.suspicion_score > 0.5]
        
        summary = {
            "analysis_overview": {
                "total_segments_analyzed": total_segments,
                "suspicious_segments_found": len(suspicious_segments),
                "overall_suspicion_rate": len(suspicious_segments) / total_segments if total_segments > 0 else 0,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "key_findings": {
                "primary_suspicious_speakers": list(set(s.speaker for s in suspicious_segments)),
                "most_common_indicators": self._get_most_common_indicators(suspicious_segments),
                "average_suspicion_score": np.mean([s.suspicion_score for s in segments]),
                "max_suspicion_score": np.max([s.suspicion_score for s in segments])
            },
            "forensic_metrics_summary": dossier.get("forensic_metrics", {}),
            "recommendations": dossier.get("recommendations", []),
            "risk_assessment": self._assess_overall_risk(segments)
        }
        
        # Save summary report
        summary_path = self.output_dir / "forensic_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to {summary_path}")
        
        return summary
    
    def _get_most_common_indicators(self, suspicious_segments: List[ForensicSegment]) -> Dict[str, float]:
        """Get most common forensic indicators in suspicious segments."""
        
        if not suspicious_segments:
            return {}
        
        indicators = {
            "high_freq_shelf": np.mean([s.high_freq_shelf for s in suspicious_segments]),
            "pitch_artifacts": np.mean([s.pitch_artifacts for s in suspicious_segments]),
            "unnatural_rhythm": np.mean([s.unnatural_rhythm for s in suspicious_segments])
        }
        
        # Sort by frequency
        sorted_indicators = dict(sorted(indicators.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_indicators
    
    def _assess_overall_risk(self, segments: List[ForensicSegment]) -> Dict[str, Any]:
        """Assess overall risk level of the audio analysis."""
        
        total_segments = len(segments)
        suspicious_segments = [s for s in segments if s.suspicion_score > 0.5]
        suspicion_rate = len(suspicious_segments) / total_segments if total_segments > 0 else 0
        
        # Determine risk level
        if suspicion_rate > 0.5:
            risk_level = "HIGH"
            risk_description = "More than 50% of segments show suspicious characteristics"
        elif suspicion_rate > 0.2:
            risk_level = "MEDIUM"
            risk_description = "Significant portion of segments show suspicious patterns"
        elif suspicion_rate > 0.05:
            risk_level = "LOW"
            risk_description = "Some suspicious segments detected, but overall risk is low"
        else:
            risk_level = "MINIMAL"
            risk_description = "Very few or no suspicious segments detected"
        
        return {
            "risk_level": risk_level,
            "risk_description": risk_description,
            "suspicion_rate": suspicion_rate,
            "confidence": "high" if suspicion_rate > 0.3 or suspicion_rate < 0.05 else "medium"
        }


def main():
    """Main function for running the forensic pipeline."""
    
    parser = argparse.ArgumentParser(description="Forensic Deepfake Detection Pipeline")
    parser.add_argument("--csv", required=True, help="Path to analysis CSV file")
    parser.add_argument("--audio", help="Path to audio file or directory")
    parser.add_argument("--output", default="forensic_output", help="Output directory")
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate")
    parser.add_argument("--suspicion-threshold", type=float, default=0.5,
                       help="Threshold for identifying suspicious segments")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--max-segments", type=int, default=None,
                       help="Maximum number of segments to analyze (for testing)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ForensicConfig()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Override with command line arguments
    config.sample_rate = args.sample_rate
    config.output_dir = args.output
    config.create_plots = not args.no_plots
    
    # Initialize pipeline
    pipeline = ForensicPipeline(config)
    
    # Run analysis
    try:
        results = pipeline.analyze_csv_data(args.csv, args.audio, args.max_segments)
        
        if results["status"] == "completed":
            print(f"\n{'='*60}")
            print("FORENSIC ANALYSIS COMPLETED")
            print(f"{'='*60}")
            print(f"Total segments analyzed: {results['segments_analyzed']}")
            print(f"Suspicious segments found: {results['suspicious_segments']}")
            print(f"Overall suspicion rate: {results['summary']['analysis_overview']['overall_suspicion_rate']:.1%}")
            print(f"Risk level: {results['summary']['risk_assessment']['risk_level']}")
            print(f"Output directory: {config.output_dir}")
            print(f"\nKey findings:")
            for finding in results['summary']['key_findings']['primary_suspicious_speakers']:
                print(f"  - Suspicious speaker: {finding}")
            print(f"\nRecommendations:")
            for rec in results['summary']['recommendations']:
                print(f"  - {rec}")
            print(f"{'='*60}")
        else:
            print(f"Analysis status: {results['status']}")
            
    except Exception as e:
        logger.error(f"Error running forensic analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
