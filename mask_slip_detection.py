#!/usr/bin/env python3
"""
Mask Slip Detection System
Forensic Audio Analysis Tool for Voice Clone Anomaly Detection

This script identifies and extracts audio segments where voice cloning models
show signs of instability or "mask slip" - moments where the real voice
leaks through the synthetic overlay.

Key Features:
- Detects speaker labels indicating cloning instability
- Cross-references with volume spikes for anomaly detection
- Exports forensic audio clips with padding for transition analysis
- Applies forensic filtering to isolate human vocal frequencies
"""

import os
import json
import pandas as pd
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mask_slip_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MaskSlipDetector:
    """
    Forensic audio analysis tool for detecting voice clone anomalies
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the detector with configuration
        
        Args:
            config: Dictionary containing detection parameters
        """
        self.config = config
        self.input_audio = Path(config['input_audio'])
        self.output_dir = Path(config['output_directory'])
        self.padding_seconds = config['padding_seconds']
        self.target_flags = config['target_flags']
        self.min_volatility_threshold = config.get('min_volatility_threshold', 0.5)
        self.min_volume_rms = config.get('min_volume_rms', 0.04)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.suspicious_segments = []
        self.analysis_summary = {}
        
        logger.info(f"Initialized Mask Slip Detector")
        logger.info(f"Input audio: {self.input_audio}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Target flags: {self.target_flags}")

    def load_diarization_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and prepare diarization data from CSV
        
        Args:
            csv_path: Path to the CSV file containing analysis results
            
        Returns:
            DataFrame with processed diarization data
        """
        try:
            # Try to read with header first
            try:
                df = pd.read_csv(csv_path)
            except:
                # If that fails, try reading without header and assign column names
                df = pd.read_csv(csv_path, header=None, names=[
                    "ts_utc", "file", "speaker", "pred_class", "confidence", 
                    "entropy", "rms", "text", "prob_happy", "prob_angry", 
                    "prob_fear", "prob_calm", "prob_sad", "prob_surprise"
                ])
            
            # Convert timestamp to datetime - handle extra spaces and ISO format
            df['ts_utc'] = pd.to_datetime(df['ts_utc'].str.strip(), format='ISO8601')
            
            # Calculate additional metrics
            df['volatility_index'] = self._calculate_volatility(df)
            df['time_seconds'] = self._timestamp_to_seconds(df['ts_utc'])
            
            logger.info(f"Loaded {len(df)} records from {csv_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading diarization data: {e}")
            raise

    def _calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate emotional volatility based on probability distribution entropy
        
        Args:
            df: DataFrame with emotion probabilities
            
        Returns:
            Series with volatility scores
        """
        prob_cols = [col for col in df.columns if col.startswith('prob_')]
        
        # Calculate Shannon entropy as volatility measure
        def entropy(row):
            probs = row[prob_cols].values
            # Add small epsilon to avoid log(0)
            probs = probs + 1e-10
            probs = probs / probs.sum()  # Normalize
            return -np.sum(probs * np.log2(probs))
        
        return df.apply(entropy, axis=1)

    def _timestamp_to_seconds(self, timestamps: pd.Series) -> pd.Series:
        """
        Convert timestamps to seconds from start of recording
        
        Args:
            timestamps: Series of datetime timestamps
            
        Returns:
            Series of seconds from first timestamp
        """
        if len(timestamps) == 0:
            return pd.Series([], dtype=float)
        
        start_time = timestamps.min()
        return (timestamps - start_time).dt.total_seconds()

    def detect_suspicious_segments(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify segments with potential voice clone anomalies
        
        Args:
            df: DataFrame with diarization data
            
        Returns:
            List of suspicious segment dictionaries
        """
        suspicious_segments = []
        
        logger.info("Analyzing segments for voice clone anomalies...")
        
        for idx, row in df.iterrows():
            # CHECK 1: Label Matching
            is_suspect_label = any(flag in str(row['speaker']) for flag in self.target_flags)
            
            # CHECK 2: Volume Cross-Reference
            avg_volume = row['rms']
            is_loud = avg_volume > self.min_volume_rms
            
            # CHECK 3: Volatility Threshold
            high_volatility = row['volatility_index'] > self.min_volatility_threshold
            
            # CHECK 4: Unknown speaker with high volume (potential artifact)
            is_unknown_loud = ('Unknown' in str(row['speaker']) and is_loud)
            
            # LOGIC: We want labeled failures OR high-volume unknown artifacts
            if is_suspect_label or is_unknown_loud or (is_loud and high_volatility):
                
                # Calculate segment boundaries
                segment_start = row['time_seconds']
                segment_end = segment_start + 2.0  # Assume 2-second segments
                
                # Apply padding
                clip_start = max(0, segment_start - self.padding_seconds)
                clip_end = segment_end + self.padding_seconds
                
                # Determine detection reason
                reasons = []
                if is_suspect_label:
                    reasons.append("Label Match")
                if is_unknown_loud:
                    reasons.append("Unknown Speaker + High Volume")
                if is_loud and high_volatility:
                    reasons.append("High Volume + High Volatility")
                
                # Create extraction job
                job = {
                    'start': clip_start,
                    'end': clip_end,
                    'label': row['speaker'],
                    'confidence': row['confidence'],
                    'emotion': row['pred_class'],
                    'rms': row['rms'],
                    'volatility': row['volatility_index'],
                    'reason': ', '.join(reasons),
                    'original_timestamp': row['ts_utc'],
                    'text': row.get('text', ''),
                    'csv_index': idx
                }
                
                suspicious_segments.append(job)
                
                logger.debug(f"Suspicious segment detected at {clip_start:.2f}s: {reasons}")
        
        logger.info(f"Found {len(suspicious_segments)} suspicious segments")
        return suspicious_segments

    def extract_audio_segments(self, segments: List[Dict]) -> List[str]:
        """
        Extract audio clips using FFmpeg without re-encoding
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of paths to extracted audio files
        """
        extracted_files = []
        
        if not self.input_audio.exists():
            logger.error(f"Input audio file not found: {self.input_audio}")
            return extracted_files
        
        logger.info(f"Extracting {len(segments)} audio segments...")
        
        for i, segment in enumerate(segments):
            try:
                # Generate output filename
                safe_label = "".join(c for c in segment['label'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"evidence_{i:03d}_{segment['start']:.2f}s_{safe_label}.wav"
                output_path = self.output_dir / filename
                
                # Construct FFmpeg command
                cmd = [
                    'ffmpeg',
                    '-i', str(self.input_audio),
                    '-ss', str(segment['start']),
                    '-to', str(segment['end']),
                    '-c', 'copy',  # Copy without re-encoding to preserve artifacts
                    '-y',  # Overwrite output files
                    str(output_path)
                ]
                
                # Execute FFmpeg
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    extracted_files.append(str(output_path))
                    logger.info(f"Extracted: {filename} ({segment['reason']})")
                else:
                    logger.error(f"FFmpeg error for {filename}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout extracting segment {i}")
            except Exception as e:
                logger.error(f"Error extracting segment {i}: {e}")
        
        logger.info(f"Successfully extracted {len(extracted_files)} audio segments")
        return extracted_files

    def apply_forensic_filter(self, audio_path: str) -> str:
        """
        Apply forensic bandpass filter to isolate human vocal frequencies
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to filtered audio file
        """
        try:
            input_path = Path(audio_path)
            output_path = input_path.parent / f"filtered_{input_path.name}"
            
            # Construct FFmpeg filter command
            # High-pass at 300Hz (remove rumble)
            # Low-pass at 3.4kHz (remove metallic AI hiss)
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-af', 'highpass=f=300,lowpass=f=3400',
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Applied forensic filter: {output_path.name}")
                return str(output_path)
            else:
                logger.error(f"Filter error for {input_path.name}: {result.stderr}")
                return audio_path
                
        except Exception as e:
            logger.error(f"Error applying filter to {audio_path}: {e}")
            return audio_path

    def generate_analysis_report(self, segments: List[Dict], extracted_files: List[str]) -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            segments: List of detected segments
            extracted_files: List of extracted file paths
            
        Returns:
            Path to generated report file
        """
        report_path = self.output_dir / "mask_slip_analysis_report.json"
        
        # Calculate statistics
        total_segments = len(segments)
        reasons_count = {}
        emotion_distribution = {}
        confidence_stats = {
            'mean': np.mean([s['confidence'] for s in segments]),
            'min': np.min([s['confidence'] for s in segments]),
            'max': np.max([s['confidence'] for s in segments])
        }
        
        for segment in segments:
            # Count reasons
            for reason in segment['reason'].split(', '):
                reasons_count[reason] = reasons_count.get(reason, 0) + 1
            
            # Count emotions
            emotion = segment['emotion']
            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
        
        # Create report
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'input_audio': str(self.input_audio),
                'total_segments_analyzed': total_segments,
                'suspicious_segments_found': len(segments),
                'audio_clips_extracted': len(extracted_files),
                'configuration': self.config
            },
            'detection_statistics': {
                'detection_reasons': reasons_count,
                'emotion_distribution': emotion_distribution,
                'confidence_statistics': confidence_stats,
                'average_volatility': np.mean([s['volatility'] for s in segments]),
                'average_rms': np.mean([s['rms'] for s in segments])
            },
            'suspicious_segments': segments,
            'extracted_files': extracted_files,
            'forensic_recommendations': self._generate_recommendations(segments)
        }
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis report saved: {report_path}")
        return str(report_path)

    def _generate_recommendations(self, segments: List[Dict]) -> List[str]:
        """
        Generate forensic analysis recommendations based on detected patterns
        
        Args:
            segments: List of suspicious segments
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Analyze patterns
        high_volatility_count = sum(1 for s in segments if s['volatility'] > 0.7)
        low_confidence_count = sum(1 for s in segments if s['confidence'] < 0.6)
        
        if high_volatility_count > len(segments) * 0.5:
            recommendations.append(
                "High volatility detected in >50% of segments. "
                "Consider reviewing model stability and training data."
            )
        
        if low_confidence_count > len(segments) * 0.3:
            recommendations.append(
                "Low confidence detected in >30% of segments. "
                "Model may be struggling with speaker characteristics."
            )
        
        recommendations.extend([
            "Listen for 'double-voice' effect in extracted clips",
            "Apply forensic filters to isolate human vocal frequencies",
            "Cross-reference with original recording for comparison",
            "Consider spectral analysis for frequency domain anomalies",
            "Review text transcription for semantic inconsistencies"
        ])
        
        return recommendations

    def run_analysis(self, csv_path: str, apply_filters: bool = True) -> Dict:
        """
        Run complete mask slip detection analysis
        
        Args:
            csv_path: Path to CSV analysis data
            apply_filters: Whether to apply forensic filters
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Load data
            df = self.load_diarization_data(csv_path)
            
            # Detect suspicious segments
            self.suspicious_segments = self.detect_suspicious_segments(df)
            
            # Extract audio segments
            extracted_files = self.extract_audio_segments(self.suspicious_segments)
            
            # Apply forensic filters if requested
            if apply_filters:
                logger.info("Applying forensic filters...")
                filtered_files = []
                for audio_file in extracted_files:
                    filtered_file = self.apply_forensic_filter(audio_file)
                    filtered_files.append(filtered_file)
            else:
                filtered_files = extracted_files
            
            # Generate analysis report
            report_path = self.generate_analysis_report(self.suspicious_segments, filtered_files)
            
            # Prepare results
            results = {
                'suspicious_segments': self.suspicious_segments,
                'extracted_files': extracted_files,
                'filtered_files': filtered_files,
                'report_path': report_path,
                'statistics': {
                    'total_segments': len(df),
                    'suspicious_count': len(self.suspicious_segments),
                    'extraction_success_rate': len(extracted_files) / len(self.suspicious_segments) if self.suspicious_segments else 0
                }
            }
            
            logger.info("Mask slip detection analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def create_default_config() -> Dict:
    """
    Create default configuration for mask slip detection
    
    Returns:
        Default configuration dictionary
    """
    return {
        'input_audio': 'source_recording.wav',
        'output_directory': './forensic_clips/',
        'padding_seconds': 0.5,
        'target_flags': [
            'Match: Low - Tone Change?',
            'Match: Medium - Check for Clone',
            'Match: High',  # Include high matches for completeness
            'Unknown'     # Unknown speakers
        ],
        'min_volatility_threshold': 0.5,
        'min_volume_rms': 0.04
    }


def main():
    """
    Main execution function with command-line interface
    """
    parser = argparse.ArgumentParser(
        description='Mask Slip Detection - Forensic Voice Clone Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mask_slip_detection.py --csv analysis_log.csv --audio recording.wav
  python mask_slip_detection.py --csv analysis_log.csv --audio recording.wav --no-filters
  python mask_slip_detection.py --csv analysis_log.csv --audio recording.wav --padding 1.0
        """
    )
    
    parser.add_argument('--csv', required=True, help='Path to analysis CSV file')
    parser.add_argument('--audio', required=True, help='Path to source audio file')
    parser.add_argument('--output', default='./forensic_clips/', help='Output directory (default: ./forensic_clips/)')
    parser.add_argument('--padding', type=float, default=0.5, help='Padding seconds before/after segments (default: 0.5)')
    parser.add_argument('--no-filters', action='store_true', help='Skip forensic filtering')
    parser.add_argument('--volatility', type=float, default=0.5, help='Min volatility threshold (default: 0.5)')
    parser.add_argument('--volume', type=float, default=0.04, help='Min RMS volume threshold (default: 0.04)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_default_config()
    config.update({
        'input_audio': args.audio,
        'output_directory': args.output,
        'padding_seconds': args.padding,
        'min_volatility_threshold': args.volatility,
        'min_volume_rms': args.volume
    })
    
    # Validate inputs
    if not Path(args.csv).exists():
        logger.error(f"CSV file not found: {args.csv}")
        return 1
    
    if not Path(args.audio).exists():
        logger.error(f"Audio file not found: {args.audio}")
        return 1
    
    try:
        # Run analysis
        detector = MaskSlipDetector(config)
        results = detector.run_analysis(args.csv, apply_filters=not args.no_filters)
        
        # Print summary
        print("\n" + "="*60)
        print("MASK SLIP DETECTION ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total segments analyzed: {results['statistics']['total_segments']}")
        print(f"Suspicious segments found: {results['statistics']['suspicious_count']}")
        print(f"Audio clips extracted: {len(results['extracted_files'])}")
        print(f"Success rate: {results['statistics']['extraction_success_rate']:.1%}")
        print(f"Output directory: {config['output_directory']}")
        print(f"Analysis report: {results['report_path']}")
        print("="*60)
        
        # Print top 5 suspicious segments
        print("\nTOP 5 SUSPICIOUS SEGMENTS:")
        for i, segment in enumerate(results['suspicious_segments'][:5]):
            print(f"{i+1}. {segment['start']:.2f}s - {segment['label']}")
            print(f"   Reason: {segment['reason']}")
            print(f"   Emotion: {segment['emotion']} (Confidence: {segment['confidence']:.2f})")
            print(f"   Volatility: {segment['volatility']:.2f}, RMS: {segment['rms']:.3f}")
            print()
        
        print("Listen for 'double-voice' effects in the extracted clips.")
        print("Check the analysis report for detailed findings.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
