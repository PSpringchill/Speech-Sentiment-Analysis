import os
import argparse
import subprocess
from pathlib import Path

def parse_rttm(rttm_path):
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                # SPEAKER <file> <channel> <start> <duration> <ortho> <lookahead> <speaker_id>
                start = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]
                segments.append({
                    'start': start,
                    'duration': duration,
                    'speaker': speaker_id
                })
    return segments

def segment_audio(audio_path, rttm_path, output_dir):
    """
    Segments the audio file into speaker-specific folders based on RTTM.
    Uses ffmpeg for lossless cutting.
    """
    segments = parse_rttm(rttm_path)
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Segmenting {audio_path.name} into {output_dir}...")
    
    speaker_counts = {}
    
    for i, seg in enumerate(segments):
        speaker = seg['speaker']
        speaker_dir = output_dir / speaker
        os.makedirs(speaker_dir, exist_ok=True)
        
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        output_file = speaker_dir / f"segment_{speaker_counts[speaker]:03d}_{seg['start']:.2f}.wav"
        
        # ffmpeg command for precise cutting
        # -ss: start time, -t: duration, -i: input
        cmd = [
            'ffmpeg',
            '-y', # overwrite
            '-ss', str(seg['start']),
            '-t', str(seg['duration']),
            '-i', str(audio_path),
            '-acodec', 'copy', # lossless copy
            '-loglevel', 'error',
            str(output_file)
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error cutting segment {i}: {e}")
            
    print(f"Done. Created folders for {len(speaker_counts)} speakers.")
    for spk, count in speaker_counts.items():
        print(f" - {spk}: {count} segments")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop audio into speaker segments for Deep Analysis")
    parser.add_argument("--audio", required=True, help="Path to original audio file")
    parser.add_argument("--rttm", required=True, help="Path to RTTM diarization output")
    parser.add_argument("--out-dir", default="speaker_segments", help="Output directory for segments")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
    elif not os.path.exists(args.rttm):
        print(f"Error: RTTM file not found: {args.rttm}")
    else:
        segment_audio(args.audio, args.rttm, args.out_dir)
