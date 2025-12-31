# CSV Data Format Guide

## Forensic Analysis CSV Structure

The forensic pipeline expects CSV files with the following column structure:

### Required Columns

| Column | Description | Example |
|--------|-------------|---------|
| `ts_utc` | Timestamp in UTC format | `2025-12-31T09:00:23+00:00` |
| `file` | Source file identifier | `mic:0`, `audio_file.wav` |
| `speaker` | Speaker identifier | `Speaker_0`, `Speaker_1` |
| `pred_class` | Predicted emotion class | `happy`, `angry`, `fear`, `calm`, `sad` |
| `confidence` | Prediction confidence (0-1) | `0.94606036` |
| `entropy` | Prediction entropy | `0.22745989` |
| `rms` | Audio RMS volume | `0.01597834` |
| `text` | Transcribed text | `ไกล ใกล้เคียง ที่สุด น่ะ นัก` |
| `prob_happy` | Happy probability | `0.00000000` |
| `prob_angry` | Angry probability | `0.04855958` |
| `prob_fear` | Fear probability | `0.00000000` |
| `prob_calm` | Calm probability | `0.94606036` |
| `prob_sad` | Sad probability | `0.00538001` |
| `prob_surprise` | Surprise probability | `0.00000000` |

### Sample CSV Row

```csv
ts_utc,file,speaker,pred_class,confidence,entropy,rms,text,prob_happy,prob_angry,prob_fear,prob_calm,prob_sad,prob_surprise
2025-12-31T09:00:23+00:00,mic:0,Speaker_0,happy,0.94606036,0.22745989,0.01597834,ไกล ใกล้เคียง ที่สุด น่ะ นัก,0.00000000,0.04855958,0.00000000,0.94606036,0.00538001,0.00000000
```

## Data Requirements

### 1. Timestamp Format
- Must be in ISO 8601 UTC format: `YYYY-MM-DDTHH:MM:SS+00:00`
- Used for chronological analysis and segment ordering

### 2. Speaker Identification
- Consistent naming: `Speaker_0`, `Speaker_1`, etc.
- Can include clone detection warnings: `Speaker_0 [Match: Medium - Check for Clone]`

### 3. Probability Values
- All probability columns must sum to 1.0 (or very close)
- Values should be between 0.0 and 1.0
- Used for entropy calculation and uncertainty analysis

### 4. Audio Features
- `rms`: Root Mean Square audio volume
- `entropy`: Prediction uncertainty (higher = more uncertain)
- `confidence`: Model confidence in prediction

## Forensic Analysis Integration

The forensic pipeline uses this data to:

1. **Identify Suspicious Segments**
   - High entropy (> 1.0) indicates uncertainty
   - Low confidence (< 0.7) suggests potential issues
   - Unusual emotion transitions

2. **Speaker Analysis**
   - Groups segments by speaker
   - Detects voice cloning patterns
   - Analyzes speaker-specific anomalies

3. **Temporal Patterns**
   - Uses timestamps for sequence analysis
   - Detects unnatural rhythm patterns
   - Identifies missing breath events

## Creating Test Data

Use the provided `test_sample.csv` as a template:

```bash
# Run forensic analysis on test data
python forensic_pipeline.py --csv test_sample.csv --max-segments 5

# Or use the full analysis log
python forensic_pipeline.py --csv analysis_log.csv --max-segments 10
```

## Common Issues

1. **Missing Columns**: Ensure all 14 columns are present
2. **Invalid Timestamps**: Use proper ISO 8601 format
3. **Probability Mismatch**: Check that probabilities sum to ~1.0
4. **Encoding Issues**: Save as UTF-8 for Thai text support

## Output Integration

The forensic pipeline will:
- Extract suspicious segments based on this data
- Generate synthetic audio if no audio files provided
- Apply RPCA Audio X-Ray analysis
- Produce comprehensive forensic reports
