# CSV Management Guide

## Overview
The CSV Manager (`csv_manager.py`) provides comprehensive lifecycle management for analysis data, allowing you to archive, reset, and maintain CSV files for different analysis sessions.

## Key Features

### 🗃️ **Archive Management**
- Automatic timestamped archiving
- Custom archive naming
- Metadata preservation
- Restore from archive

### 🔄 **CSV Reset**
- Clean slate for new analysis
- Automatic backup before reset
- Proper header structure
- Configurable emotion classes

### 📊 **Data Operations**
- Statistics and insights
- File merging capabilities
- Cleanup of old archives
- Detailed archive listing

## Usage Examples

### **Basic Operations**
```bash
# View current CSV statistics
python3 csv_manager.py stats

# Archive current data with custom name
python3 csv_manager.py archive --name "important_session"

# Reset for new analysis (with automatic backup)
python3 csv_manager.py reset

# Reset without backup (use with caution)
python3 csv_manager.py reset --no-backup
```

### **Archive Management**
```bash
# List all archived files
python3 csv_manager.py list

# List with detailed information
python3 csv_manager.py list --detailed

# Restore from archive
python3 csv_manager.py restore analysis_log_20231231_150000.csv

# Clean up archives older than 7 days
python3 csv_manager.py cleanup --days 7
```

### **Advanced Operations**
```bash
# Merge multiple CSV files
python3 csv_manager.py merge file1.csv file2.csv --output merged_data.csv

# Reset with custom emotion classes
python3 csv_manager.py reset --classes happy sad angry fear calm surprise
```

## File Structure

### **Current CSV** (`analysis_log.csv`)
- Active analysis data
- Real-time logging target
- Dashboard data source

### **Archive Directory** (`csv_archive/`)
```
csv_archive/
├── session_2025_12_31_speech_analysis_20251231_155534.csv
├── session_2025_12_31_speech_analysis_20251231_155534.json  # Metadata
├── analysis_log_20251231_155605.csv
└── analysis_log_20251231_155605.json                          # Metadata
```

### **Metadata Files**
Each archived CSV has a corresponding JSON metadata file containing:
- Original file path
- Archive timestamp
- File size and record count
- Additional context

## Integration with Analysis Pipeline

### **Before Starting New Analysis**
```bash
# 1. Check current data
python3 csv_manager.py stats

# 2. Archive current session
python3 csv_manager.py archive --name "previous_session"

# 3. Reset for new analysis
python3 csv_manager.py reset

# 4. Start new analysis
python3 realtime_analysis.py --mode mic --log-csv analysis_log.csv
```

### **After Analysis Completion**
```bash
# 1. Archive completed analysis
python3 csv_manager.py archive --name "completed_experiment"

# 2. Run forensic analysis on archived data
python3 mask_slip_detection.py --csv csv_archive/completed_experiment_*.csv --audio recording.wav

# 3. Reset for next session
python3 csv_manager.py reset
```

## Best Practices

### **Data Management**
1. **Always archive before reset** - preserves valuable analysis data
2. **Use descriptive archive names** - easier to identify sessions later
3. **Regular cleanup** - prevents archive directory from growing too large
4. **Check statistics** - verify data integrity before operations

### **Analysis Workflow**
1. Archive previous session with meaningful name
2. Reset CSV for clean start
3. Run real-time analysis
4. Review results in dashboard
5. Archive completed analysis
6. Run forensic tools if needed

### **Backup Strategy**
- Automatic backup on reset (default behavior)
- Manual archiving for important sessions
- Regular cleanup of old archives
- Consider external backup for critical data

## Error Handling

### **Common Issues**
- **File not found**: Check file paths and permissions
- **CSV format errors**: Verify CSV structure and headers
- **Archive corruption**: Check archive directory integrity

### **Recovery**
```bash
# Restore from most recent archive
python3 csv_manager.py restore $(ls csv_archive/*.csv | tail -1 | xargs basename)

# Check archive integrity
python3 csv_manager.py list --detailed
```

## Integration with Other Tools

### **Mask Slip Detection**
```bash
# Run forensic analysis on archived data
python3 mask_slip_detection.py --csv csv_archive/suspicious_session_*.csv --audio source.wav
```

### **Dashboard Visualization**
```bash
# Start dashboard with current data
streamlit run visualize_report.py

# Dashboard automatically reads from analysis_log.csv
```

### **Research Analysis**
```bash
# Generate t-SNE plots from archived data
python3 visualize_paper.py --input csv_archive/research_session_*.csv
```

## Troubleshooting

### **Reset Issues**
```bash
# Force reset without backup (emergency only)
python3 csv_manager.py reset --no-backup
```

### **Archive Issues**
```bash
# Manually check archive directory
ls -la csv_archive/

# Verify archive integrity
python3 csv_manager.py list --detailed
```

### **Data Merging**
```bash
# Merge multiple sessions for comprehensive analysis
python3 csv_manager.py merge csv_archive/session1.csv csv_archive/session2.csv --output combined_analysis.csv
```

## Automation Examples

### **Session Management Script**
```bash
#!/bin/bash
# auto_session.sh

SESSION_NAME="session_$(date +%Y%m%d_%H%M%S)"

echo "Starting new analysis session: $SESSION_NAME"

# Archive previous data
python3 csv_manager.py archive --name "$SESSION_NAME"

# Reset for new analysis
python3 csv_manager.py reset

# Start analysis
python3 realtime_analysis.py --mode mic --log-csv analysis_log.csv

echo "Session complete. Data archived as: $SESSION_NAME"
```

### **Daily Cleanup**
```bash
#!/bin/bash
# daily_cleanup.sh

# Clean up archives older than 30 days
python3 csv_manager.py cleanup --days 30

# Archive current data if it exists
if [ -f "analysis_log.csv" ] && [ -s "analysis_log.csv" ]; then
    python3 csv_manager.py archive --name "daily_backup_$(date +%Y%m%d)"
    python3 csv_manager.py reset
fi
```

This CSV management system ensures your analysis data is properly organized, backed up, and ready for both real-time analysis and forensic investigation.
