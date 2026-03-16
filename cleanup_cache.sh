#!/bin/bash
# Clean up all temporary files and cache for PubSpeaker

echo "================================================"
echo "PubSpeaker Cache & Temp Files Cleanup"
echo "================================================"

# 1. Application temp directory
echo ""
echo "1. Cleaning application temp files..."
if [ -d "/tmp/capstone_local" ]; then
    echo "   Removing /tmp/capstone_local/"
    rm -rf /tmp/capstone_local/*
    echo "   ✓ Cleaned: /tmp/capstone_local/"
else
    echo "   - Directory not found: /tmp/capstone_local"
fi

# 2. MFA temp and cache files
echo ""
echo "2. Cleaning MFA temp files..."
if [ -d "$HOME/Documents/MFA" ]; then
    echo "   Removing $HOME/Documents/MFA/"
    rm -rf "$HOME/Documents/MFA"/*
    echo "   ✓ Cleaned: $HOME/Documents/MFA/"
else
    echo "   - Directory not found: $HOME/Documents/MFA"
fi

# 3. Python cache files
echo ""
echo "3. Cleaning Python cache files..."
cd "/Users/kenji/Web Development Projects/PubSpeaker" || exit
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "   ✓ Cleaned: __pycache__ directories"

# 4. Log files (if any)
echo ""
echo "4. Cleaning log files..."
if [ -d "logs" ]; then
    rm -rf logs/*
    echo "   ✓ Cleaned: logs/"
else
    echo "   - No logs directory"
fi

# 5. Show what remains
echo ""
echo "================================================"
echo "Cleanup Summary"
echo "================================================"

# Check sizes
if [ -d "/tmp/capstone_local" ]; then
    TEMP_SIZE=$(du -sh /tmp/capstone_local 2>/dev/null | cut -f1)
    echo "  /tmp/capstone_local: $TEMP_SIZE"
fi

if [ -d "$HOME/Documents/MFA" ]; then
    MFA_SIZE=$(du -sh "$HOME/Documents/MFA" 2>/dev/null | cut -f1)
    echo "  MFA temp files: $MFA_SIZE"
fi

echo ""
echo "✅ Cleanup complete!"
echo "================================================"
