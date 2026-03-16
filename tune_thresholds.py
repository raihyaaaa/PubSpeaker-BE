#!/usr/bin/env python3
"""
Tune pronunciation detection thresholds to find the right sensitivity.

This script helps you adjust the thresholds in config.py to get the right
balance between catching real mispronunciations and avoiding false positives.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("Pronunciation Detection Threshold Tuner")
print("="*80)

print("\nCurrent thresholds in config.py:")
print("-"*80)

from config import (
    PRONUNCIATION_DEVIATION_THRESHOLD,
    PRONUNCIATION_SEVERITY_MINOR,
    PRONUNCIATION_SEVERITY_MODERATE,
    PRONUNCIATION_SEVERITY_NOTABLE
)

print(f"PRONUNCIATION_DEVIATION_THRESHOLD = {PRONUNCIATION_DEVIATION_THRESHOLD}")
print(f"  → Reports deviations when similarity < {PRONUNCIATION_DEVIATION_THRESHOLD}")
print(f"\nSeverity Classification:")
print(f"  MINOR:    similarity >= {PRONUNCIATION_SEVERITY_MINOR}")
print(f"  MODERATE: {PRONUNCIATION_SEVERITY_MODERATE} <= similarity < {PRONUNCIATION_SEVERITY_MINOR}")
print(f"  NOTABLE:  similarity < {PRONUNCIATION_SEVERITY_NOTABLE}")

print("\n" + "="*80)
print("Understanding Thresholds")
print("="*80)

print("""
PRONUNCIATION_DEVIATION_THRESHOLD (currently {:.2f}):
  • Higher value (0.85-0.95) = MORE sensitive
    → Catches more subtle pronunciation differences
    → May include acceptable accent variations
    → Good for strict pronunciation training
  
  • Lower value (0.70-0.75) = LESS sensitive
    → Only catches clear mispronunciations
    → Ignores minor accent/dialect variations
    → Good for casual/conversational assessment

Similarity Scale:
  1.00 = Perfect match
  0.90-0.99 = Very close (minor variation)
  0.80-0.89 = Noticeable difference
  0.70-0.79 = Clear difference
  0.60-0.69 = Significant difference
  < 0.60 = Very different

Current Setting: {:.2f}
  → This will catch deviations with similarity below {:.2f}
  → Words with similarity >= {:.2f} are considered acceptable
""".format(PRONUNCIATION_DEVIATION_THRESHOLD, PRONUNCIATION_DEVIATION_THRESHOLD, 
           PRONUNCIATION_DEVIATION_THRESHOLD, PRONUNCIATION_DEVIATION_THRESHOLD))

print("="*80)
print("Recommended Settings")
print("="*80)

presets = {
    "Very Strict (Pronunciation Training)": {
        "PRONUNCIATION_DEVIATION_THRESHOLD": 0.95,
        "PRONUNCIATION_SEVERITY_MINOR": 0.90,
        "PRONUNCIATION_SEVERITY_MODERATE": 0.80,
        "PRONUNCIATION_SEVERITY_NOTABLE": 0.80,
        "description": "Catches almost all variations, including minor accent differences"
    },
    "Strict (Public Speaking)": {
        "PRONUNCIATION_DEVIATION_THRESHOLD": 0.88,
        "PRONUNCIATION_SEVERITY_MINOR": 0.85,
        "PRONUNCIATION_SEVERITY_MODERATE": 0.75,
        "PRONUNCIATION_SEVERITY_NOTABLE": 0.75,
        "description": "Balances clarity requirements with natural variation"
    },
    "Moderate (Default)": {
        "PRONUNCIATION_DEVIATION_THRESHOLD": 0.85,
        "PRONUNCIATION_SEVERITY_MINOR": 0.85,
        "PRONUNCIATION_SEVERITY_MODERATE": 0.70,
        "PRONUNCIATION_SEVERITY_NOTABLE": 0.70,
        "description": "Current setting - good for general use"
    },
    "Lenient (Conversational)": {
        "PRONUNCIATION_DEVIATION_THRESHOLD": 0.75,
        "PRONUNCIATION_SEVERITY_MINOR": 0.80,
        "PRONUNCIATION_SEVERITY_MODERATE": 0.65,
        "PRONUNCIATION_SEVERITY_NOTABLE": 0.65,
        "description": "Only catches clear mispronunciations, accepts accents"
    },
    "Very Lenient (ESL Friendly)": {
        "PRONUNCIATION_DEVIATION_THRESHOLD": 0.65,
        "PRONUNCIATION_SEVERITY_MINOR": 0.75,
        "PRONUNCIATION_SEVERITY_MODERATE": 0.60,
        "PRONUNCIATION_SEVERITY_NOTABLE": 0.60,
        "description": "Most forgiving, only flags major pronunciation issues"
    }
}

for i, (name, preset) in enumerate(presets.items(), 1):
    print(f"\n{i}. {name}")
    print(f"   {preset['description']}")
    print(f"   Deviation Threshold: {preset['PRONUNCIATION_DEVIATION_THRESHOLD']}")
    print(f"   Severity Cutoffs: Minor={preset['PRONUNCIATION_SEVERITY_MINOR']}, "
          f"Moderate={preset['PRONUNCIATION_SEVERITY_MODERATE']}")

print("\n" + "="*80)
print("How to Change Thresholds")
print("="*80)

print("""
1. Edit config.py and modify these lines:

   # Pronunciation detection thresholds
   PRONUNCIATION_DEVIATION_THRESHOLD = 0.88  # Change this number
   PRONUNCIATION_SEVERITY_MINOR = 0.85
   PRONUNCIATION_SEVERITY_MODERATE = 0.75
   PRONUNCIATION_SEVERITY_NOTABLE = 0.75

2. Restart your server:
   python app.py

3. Re-run your tests:
   python test_pronunciation_detection.py

4. Iterate until you find the right sensitivity

Tips:
  • Start with STRICT preset if not catching enough deviations
  • Use LENIENT preset if getting too many false positives
  • Fine-tune by adjusting values in 0.05 increments
  • Test with different accents to see the impact
""")

print("="*80)
print("Quick Comparison Tool")
print("="*80)

print("\nEnter a similarity score to see how it would be classified:")
print("(or press Enter to skip)")

try:
    user_input = input("Similarity (0.0-1.0): ").strip()
    if user_input:
        similarity = float(user_input)
        if 0.0 <= similarity <= 1.0:
            would_report = similarity < PRONUNCIATION_DEVIATION_THRESHOLD
            
            if similarity >= PRONUNCIATION_SEVERITY_MINOR:
                severity = "minor"
            elif similarity >= PRONUNCIATION_SEVERITY_MODERATE:
                severity = "moderate"
            else:
                severity = "notable"
            
            print(f"\nWith similarity {similarity:.2f}:")
            print(f"  Would be reported: {'YES ✓' if would_report else 'NO ✗'}")
            if would_report:
                print(f"  Severity: {severity.upper()}")
            
            # Show for all presets
            print(f"\n  Comparison with presets:")
            for name, preset in presets.items():
                would_report_preset = similarity < preset['PRONUNCIATION_DEVIATION_THRESHOLD']
                if similarity >= preset['PRONUNCIATION_SEVERITY_MINOR']:
                    sev = "minor"
                elif similarity >= preset['PRONUNCIATION_SEVERITY_MODERATE']:
                    sev = "moderate"
                else:
                    sev = "notable"
                
                status = "✓ Report" if would_report_preset else "✗ Skip"
                print(f"    {name:30s}: {status} ({sev})")
        else:
            print("Invalid range. Must be between 0.0 and 1.0")
except ValueError:
    pass
except EOFError:
    pass

print("\n" + "="*80)
print("Done! Update config.py with your preferred thresholds.")
print("="*80)
