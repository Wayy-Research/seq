# Sequence Analyzer

A powerful CLI tool and Python library that analyzes number sequences with missing values and predicts the most likely answers using multiple detection methods with probability scores.

Perfect for solving sequence puzzles like [SEQUE-NCD](https://sequencd.com.au/) and understanding mathematical patterns.

## Features

- **Multiple Detection Methods**: Combines rule-based pattern matching, OEIS lookup, and deep learning
- **Probability Scores**: Each prediction comes with a confidence score
- **Rich CLI Output**: Beautiful terminal interface with color-coded results
- **Extensible**: Easy to add new pattern detectors

### Supported Sequence Types

| Type | Example | Detection Method |
|------|---------|------------------|
| Arithmetic | 2, 4, 6, 8, 10 | Constant difference |
| Geometric | 2, 4, 8, 16, 32 | Constant ratio |
| Fibonacci | 1, 1, 2, 3, 5, 8 | Sum of previous two |
| Squares | 1, 4, 9, 16, 25 | Polynomial (degree 2) |
| Cubes | 1, 8, 27, 64, 125 | Polynomial (degree 3) |
| Linear Recurrence | General patterns | Matrix solving |
| OEIS Sequences | 350,000+ sequences | API lookup |

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd seq

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install the package
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- Dependencies: typer, rich, httpx, numpy, sympy, torch

## Usage

### CLI Commands

#### Analyze a Sequence

```bash
# Basic analysis
seq analyze "55, ?, 144, 233, 377, 610"

# With explanations
seq analyze "55, ?, 144, 233, 377, 610" --explain

# Show top 3 predictions
seq analyze "1, 4, ?, 16, 25" --top 3

# Disable specific methods
seq analyze "1, 2, ?, 4" --no-ml --no-oeis
```

**Output:**
```
╭───────────────────────────── Sequence Analysis ──────────────────────────────╮
│ Input: 55, [?], 144, 233, 377, 610                                           │
╰──────────────────────────────────────────────────────────────────────────────╯

Methods: R=Rules  O=OEIS  T=Transformer  L=LSTM

                            Position 2 (index 1)
╭──────┬───────┬────────────┬─────────┬────────────────────────────────────╮
│ Rank │ Value │ Confidence │ Methods │ Explanation                        │
├──────┼───────┼────────────┼─────────┼────────────────────────────────────┤
│ 1    │    89 │     100.0% │    R    │ Fibonacci-like sequence: a = a + a │
╰──────┴───────┴────────────┴─────────┴────────────────────────────────────╯

╭───────────────────────────── Completed Sequence ─────────────────────────────╮
│ Result: 55, 89, 144, 233, 377, 610                                           │
╰──────────────────────────────────────────────────────────────────────────────╯
```

#### Quick Solve

Get just the answer without the full display:

```bash
seq solve "55, ?, 144, 233, 377, 610"
# Output: 89

seq solve "2, 4, ?, 16, 32"
# Output: 8
```

#### Check a Sequence

Verify if a complete sequence matches known patterns:

```bash
seq check "1, 1, 2, 3, 5, 8, 13"
# Detects: Fibonacci sequence
```

### Missing Value Indicators

The parser accepts various ways to indicate missing values:

- `?` or `??` - Question marks
- `_` or `__` - Underscores
- `x` or `X` - Letter x
- `-` - Dash

```bash
seq solve "1, 2, _, 4, 5"    # Works
seq solve "1, 2, x, 4, 5"    # Works
seq solve "1, 2, ?, 4, 5"    # Works
```

### Input Formats

Flexible input parsing:

```bash
seq solve "1, 2, ?, 4"       # Comma-separated
seq solve "1 2 ? 4"          # Space-separated
seq solve "[1, 2, ?, 4]"     # With brackets
seq solve "1,2,?,4"          # No spaces
```

## Python API

Use the analyzer programmatically:

```python
import asyncio
from seq.parser import parse_sequence
from seq.ensemble import EnsembleDetector, analyze_sequence

# Quick analysis
async def main():
    result = await analyze_sequence("55, ?, 144, 233, 377, 610")

    for idx in [1]:  # Index of missing value
        best = result.get_best(idx)
        print(f"Position {idx}: {best.value} (confidence: {best.combined_confidence:.1%})")
        print(f"Explanation: {best.best_explanation}")

asyncio.run(main())
```

### Using Individual Detectors

```python
from seq.parser import parse_sequence
from seq.detectors.rules import RuleBasedDetector
import asyncio

async def detect_pattern():
    seq = parse_sequence("1, 4, 9, ?, 25, 36")
    detector = RuleBasedDetector()
    result = await detector.detect(seq)

    for idx, predictions in result.predictions.items():
        for pred in predictions:
            print(f"{pred.pattern_name}: {pred.value} ({pred.confidence:.1%})")

asyncio.run(detect_pattern())
# Output: Square numbers: 16 (100.0%)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Interface                           │
│                  seq analyze "55, ?, 144, 233"                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Sequence Parser                           │
│              Parse input, identify missing positions            │
└─────────────────────────────────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Rule-Based    │   │      OEIS       │   │  Deep Learning  │
│    Detector     │   │     Lookup      │   │     Models      │
│                 │   │                 │   │                 │
│ • Arithmetic    │   │ • 350k+         │   │ • Transformer   │
│ • Geometric     │   │   sequences     │   │ • LSTM          │
│ • Fibonacci     │   │ • API queries   │   │                 │
│ • Polynomial    │   │                 │   │                 │
│ • Recurrence    │   │                 │   │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Ensemble Combiner                          │
│         Aggregate predictions, compute final probabilities      │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Rich Output                              │
│          Display predictions with confidence + explanations     │
└─────────────────────────────────────────────────────────────────┘
```

### Detection Methods

#### 1. Rule-Based Detection (`seq/detectors/rules.py`)

Mathematical pattern matching:

- **Arithmetic**: Detects constant differences (e.g., 2, 4, 6, 8)
- **Geometric**: Detects constant ratios (e.g., 2, 4, 8, 16)
- **Fibonacci**: Detects sum-of-previous-two patterns
- **Polynomial**: Fits polynomials up to degree 4 (squares, cubes, etc.)
- **Linear Recurrence**: General recurrence relations via matrix solving

#### 2. OEIS Integration (`seq/detectors/oeis.py`)

Queries the [Online Encyclopedia of Integer Sequences](https://oeis.org/):

- Searches 350,000+ known sequences
- Finds alignment between input and OEIS entries
- Extracts missing values from matched sequences

#### 3. Deep Learning (`seq/detectors/ml.py`)

Neural network models (requires training for best results):

- **Transformer**: Encoder-only architecture for sequence prediction
- **LSTM**: Bidirectional LSTM for local pattern recognition

### Ensemble Combiner (`seq/ensemble.py`)

Combines predictions from all methods:

- Groups similar predictions (within tolerance)
- Weights by method reliability and confidence
- Boosts confidence when multiple methods agree
- Prefers integer results over floats

## Project Structure

```
seq/
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── seq/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point (typer)
│   ├── parser.py           # Sequence parsing
│   ├── ensemble.py         # Combine predictions
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py         # Base detector interface
│   │   ├── rules.py        # Rule-based patterns
│   │   ├── oeis.py         # OEIS lookup
│   │   └── ml.py           # Deep learning models
│   └── models/
│       ├── transformer.py  # Transformer architecture
│       └── lstm.py         # LSTM architecture
└── tests/
    ├── test_parser.py      # Parser tests
    └── test_rules.py       # Rule detector tests
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_rules.py -v

# Run with coverage
pytest tests/ --cov=seq --cov-report=html
```

### Adding a New Detector

1. Create a new file in `seq/detectors/`
2. Inherit from `Detector` base class
3. Implement the `detect()` method
4. Register in `seq/detectors/__init__.py`
5. Add to `EnsembleDetector` in `seq/ensemble.py`

Example:

```python
from .base import Detector, DetectionResult, Prediction, Sequence

class MyDetector(Detector):
    name = "my-detector"

    async def detect(self, sequence: Sequence) -> DetectionResult:
        result = DetectionResult()
        # Your detection logic here
        return result
```

## Examples

### Solving SEQUE-NCD Puzzles

```bash
# Today's puzzle: Fibonacci sequence
seq solve "55, ?, 144, 233, 377, 610"
# Answer: 89

# Geometric sequence
seq solve "3, 9, ?, 81, 243"
# Answer: 27

# Square numbers
seq solve "1, 4, 9, ?, 25, 36"
# Answer: 16

# Arithmetic sequence
seq solve "5, 10, ?, 20, 25"
# Answer: 15
```

### Multiple Missing Values

```bash
seq analyze "1, ?, 3, ?, 5" --explain
# Predicts: 2 and 4 (arithmetic sequence)
```

## License

MIT License

## Acknowledgments

- [OEIS](https://oeis.org/) - The Online Encyclopedia of Integer Sequences
- [SEQUE-NCD](https://sequencd.com.au/) - Daily sequence puzzles
