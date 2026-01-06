"""Command-line interface for the sequence analyzer."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from .parser import parse_sequence, format_sequence
from .ensemble import EnsembleDetector, EnsembleResult, EnsemblePrediction

app = typer.Typer(
    name="seq",
    help="Sequence Analyzer - Fill in missing numbers using AI and math",
    add_completion=False,
)
console = Console()


def format_confidence(confidence: float) -> str:
    """Format confidence as colored percentage."""
    pct = confidence * 100
    if pct >= 90:
        return f"[green]{pct:.1f}%[/green]"
    elif pct >= 70:
        return f"[yellow]{pct:.1f}%[/yellow]"
    elif pct >= 50:
        return f"[orange3]{pct:.1f}%[/orange3]"
    else:
        return f"[red]{pct:.1f}%[/red]"


def format_methods(prediction: EnsemblePrediction) -> str:
    """Format the methods that contributed to a prediction."""
    method_icons = {
        "rule-based": "[blue]R[/blue]",
        "oeis": "[green]O[/green]",
        "constants": "[yellow]C[/yellow]",
        "ml/transformer": "[magenta]T[/magenta]",
        "ml/lstm": "[cyan]L[/cyan]",
    }

    icons = []
    for method in prediction.methods:
        for key, icon in method_icons.items():
            if method.startswith(key) or method == key:
                if icon not in icons:
                    icons.append(icon)
                break
        else:
            icons.append("[white]?[/white]")

    return " ".join(icons)


def display_result(
    sequence_str: str,
    result: EnsembleResult,
    top_k: int = 5,
    explain: bool = False,
) -> None:
    """Display analysis results in a beautiful format."""
    from .parser import parse_sequence

    sequence = parse_sequence(sequence_str)

    # Header
    console.print()
    console.print(Panel(
        f"[bold]Input:[/bold] {format_sequence(sequence)}",
        title="[bold blue]Sequence Analysis[/bold blue]",
        border_style="blue",
    ))

    # Legend
    legend = "[blue]R[/blue]=Rules  [green]O[/green]=OEIS  [yellow]C[/yellow]=Constants  [magenta]T[/magenta]=Transformer  [cyan]L[/cyan]=LSTM"
    console.print(f"\n[dim]Methods: {legend}[/dim]\n")

    # Results for each missing position
    for idx in sequence.missing_indices:
        predictions = result.get_top_k(idx, k=top_k)

        if not predictions:
            console.print(f"[red]Position {idx + 1}: No predictions found[/red]")
            continue

        # Create table for this position
        table = Table(
            title=f"[bold]Position {idx + 1}[/bold] (index {idx})",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )

        table.add_column("Rank", style="dim", width=4)
        table.add_column("Value", style="bold cyan", justify="right")
        table.add_column("Confidence", justify="right")
        table.add_column("Methods", justify="center")

        if explain:
            table.add_column("Explanation", style="dim")

        for rank, pred in enumerate(predictions, 1):
            row = [
                str(rank),
                str(pred.value),
                format_confidence(pred.combined_confidence),
                format_methods(pred),
            ]

            if explain:
                row.append(pred.best_explanation or "-")

            table.add_row(*row)

        console.print(table)
        console.print()

    # Summary with filled sequence
    best_predictions = {}
    for idx in sequence.missing_indices:
        best = result.get_best(idx)
        if best:
            best_predictions[idx] = best.value

    if best_predictions:
        filled_values = []
        for i, v in enumerate(sequence.values):
            if v is None:
                if i in best_predictions:
                    filled_values.append(f"[bold green]{best_predictions[i]}[/bold green]")
                else:
                    filled_values.append("[red]?[/red]")
            else:
                filled_values.append(str(int(v) if v == int(v) else v))

        filled_str = ", ".join(filled_values)
        console.print(Panel(
            f"[bold]Result:[/bold] {filled_str}",
            title="[bold green]Completed Sequence[/bold green]",
            border_style="green",
        ))


@app.command()
def analyze(
    sequence: str = typer.Argument(
        ...,
        help="Sequence with missing values (e.g., '55, ?, 144, 233, 377, 610')",
    ),
    top: int = typer.Option(
        5,
        "--top", "-t",
        help="Number of predictions to show per position",
    ),
    explain: bool = typer.Option(
        False,
        "--explain", "-e",
        help="Show explanations for predictions",
    ),
    rules: bool = typer.Option(
        True,
        "--rules/--no-rules",
        help="Use rule-based detection",
    ),
    oeis: bool = typer.Option(
        True,
        "--oeis/--no-oeis",
        help="Use OEIS lookup",
    ),
    ml: bool = typer.Option(
        True,
        "--ml/--no-ml",
        help="Use ML models",
    ),
    constants: bool = typer.Option(
        True,
        "--constants/--no-constants",
        help="Use mathematical constants detection (e, pi, etc.)",
    ),
) -> None:
    """
    Analyze a sequence and predict missing values.

    Examples:
        seq analyze "1, 2, ?, 4, 5"
        seq analyze "55, ?, 144, 233, 377, 610" --explain
        seq analyze "2, 4, ?, 16" --top 3 --no-oeis
    """
    try:
        parsed = parse_sequence(sequence)
    except ValueError as e:
        console.print(f"[red]Error parsing sequence:[/red] {e}")
        raise typer.Exit(1)

    if not parsed.missing_indices:
        console.print("[yellow]No missing values found in sequence.[/yellow]")
        console.print(f"Parsed: {format_sequence(parsed)}")
        raise typer.Exit(0)

    # Run analysis
    async def run_analysis():
        detector = EnsembleDetector(
            use_rules=rules,
            use_oeis=oeis,
            use_ml=ml,
            use_constants=constants,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Analyzing sequence...", total=None)
            return await detector.detect(parsed)

    result = asyncio.run(run_analysis())
    display_result(sequence, result, top_k=top, explain=explain)


@app.command()
def solve(
    sequence: str = typer.Argument(
        ...,
        help="Sequence with missing values",
    ),
) -> None:
    """
    Quick solve - just output the most likely answer.

    Examples:
        seq solve "55, ?, 144, 233, 377, 610"
    """
    try:
        parsed = parse_sequence(sequence)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not parsed.missing_indices:
        console.print(format_sequence(parsed))
        raise typer.Exit(0)

    async def run_analysis():
        detector = EnsembleDetector()
        return await detector.detect(parsed)

    with console.status("Solving..."):
        result = asyncio.run(run_analysis())

    # Output just the answers
    answers = []
    for idx in parsed.missing_indices:
        best = result.get_best(idx)
        if best:
            answers.append(str(best.value))
        else:
            answers.append("?")

    if len(answers) == 1:
        console.print(answers[0])
    else:
        console.print(", ".join(answers))


@app.command()
def check(
    sequence: str = typer.Argument(
        ...,
        help="Complete sequence to verify",
    ),
) -> None:
    """
    Check if a sequence matches known patterns.

    Examples:
        seq check "1, 1, 2, 3, 5, 8, 13"
    """
    try:
        parsed = parse_sequence(sequence)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if parsed.missing_indices:
        console.print("[yellow]Warning: Sequence has missing values[/yellow]")

    # Analyze by masking each position
    console.print(f"\nAnalyzing: {format_sequence(parsed)}\n")

    async def analyze_patterns():
        from .detectors.rules import RuleBasedDetector
        from .detectors.oeis import OEISDetector

        rule_detector = RuleBasedDetector()
        oeis_detector = OEISDetector()

        # For pattern checking, we don't mask anything
        # Just run detectors on the full sequence
        rule_result = await rule_detector.detect(parsed)
        oeis_result = await oeis_detector.detect(parsed)

        return rule_result, oeis_result

    rule_result, oeis_result = asyncio.run(analyze_patterns())

    # Display detected patterns
    patterns_found = []

    # Check rule-based patterns by looking at what patterns would be detected
    # if we artificially masked a position
    from .detectors.base import Sequence

    test_sequence = Sequence(values=[None] + parsed.values[1:])

    async def test_patterns():
        from .detectors.rules import RuleBasedDetector
        detector = RuleBasedDetector()
        return await detector.detect(test_sequence)

    test_result = asyncio.run(test_patterns())

    if test_result.predictions:
        for idx, preds in test_result.predictions.items():
            for pred in preds:
                if pred.pattern_name and pred.confidence > 0.8:
                    patterns_found.append((pred.pattern_name, pred.explanation or ""))

    if patterns_found:
        console.print("[green]Detected patterns:[/green]")
        for name, explanation in set(patterns_found):
            console.print(f"  [bold]{name}[/bold]: {explanation}")
    else:
        console.print("[yellow]No strong pattern matches found[/yellow]")


@app.command()
def version() -> None:
    """Show version information."""
    from . import __version__
    console.print(f"seq-analyzer version {__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
