import random
import time

from rich.console import Console

console = Console()


def main():  # noqa: ANN201, D103
    lights = ["🟩", "🟥", "🟦", "🟨", "🟪", "⬜", "⬛"]
    console.print("[bold magenta]🪩  DISCOPAT is grooving... 🪩[/bold magenta]")
    for _ in range(20):
        line = "".join(random.choice(lights) for _ in range(20))  # noqa: S311
        console.print(line)
        time.sleep(0.1)
    console.print("[bold cyan]Now back to science 🧠[/bold cyan]")
