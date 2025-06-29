import torch 

from objects import CipherNoiseGRU
from helper_func import TARGET_VOCAB,INPUT_VOCAB, predict_correct_word

model_path = r"C:\Users\okere\OneDrive\Documents\ResearchFolder\Contex Cipher\prototype\ai_context_cipher_v2.pt"

gru_model = CipherNoiseGRU(embed_dim=16,
                      vocab_size=len(INPUT_VOCAB),
                      hidden_size=48,
                      pad_idx=INPUT_VOCAB.index('X'),
                      num_layers=3,
                      output_size=len(TARGET_VOCAB))

gru_model.load_state_dict(torch.load(model_path, weights_only=True))


# while True:
#     user_input = input("Enter a word (or type 'exit' to quit): ")
#     if user_input.lower() == 'exit':
#         break
#     predicted_word = predict_correct_word(gru_model, user_input)
#     print(f"Predicted correction: {predicted_word}")

print()
from rich.console import Console
from rich.panel import Panel
from rich import box
import sys, time

console = Console()
# ---------------------------------------------------------------------
# OPTIONAL: animated typing effect (set TYPE_DELAY = 0 to disable)
# ---------------------------------------------------------------------
TYPE_DELAY = 0.035  # seconds between characters

def type_print(text: str, style: str = "white"):
    """Print text with a typewriter animation (no newline)."""
    for ch in text:
        console.print(ch, end="", style=style, overflow="ignore")
        sys.stdout.flush()
        if TYPE_DELAY:
            time.sleep(TYPE_DELAY)
    console.print()  # newline


# ---------------------------------------------------------------------
# Inference loop (logic unchanged)
# ---------------------------------------------------------------------
console.print(
    Panel(
        "🧠  [bold bright_white]Polyphonic Cipher v2[/bold bright_white]\n"
        "[dim]Type a word, phrase, or sentence and press ⏎[/dim]",
        style="cyan",
        box=box.ROUNDED,
        expand=False,
    )
)

while True:
    user_input = console.input(
        "\n[bold bright_white]>>>[/bold bright_white] "
    ).strip()

    if user_input.lower() == "exit":
        console.print(Panel("[bold yellow]Goodbye![/bold yellow]", expand=False))
        break

    # ---------- PREDICTION (unchanged) ----------
    with console.status("[bold cyan]Thinking…[/bold cyan]", spinner="dots"):
        predicted_word = predict_correct_word(gru_model, user_input)
    # --------------------------------------------

    # nicely formatted output
    console.print()
    type_print("Plain-Text → ", style="bold bright_white")
    console.print("[bold bright_green]>>>[/bold bright_green]", end=' ')
    type_print(predicted_word, style="bold bright_green")
