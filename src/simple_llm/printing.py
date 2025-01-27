import os
from termcolor import colored

def print_role(role: str):
    print(colored("\n"+role+":", "yellow"), end="\n\n")

def print_dashed_line(frac: float = 0.7):
    line_width = int(os.get_terminal_size().columns * frac)
    print("-"*line_width)

def print_message(role: str, content: str) -> None:
    """
    Print messages for chat display
    """
    
    # Print role in yellow
    print(colored("\n"+role+":", "yellow"), end="\n\n")

    # Print main message content
    print(content, end="\n\n")

    # Print dashes covering the width of the terminal
    print_dashed_line()