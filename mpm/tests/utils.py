import sys
import os

tests_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(tests_dir))


class TextColor:
    Yellow = "\033[93m"
    Green = "\033[92m"
    Cyan = "\033[96m"
    Red = "\033[91m"
    End = "\033[0m"


def print_green(text: str) -> str:
    return print_colored_text(text, TextColor.Green)


def print_colored_text(text: str, color: str) -> str:
    return f"{color}{text}{TextColor.End}"


def print_cyan(text: str) -> str:
    return print_colored_text(text, TextColor.Cyan)


def print_yellow(text: str) -> str:
    return print_colored_text(text, TextColor.Yellow)


def print_red(text: str) -> str:
    return print_colored_text(text, TextColor.Red)
