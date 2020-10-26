import sys

from typing import List, Optional, Tuple, Union
from pprint import pprint


from PyInquirer import style_from_dict, Token, prompt, Separator


from gans.core.exit_status import ExitStatus
from gans.models.your_first_gan import YourFirstGan


__GANS__ = """

 ▄▄ •  ▄▄▄·  ▐ ▄ .▄▄ ·
▐█ ▀ ▪▐█ ▀█ •█▌▐█▐█ ▀.
▄█ ▀█▄▄█▀▀█ ▐█▐▐▌▄▀▀▀█▄
▐█▄▪▐█▐█ ▪▐▌██▐█▌▐█▄▪▐█
·▀▀▀▀  ▀  ▀ ▀▀ █▪ ▀▀▀▀

"""

style = style_from_dict(
    {
        Token.Separator: "#cc5454",
        Token.QuestionMark: "#673ab7 bold",
        Token.Selected: "#cc5454",  # default
        Token.Pointer: "#673ab7 bold",
        Token.Instruction: "",  # default
        Token.Answer: "#f44336 bold",
        Token.Question: "",
    }
)


def main(args: List[Union[str, bytes]] = sys.argv) -> ExitStatus:
    """
    The main function

    :param args: [description], defaults to sys.argv
    :type args: List[Union[str, bytes]], optional
    :rtype: ExitStatus
    """

    print(__GANS__)
    print("received arguments:", args)

    questions = [
        {
            "type": "list",
            "name": "module",
            "message": "What program?",
            "choices": [
                {"name": "Your First Gan", "value": YourFirstGan, "short": "😎"}
            ],
        }
    ]

    answers = prompt(questions, style=style)

    gan = YourFirstGan({"name": "My first GAN, man"})
    gan.train()

    return ExitStatus.SUCCESS
