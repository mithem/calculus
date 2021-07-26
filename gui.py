import PySimpleGUI as sg
from PySimpleGUI import Text, Button, Window
from functions import Constant, Linear, Polynomial


def refresh_layout():
    global layout
    layout = [[Text(f"f{i}"), Text(str(functions[i])), Button(
        f"Delete f{i}")] for i in range(len(functions))]


functions = [Constant(2), Linear(3), Polynomial({2: 1, 3: 0.2})]
layout = None
refresh_layout()

window = Window("Calculus", layout)
while True:
    event, values = window.read()
    if event == "OK" or event == sg.WINDOW_CLOSED:
        break
    if event.startswith("Delete f"):
        i = int(event.replace("Delete f", ""))
        del functions[i]
        refresh_layout()
        window.layout(layout)

window.close()
