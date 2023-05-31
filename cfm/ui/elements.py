# Modules
from collections import defaultdict
import PySimpleGUI as sg

# Parameters

# Methods
## Input for Ports
def sg_input_port(key, port):
    return sg.Input(
        key=key,
        size=5,
        default_text=str(port)
    )

# Classes
## Abstract Element with disable/enable, status
class AbstractElement:
    # Constructor
    def __init__(self) -> None:
        self.events = set()
        self.elements = list()
        return
    # Handle
    def handle(self, **kwargs):
        raise NotImplementedError()
    # Disable
    def disable(self):
        for element in self.elements:
            element.update(disabled=True)
        return
    def enable(self):
        for element in self.elements:
            element.update(disabled=False)
        return
    # Values
    def add_values(self, values):
        raise NotImplementedError()
## Advanced Ports Menu
class PortsMenu(AbstractElement):
    # TODO: create all ports inputs in a single element
    # - add toggle functionality to expand/collapse or display/hide ports options
    # Constructor
    def __init__(self) -> None:
        return
## Joined Combos
class CombosJoined(AbstractElement):
    # TODO: joined two combos based on each other values -> change colors of valid choices
    # e.g. binsize and format (shape of image crop)
    # Constructor
    def __init__(self, text1: str, text2: str, v1_to_v2s: dict[str, str], default_v1: str, default_v2: str, key1: str, key2: str) -> None:
        # Parameters
        self.v1_to_v2s = {
            k: sorted(v) for k,v in v1_to_v2s.items()
        }
        self.v2_to_v1s = defaultdict(set)
        for v1,v2s in self.v1_to_v2s.items():
            for v2 in v2s:
                self.v2_to_v1s[v2].add(v1)
        self.v2_to_v1s = {
            k:sorted(v) for k,v in self.v2_to_v1s.items()
        }
        self.default_v1, self.default_v2 = default_v1, default_v2
        self.key1, self.key2 = key1, key2
        self.events = { self.key1, self.key2 }
        # Elements
        assert self.default_v1 in self.v1_to_v2s, "`default_v1` is not in `v1_to_v2s`"
        assert self.default_v2 in self.v1_to_v2s[self.default_v1], "`default_v2` is not in `v1_to_v2s[default_v1]`"
        self.text1 = sg.Text(text1)  # TODO: add tooltip
        self.text2 = sg.Text(text2) # TODO: add tooltip
        self.combo1 = sg.Combo(
            values=self.v2_to_v1s[self.default_v2],
            default_value=self.default_v1,
            size=(20,3),  # TODO: change this value. not correct
            key=self.key1,
            enable_events=True
        )
        self.combo2 = sg.Combo(
            values=self.v1_to_v2s[self.default_v1],
            default_value=self.default_v2,
            size=(20,3),  # TODO: change this value. not correct
            key=self.key2,
            enable_events=True
        )
        self.elements = [ self.text1, self.combo1, self.text2, self.combo2, ]
        return
    # Handle
    def handle(self, **kwargs):
        event = kwargs['event']
        if event == self.key1:  # Update valid choices for v2
            v1 = kwargs[self.key1]
            v2s = self.v1_to_v2s[v1]
            self.combo2.update(values=v2s, value=v2s[0])
        elif event == self.key2:   # Update valid choices for v1
            v2 = kwargs[self.key2]
            v1s = self.v2_to_v1s[v2]
            self.combo1.update(values=v1s, value=v1s[0])
        return
    # Values
    def add_values(self, values):
        return
## Joined Input and Slider
class InputSlider(AbstractElement):
    """Input and slider connected to each other."""
    # Constructor
    def __init__(self, text, key, default_value=0.0, range=(0, 100), resolution=1, size_input=6, type_caster=float) -> None:
        # Parameters
        self.text = text
        self.key = key
        self.key_input = f"{self.key}_input"
        self.key_slider = f"{self.key}_slider"
        self.events = {
            self.key, self.key_input, self.key_slider
        }
        self.default_value = default_value
        self.range = range
        self.resolution = resolution
        self.size_input = size_input
        self.type_caster = type_caster
        # Elements
        self.text = sg.Text(text)  # add tooltip `tooltip`
        self.input = sg.Input(
            key=self.key_input,
            default_text=str(default_value),
            size=self.size_input,
            enable_events=True
        )  # update(value = value_new)
        self.slider = sg.Slider(
            key=self.key_slider,
            range=range,
            default_value=default_value,
            resolution=resolution,
            orientation='h',
            enable_events=True
        )
        self.elements = [self.text, self.input, self.slider]
        return
    # TODO
    # - connection between `input` and `slider`
    # Handle
    def handle(self, **kwargs):
        event = kwargs['event']
        if event == self.key_input:  # Changed Input -> Reflect on Slider
            value_input = self.type_caster( self.input.get() )
            self.slider.update(value = value_input)
        elif event == self.key_slider:  # Changed Slider -> Reflect on Input
            value_slider = self.type_caster( kwargs[self.key_slider] )
            self.input.update(value = value_slider)
    # Values
    def add_values(self, values):
        values[self.key] = self.type_caster( self.input.get() )
        return
    