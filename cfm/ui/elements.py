# Modules
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
        return
    # Handle
    def handle(self, **kwargs):
        raise NotImplementedError()
    # Disable
    def disable(self):
        raise NotImplementedError()
    def enable(self):
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
    def __init__(self) -> None:
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
    # Disable
    def disable(self):
        self.input.update(disabled=True)
        self.slider.update(disabled=True)
        return
    def enable(self):
        self.input.update(disabled=False)
        self.slider.update(disabled=False)
        return
    # Value
    @property
    def value(self):
        return self.type_caster(
            self.input.get()
        )
    