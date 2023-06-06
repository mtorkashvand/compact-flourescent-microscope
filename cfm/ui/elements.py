# Modules
from typing import Dict, List
from collections import defaultdict
import PySimpleGUI as sg

from cfm.zmq.client_with_gui import GUIClient

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
        self.client = None
        return
    # Handle
    def handle(self, **kwargs):
        raise NotImplementedError()
    # Disable
    def disable(self):
        for element in self.elements:
            try:
                element.update(disabled=True)
            except Exception as e:
                pass
        return
    def enable(self):
        for element in self.elements:
            try:
                element.update(disabled=False)
            except Exception as e:
                pass
        return
    # Values
    def add_values(self, values):
        raise NotImplementedError()
    # Get Value
    def get(self):
        raise NotImplementedError()
    # Set Client
    def set_client(self, client: GUIClient):
        self.client = client
        return
## Return Key Handler
class ReturnHandler(AbstractElement):
    # Constructor
    def __init__(self) -> None:
        super().__init__()
        self.key = "RETURN-KEY"
        self.elements = [
            sg.Button(visible=False, key=self.key, bind_return_key=True)
        ]
        self.events = { self.key }
        return
    def set_window(self, window):
        self.window = window
        return
    def handle(self, **kwargs):
        element_focused = self.window.find_element_with_focus()
        self.window.write_event_value(
            element_focused.key,
            '<RETURN-KEY-CALL>'
        )
        return
    def add_values(self, values):
        return
    def disable(self):
        return
    def enable(self):
        return
## Input Element with Autoselect and Enter Event handling
class InputAutoselect(AbstractElement):
    def __init__(self, key: str, default_text: int, size, type_caster=int, bounds=(None, None)) -> None:
        super().__init__()
        self.key = key
        self.default_text = default_text
        self.type_caster = type_caster
        self.bound_lower, self.bound_upper = bounds
        self.input = sg.Input(
            default_text=self.default_text,
            key=self.key,
            justification='righ',
            size=size
        )
        self.events = { self.key }
        self.elements = [ self.input ]
        return
    # Handle
    def handle(self, **kwargs):
        value = self.get()
        if self.bound_lower is not None:
            value = max( self.bound_lower, value )
        if self.bound_upper is not None:
            value = min( self.bound_upper, value )
        self.input.update(value=value)
        return
    # Values
    def add_values(self, values):
        values[self.key] = self.get()
        return
    # Value
    def get(self):
        raw = self.input.get()
        if not isinstance(raw, str):
            return self.type_caster(raw)
        # remove character values
        res = "0" + "".join([
            c for c in raw if c.isdigit()
        ])
        return self.type_caster( res )
    # Bounds
    def set_bounds(self, bound_lower=None, bound_upper=None):
        if bound_lower:
            self.bound_lower = bound_lower
        if bound_upper:
            self.bound_upper = bound_upper
        self.handle()
        return
## LED Combined Elements
class LEDCompound(AbstractElement):
    # Cosntructor
    def __init__(
            self, button_text: str, text:str, key: str,
            led_name: str,
            color_off: str = '#ff0000', color_on: str = "#00ff00",
            type_caster = int,
            bounds=(None, None)
        ) -> None:
        super().__init__()
        self.color_on = color_on
        self.color_off = color_off
        self.button_text = button_text
        self.led_name = led_name
        self.bounds = bounds
        self.key = key
        self.key_toggle = f"{self.key}-TOGGLE"
        self.key_input= f"{self.key}-INPUT"
        self.button = sg.Button(
            button_text=self.button_text,
            key=self.key_toggle,
            button_color=color_off
        )
        self.text = sg.Text(
            text=text
        )
        self.input_as = InputAutoselect(
            key=self.key_input, default_text='0', size=3, type_caster=type_caster,
            bounds=self.bounds
        )
        self.elements = [
            self.button, self.text, *self.input_as.elements
        ]
        self.events = {
            self.key_toggle, self.key_input
        }
        self.toggle = False
        return
    # Handle
    def handle(self, **kwargs):
        event = kwargs['event']
        if event == self.key_toggle:
            self.toggle = not self.toggle
            button_color = self.color_on if self.toggle else self.color_off
            self.button.update(button_color=button_color)
        elif event == self.key_input:
            self.input_as.handle(**kwargs)
        if self.toggle:
            intensity = self.input_as.get()
            client_cli_cmd = f"DO _teensy_commands_set_led {self.led_name} {intensity}"
            print(f"Executing: '{client_cli_cmd}'")
            self.client.process(client_cli_cmd)
        return
    def add_values(self, values):
        values[self.key] = self.input_as.get()
        return
    def set_bounds(self, bound_lower=None, bound_upper=None):
        self.input_as.set_bounds(
            bound_lower=bound_lower,
            bound_upper=bound_upper
        )
        return
    def get(self):
        if not self.toggle:
            return 0
        return self.input_as.get()
## LED IR
class LEDIR(AbstractElement):
    # Cosntructor
    def __init__(
            self,
            button_text: str = 'IR LED - Icon',
            text:str = 'Toggle IR LED',
            key: str = 'led_ir',
            color_off: str = '#ff0000', color_on: str = "#00ff00",
            type_caster = int,
            bounds=(None, None)
        ) -> None:
        super().__init__()
        self.color_on = color_on
        self.color_off = color_off
        self.button_text = button_text
        self.bounds = bounds
        self.key = key
        self.key_toggle = f"{self.key}-TOGGLE"
        self.button = sg.Button(
            button_text=self.button_text,
            key=self.key_toggle,
            button_color=color_off
        )
        self.text = sg.Text(
            text=text
        )
        self.input = sg.Input(default_text='255', size=3, disabled=True, key='')
        self.elements = [
            self.button, self.text, self.input
        ]
        self.events = {
            self.key_toggle
        }
        self.toggle = False
        return
    # Handle
    def handle(self, **kwargs):
        event = kwargs['event']
        if event == self.key_toggle:
            self.toggle = not self.toggle
            button_color = self.color_on if self.toggle else self.color_off
            self.button.update(button_color=button_color)
        state_str = "n" if self.toggle else "f"
        client_cli_cmd = f"DO _teensy_commands_set_toggle_led {state_str}"
        print(f"Executing: '{client_cli_cmd}'")
        self.client.process(client_cli_cmd)
        return
    def add_values(self, values):
        values[self.key] = self.get()
        return
    def get(self):
        return self.toggle
## Exposure Combined Elements
class ExposureCompound(AbstractElement):
    # Cosntructor
    def __init__(
            self, button_text: str, text:str, key: str,
            camera_name: str,
            type_caster = int,
            bounds=(None, None)
        ) -> None:
        super().__init__()
        self.button_text = button_text
        self.key = key
        self.camera_name = camera_name
        self.bounds = bounds
        self.button = sg.Button(
            button_text=self.button_text,
            disabled=True,
            enable_events=False
        )
        self.text = sg.Text(
            text=text
        )
        self.input_as = InputAutoselect(
            key=self.key, default_text='18000', size=8, type_caster=type_caster,
            bounds=self.bounds
        )
        self.elements = [
            self.button, self.text, *self.input_as.elements
        ]
        self.events = {
            self.key
        }
        return
    # Handle
    def handle(self, **kwargs):
        framerate = kwargs['framerate']
        self.input_as.handle(**kwargs)
        exposure_time = self.input_as.get()
        client_cli_cmd = "DO _flir_camera_set_exposure_framerate_{} {} {}".format(
            self.camera_name, exposure_time, framerate
        )
        print(f"Executing: '{client_cli_cmd}'")
        self.client.process(client_cli_cmd)
        return
    def add_values(self, values):
        values[self.key] = self.input_as.get()
        return
    def set_bounds(self, bound_lower=None, bound_upper=None):
        self.input_as.set_bounds(
            bound_lower=bound_lower,
            bound_upper=bound_upper
        )
        return
    def get(self):
        return self.input_as.get()
## Framerate Compound
class FramerateCompound(AbstractElement):
    # Cosntructor
    def __init__(
            self,
            element_exposure_behavior: ExposureCompound,
            element_exposure_gfp: ExposureCompound,
            key: str = 'framerate',
            text: str = "Frame Rate between: 1-48",
            bounds=(1, 48),
            type_caster = int
        ) -> None:
        super().__init__()
        self.element_exposure_behavior = element_exposure_behavior
        self.element_exposure_gfp = element_exposure_gfp
        self.key = key
        self.bounds = bounds
        self.button = sg.Button(
            button_text='framerate-icon',
            disabled=True,
            enable_events=False
        )
        self.text = sg.Text(
            text=text
        )
        self.input_as = InputAutoselect(
            key=self.key, default_text='20', size=3, type_caster=type_caster,
            bounds=self.bounds
        )
        self.elements = [
            self.button, self.text, *self.input_as.elements
        ]
        self.events = {
            self.key
        }
        return
    # Handle
    def handle(self, **kwargs):
        self.input_as.handle(**kwargs)
        framerate = self.get()
        # Update Bounds
        bound_upper = int( 995000/framerate )
        self.element_exposure_behavior.set_bounds(bound_upper=bound_upper)
        self.element_exposure_gfp.set_bounds(bound_upper=bound_upper)
        # Handle
        self.element_exposure_behavior.handle(framerate = framerate)
        self.element_exposure_gfp.handle(framerate = framerate)
        return
    def add_values(self, values):
        values[self.key] = self.input_as.get()
        return
    def set_bounds(self, bound_lower=None, bound_upper=None):
        self.input_as.set_bounds(
            bound_lower=bound_lower,
            bound_upper=bound_upper
        )
        return
    def get(self):
        return self.input_as.get()
## 
class InputWithIncrements(AbstractElement):
    # Constructor
    def __init__(self, text:str, key: str, default_value: int, increments: List[int] = [-1, 1], bounds: List[int] = [-1024, 1024], type_caster=int) -> None:
        super().__init__()
        self.text = text
        self.default_value = default_value
        self.bounds = bounds
        self.bound_lower = min(self.bounds)
        self.bound_upper = max(self.bounds)
        self.key = key
        self.increments =  increments
        self.type_caster = type_caster
        self.key_to_offset = {
            f"{key}--{inc}": inc for inc in self.increments
        }
        self.events = set(self.key_to_offset)
        self.events.add(self.key)
        self.input = sg.Input(default_text=self.default_value, key=key, size=5)
        self.elements = [
            sg.Text(self.text)
        ] + [
            sg.Button(button_text=f"{inc}",key=event) for event, inc in self.key_to_offset.items() if inc < 0
        ] + [ self.input ] + [
            sg.Button(button_text=f"{inc}",key=event) for event, inc in self.key_to_offset.items() if inc > 0
        ]
        return
    # Handle
    def handle(self, **kwargs):
        event = kwargs['event']
        if event == self.key:
            pass
        else:
            inc = self.key_to_offset[event]
            value_current = self.type_caster( kwargs[self.key] )
            value_new = min(
                self.bound_upper,
                max(
                    self.bound_lower,
                    value_current + inc
                )
            )
            self.input.update(value = value_new)
        return
    # Values
    def add_values(self, values):
        values[self.key] = self.type_caster(
            self.input.get()
        )
        return
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
    def __init__(self, text1: str, text2: str, v1_to_v2s: Dict[str, str], default_v1: str, default_v2: str, key1: str, key2: str) -> None:
        super().__init__()
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
        super().__init__()
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
## Z-Interpolation Tracking
class ZInterpolationTracking(AbstractElement):
    # Cosntructor
    def __init__(self) -> None:
        super().__init__()
        self.key = "ZINTERP"
        self.color_disabled = "#555555"
        self.color_set = "#00ff00"
        self.color_unset = "#ff0000"
        self.key_checkbox = f"{self.key}-CHECKBOX"
        self.key_p1 = f"{self.key}-P1"
        self.key_p2 = f"{self.key}-P2"
        self.key_p3 = f"{self.key}-P3"

        self.checkbox = sg.Checkbox(
            text="Plane-Interpolation Z-Tracking",
            key=self.key_checkbox,
            enable_events=True,
            default=False
        )
        self.p1 = sg.Button(
            button_text="Set Point 1",
            key=self.key_p1,
            enable_events=True,
            disabled=True,
            button_color=self.color_disabled
        )
        self.p1_is_set = False
        self.p2 = sg.Button(
            button_text="Set Point 2",
            key=self.key_p2,
            enable_events=True,
            disabled=True,
            button_color=self.color_disabled
        )
        self.p2_is_set = False
        self.p3 = sg.Button(
            button_text="Set Point 3",
            key=self.key_p3,
            enable_events=True,
            disabled=True,
            button_color=self.color_disabled
        )
        self.p3_is_set = False

        self.elements = [
            self.checkbox, self.p1, self.p2, self.p3,
        ]
        self.events = {
            self.key_checkbox, self.key_p1, self.key_p2, self.key_p3,
        }
        return
    # Handle
    def handle(self, **kwargs):
        event = kwargs['event']
        if event == self.key_checkbox:
            p_disabled = not self.checkbox.get()
            button_color = self.color_disabled if p_disabled else self.color_unset
            self.p1.update(disabled=p_disabled, button_color=button_color)
            self.p2.update(disabled=p_disabled, button_color=button_color)
            self.p3.update(disabled=p_disabled, button_color=button_color)
            self.p1_is_set, self.p2_is_set, self.p3_is_set = False, False, False
        elif event == self.key_p1:
            self.p1_is_set = True
            self.p1.update(button_color=self.color_set)
            client_cli_cmd = "DO set_point 1"
            print(f"Executing: '{client_cli_cmd}'")
            self.client.process(client_cli_cmd)
        elif event == self.key_p2:
            self.p2_is_set = True
            self.p2.update(button_color=self.color_set)
            client_cli_cmd = "DO set_point 2"
            print(f"Executing: '{client_cli_cmd}'")
            self.client.process(client_cli_cmd)
        elif event == self.key_p3:
            self.p3_is_set = True
            self.p3.update(button_color=self.color_set)
            client_cli_cmd = "DO set_point 3"
            print(f"Executing: '{client_cli_cmd}'")
            self.client.process(client_cli_cmd)
        return
    def add_values(self, values):
        values[self.key] = self.get()
        return
    def get(self):
        return {
            'checkbox': self.checkbox.get(),
            'p1': self.p1_is_set,
            'p2': self.p2_is_set,
            'p3': self.p3_is_set,
        }