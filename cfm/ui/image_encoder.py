import os
import base64
from tqdm import tqdm

def icons_to_base64(folder):    
    names = [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.ico') or f.endswith('.gif')]
    outfile = open(os.path.join(folder, 'icons.py'), 'w')
    for icon in tqdm(names, desc="Converting icons to base64: "):
        content = open(os.path.join(folder, icon), 'rb').read()
        encoded = base64.b64encode(content)
        variable_name = f"ICON_{icon[:icon.rfind('.')].upper()}"
        outfile.write('{} = {}\n'.format(variable_name, encoded))
    outfile.close()


if __name__ == '__main__':
    icons_to_base64('../icons')