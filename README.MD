### Scan screenshots with Tesseract

<img src="https://github.com/hansalemaos/screenshots/raw/main/tesseractscreen.png"/>

[Video ](https://github.com/hansalemaos/screenshots/blob/main/tesseractscreen.mp4?raw=true)

```python
pip install tesseract-window-scanner
```

### Example 1 (screenshots from BlueStacks using ADB)

```python
import cv2
import pandas as pd
import numpy as np
from time import sleep
from tesseract_window_scanner import pd_add_tesseract, sub_color_in_image, \
    substitute_colors_with_equal_rgb_values, draw_tesseract_results, get_tesseractdf,keyboard,ScreenShots


def activate_stop():
    global stop
    stop = True

tesseractpath = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pd_add_tesseract(tesseractpath)
languages = "de+pt+deu"

sc2 = ScreenShots(
    hwnd=None, adb_path=r"C:\ProgramData\adb\adb.exe", adb_serial="localhost:5735"
)
quit_key = "q"
sc2.imshow_adb(sleep_time=0.05, quit_key=quit_key)
sleep(1)
sc2.enable_show_edited_images()
stop = False
keyboard.add_hotkey(quit_key, activate_stop)

showresults = True
while not stop:
    screenshot_window = sc2.imget_adb()

    # optional filter
    screenshot_window = sub_color_in_image(
        img=screenshot_window,
        conditions=(("r", ">", 200), "|", ("g", ">", 200), "|", ("b", ">", 200)),
        newcolor=(255, 255, 255),
    )
    df = get_tesseractdf(
        screenshot_window, lang=languages, drop_empty_strings=True, conf_thresh=60
    )
    tesserresults = draw_tesseract_results(
        dft=df, img=screenshot_window, conf_thresh=60
    )
    if showresults:
        sc2.show_edited_image(tesserresults)  # show the edited pic
    print(df[["text", "conf"]])
```

### Example 2 (screenshots from BlueStacks using hwnd)

```python
import cv2
import pandas as pd
import numpy as np
from time import sleep
from tesseract_window_scanner import pd_add_tesseract, sub_color_in_image, \
    substitute_colors_with_equal_rgb_values, draw_tesseract_results, get_tesseractdf,keyboard,ScreenShots

def activate_stop():
    global stop
    stop = True


tesseractpath = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
languages = "de+pt+deu"
pd_add_tesseract(tesseractpath)
sc2 = ScreenShots()
sc2.find_window_with_regex("[bB]lue[sS]tacks.*")
quit_key = "e"
sc2.imshow_hwnd(sleep_time=0.05, quit_key=quit_key)
sleep(1)
sc2.enable_show_edited_images()
stop = False
keyboard.add_hotkey(quit_key, activate_stop)

showresults = True
while not stop:
    screenshot_window = sc2.imget_hwnd()

    # screenshot_window=sub_color_in_image(img=screenshot_window, conditions=(('r' ,'>', 200) ,'|' ,('g' ,'>', 200), '|', ('b' ,'>', 200)), newcolor=(255,255,255))
    df = get_tesseractdf(
        screenshot_window, lang=languages, drop_empty_strings=True, conf_thresh=60
    )
    tesserresults = draw_tesseract_results(
        dft=df, img=screenshot_window, conf_thresh=60
    )
    if showresults:
        sc2.show_edited_image(tesserresults)  # show the edited pic
    print(df[["text", "conf"]])
```

### Example 3 (screenshots from BlueStacks using hwnd without showing the results)

```python
import cv2
import pandas as pd
import numpy as np
from time import sleep
from tesseract_window_scanner import pd_add_tesseract, sub_color_in_image, \
    substitute_colors_with_equal_rgb_values, draw_tesseract_results, get_tesseractdf,keyboard,ScreenShots


def activate_stop():
    global stop
    stop = True    

tesseractpath = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
languages = "de+pt+deu"
pd_add_tesseract(tesseractpath)
sc2 = ScreenShots()
sc2.find_window_with_regex("[bB]lue[sS]tacks.*")

quit_key = "x"
stop = False
keyboard.add_hotkey(quit_key, activate_stop)

while not stop:
    screenshot_window = sc2.imget_hwnd()

    screenshot_window = sub_color_in_image(
        img=screenshot_window,
        conditions=(("r", ">", 200), "|", ("g", ">", 200), "|", ("b", ">", 200)),
        newcolor=(255, 255, 255),
    )
    df = get_tesseractdf(
        screenshot_window, lang=languages, drop_empty_strings=True, conf_thresh=60
    )
    print(df[["text", "conf"]])
```
