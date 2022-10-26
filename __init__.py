import random
from time import sleep
from typing import Union

import keyboard
from windows_adb_screen_capture import ScreenShots
from a_pandas_ex_tesseract_multirow_regex_fuzz import (
    pd_add_regex_fuzz_multiline,
    pd_add_tesseract,
)
import pandas as pd
import numpy as np
import cv2


pd_add_regex_fuzz_multiline()


def sub_color_in_image(img: np.ndarray, conditions: tuple, newcolor: tuple):
    r_coords, r_image = _substitute_colors(
        img, conditions, newcolor=newcolor, return_image=True, return_coords=False
    )
    return r_image


def _substitute_colors(
    img: np.ndarray,
    conditions: tuple = (("r", "<", 200), "&", ("g", "<", 200), "&", ("b", "<", 200)),
    newcolor: tuple = (255, 0, 0),
    return_coords: bool = True,
    return_image: bool = True,
):

    image = img.copy()
    r_coords = None
    r_image = None
    wholecommand = ""
    for co in conditions:
        if isinstance(co, tuple):
            if co[0].lower() == "r":
                layer = 2
            elif co[0].lower() == "g":
                layer = 1
            elif co[0].lower() == "b":
                layer = 0
            else:
                layer = 3
            if (
                layer in [0, 1, 2, 3]
                and co[1] in ["<", ">", "<=", ">=", "=="]
                and -1 < int(co[2]) < 256
            ):
                subcommand = f"(image[:, :, {layer}] {co[1]} {co[2]})"
                wholecommand += subcommand
            else:
                print("Command not valid")
                return
        else:
            if co in ["&", "|"]:
                wholecommand = wholecommand + co
            else:
                print("Command not valid")
                return
    if len(newcolor) == 4:
        alpha = newcolor[1]
        newcol = list(reversed(newcolor[:3]))
        newcol.append(alpha)
    else:
        newcol = list(reversed(newcolor))
    mask = eval(wholecommand)
    if return_coords:
        locations = np.where(mask)
        r_coords = tuple(zip(*locations[::-1]))
    if return_image:
        try:
            image[mask] = newcol
        except ValueError:
            newcol.append(255)
            image[mask] = newcol

        r_image = image
    return r_coords, r_image


def substitute_colors_with_equal_rgb_values(img, thresh=200, newcolor=(0, 0, 0)):
    image = img.copy()

    if len(newcolor) == 4:
        alpha = newcolor[1]
        newcol = list(reversed(newcolor[:3]))
        newcol.append(alpha)
    else:
        newcol = list(reversed(newcolor))

    try:
        image[
            (image[:, :, 0] == image[:, :, 1])
            & (image[:, :, 1] == image[:, :, 2])
            & (image[:, :, 0] < thresh)
        ] = newcol
    except ValueError:
        newcol.append(255)
        image[
            (image[:, :, 0] == image[:, :, 1])
            & (image[:, :, 1] == image[:, :, 2])
            & (image[:, :, 0] < thresh)
        ] = newcol
    return image


def draw_tesseract_results(dft, img, conf_thresh=-1):
    image = img.copy()
    for key, item in dft.iterrows():
        fontdistance1 = 5
        fontdistance2 = 15
        try:
            if int(item["conf"]) <= conf_thresh:
                continue
        except Exception:
            pass
        if item["text"].strip() == "":
            continue
        tmp_tl_x = item["left"]
        tmp_tl_y = item["top"]
        tmp_br_x = item["width"]
        tmp_br_y = item["height"]
        conf = item["conf"]
        text = item["text"]
        r_, g_, b_ = (
            random.randrange(50, 255),
            random.randrange(50, 255),
            random.randrange(50, 255),
        )

        image = cv2.rectangle(
            image,
            (tmp_tl_x, tmp_tl_y),
            (tmp_br_x + tmp_tl_x, tmp_br_y + tmp_tl_y),
            (r_, g_, b_),
            1,
        )

        try:
            image = cv2.putText(
                image,
                str(conf),
                (tmp_tl_x, tmp_tl_y - fontdistance1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 0),
                2,
            )
        except Exception:
            fontdistance1 = 0
            fontdistance2 = 0
            image = cv2.putText(
                image,
                str(conf),
                (tmp_tl_x, tmp_tl_y - fontdistance1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 0),
                2,
            )

        image = cv2.putText(
            image,
            str(conf),
            (tmp_tl_x, tmp_tl_y - fontdistance1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (r_, g_, b_),
            1,
        )
        image = cv2.putText(
            image,
            text,
            (tmp_tl_x, tmp_tl_y - fontdistance2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
        )
        image = cv2.putText(
            image,
            text,
            (tmp_tl_x, tmp_tl_y - fontdistance2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (r_, g_, b_),
            1,
        )

    return image


def get_tesseractdf(
    screenshot: np.ndarray,
    lang: Union[str, None] = None,
    config: str = "",
    nice: int = 0,
    timeout: int = 0,
    drop_empty_strings=True,
    conf_thresh=None,
) -> pd.DataFrame:
    try:
        dft = pd.Q_Tesseract_to_DF(
            screenshot, lang=lang, config=config, nice=nice, timeout=timeout
        )
        dft["middle_x"] = dft.left + (dft.width // 2)
        dft["middle_y"] = dft.top + (dft.height // 2)
        dft = dft.dropna(subset=["conf", "text"])
        dft["conf"] = dft["conf"].astype("Float64")
        if drop_empty_strings:
            dft = dft.loc[dft["conf"] != -1]
            dft = dft.loc[dft.text.str.strip() != ""]
        if conf_thresh is not None:
            dft = dft.loc[dft["conf"] > conf_thresh]
        dft["end_x"] = dft.left + dft.width
        dft["end_y"] = dft.top + dft.height
        return dft.reset_index(drop=True)
    except Exception as fe:
        return pd.DataFrame(
            columns=[
                "level",
                "page_num",
                "block_num",
                "par_num",
                "line_num",
                "word_num",
                "left",
                "top",
                "width",
                "height",
                "conf",
                "text",
                "middle_x",
                "middle_y",
                "end_x",
                "end_y"
            ]
        )


r"""
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
"""


r"""
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
"""


r"""
# example Bluestacks hwnd no window

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
"""
