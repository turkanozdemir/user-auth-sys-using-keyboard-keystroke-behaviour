from datetime import datetime
import pyxhook
import os
import time
import csv

# Constants for special keys
RETURN_KEY = "Return"
CONTROL_LEFT_KEY = "Control_L"
SHIFT_LEFT_KEY = "Shift_L"
SHIFT_RIGHT_KEY = "Shift_R"
CAPS_LOCK_KEY = "Caps_Lock"
SPACE_KEY = "space"
BACKSPACE_KEY = "BackSpace"
ALT_LEFT_KEY = "Alt_L"
TAB_KEY = "Tab"
ESCAPE_KEY = "Escape"

# Define a mapping for keys to events
key_event_mapping = {
    CONTROL_LEFT_KEY: "CONTROL_LEFT",
    SHIFT_LEFT_KEY: "SHIFT_LEFT",
    SHIFT_RIGHT_KEY: "SHIFT_RIGHT",
    CAPS_LOCK_KEY: "CAPS_LOCK",
    SPACE_KEY: "SPACE",
    BACKSPACE_KEY: "BACK_SPACE",
    ALT_LEFT_KEY: "ALT_LEFT",
    TAB_KEY: "TAB",
    ESCAPE_KEY: "ESCAPE",
}

def check_caps_lock(event):
    global capsLockOn

    if event.Key == CAPS_LOCK_KEY:
        capsLockOn = not capsLockOn
    
    return capsLockOn

def process_key_event(event, username, event_type):
    global capsLockOn
    capsLockOn = check_caps_lock(event)
    key = event.Key

    if key == RETURN_KEY:
        return
    
    if key == ESCAPE_KEY:
        create_csv_file(username)
        return
    
    if key in key_event_mapping:
        key_event = key_event_mapping[key]
    else:
        key_event = key.upper() if capsLockOn else key

    eventList.append((username, key_event, event_type, int(time.time() * 1000)))


def create_csv_file(username):
    date_time = datetime.now().strftime("%d.%m.%Y-%H:%M")
    csv_file = f'{os.getcwd()}/{username + "-" + date_time}.csv'

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        headers = ["User", "Key", "Event", "TimeInMillis"]
        writer.writerow(headers)
        writer.writerows(eventList)
    return csv_file

def main():
    username = input("Enter Your Name: ")
    print("\nHello " + username + "! Please enter given text above. To terminate the program you can press CTRL+C.\n")
    print("WARNING: Please do not use the numeric keypad for number entry!\a\n")

    print("Analysing keystroke dynamics, a topic that has been researched since the 1980s, helps improve security " +
          "by capturing unique patterns in user keystroke dynamics and adds an extra layer of authentication. " +
          "The main objective of the project is to analyze and model the data derived from keystroke dynamics generated during " +
          "keyboard use, aiming to authenticate users based on this information. Currently, the keystroke dynamics of users have been " +
          "collected with their explicit permission, exclusively for use within the project's defined scope. Thank you for your support!\n\n")

    global eventList
    eventList = []
    global capsLockOn
    capsLockOn = False

    new_hook = pyxhook.HookManager()
    new_hook.KeyDown = lambda event: process_key_event(event, username, "Down")
    new_hook.KeyUp = lambda event: process_key_event(event, username, "Up")
    new_hook.HookKeyboard()

    try:
        new_hook.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    except Exception as ex:
        msg = f"Error while catching events:\n  {ex}"
        pyxhook.print_err(msg)
        log_file = f'{os.getcwd()}/{username + "-" + date_time}.log'
        with open(log_file, "a") as f:
            f.write(f"\n{msg}")
    finally:
        new_hook.cancel()


if __name__ == "__main__":
    main()
