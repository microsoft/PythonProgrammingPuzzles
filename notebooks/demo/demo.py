from __future__ import print_function
import ipywidgets as widgets
from IPython.display import clear_output, display, Markdown, Javascript
import IPython
from demo import study_puzzles

from typing import List, Set, Tuple, Callable
import inspect
import shutil
import os
import re
import time
import threading
import pickle as pkl
import logging
import getpass
import subprocess
from tempfile import TemporaryDirectory


IPYNP_FILE = 'Demo.ipynb'
LOCAL = True

#temp_dir = TemporaryDirectory() # local version. Will create a new dir each time so it's not stateful...
#out_dir = temp_dir.name
out_dir = "state"
os.makedirs(out_dir, exist_ok=True)
log_path = os.path.join(out_dir, 'run.log')

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.FileHandler(log_path)],
)

_max_mins = 6
_max_seconds = _max_mins * 60


def warmup_puzzle(s: str):
    return


def log(msg, level=logging.INFO):
    logger.log(level, msg)
    state.log.append((time.time(), level, msg))


def submit_years(years: int):
    if LOCAL:
        print("This is running locally")
    else:
        if type(years) != int:
            print('Please submit an integer.')
            return

        log(f"Number of years programming in python: {years}.")
        print('Thanks!')


def update_progress_bar():
    global progress_bar
    while True:
        time.sleep(1.0)
        try:
            if progress_bar.value == progress_bar.max:
                progress_bar.description = 'Out of time'
                progress_bar.bar_style = 'danger'
            progress_bar.value = time.time() - state.cur.start_time
        except:
            pass


# globals
STATE_FILENAME = os.path.join(out_dir, "state.pkl")

state = None  # set by load_state function
__puzzle__ = None  # set by next_puzzle function
n_attempts = None  # set by puzzle function
progress_bar = None  # for display
threading.Thread(target=update_progress_bar).start()  # progress bar timer update thread
fireworks_gif = None  # for display

global_hook = IPython.core.getipython.get_ipython().ev("globals()")  # we can use to reset puzzle fun if they override


def check_hook():
    if "puzzle" in global_hook and global_hook["puzzle"] is not puzzle:
        # print("Uh oh, seems that you have over-rided the puzzle function. Please don't do that. Re-defining it.")
        log("Doh! They overrode the puzzle function. :-(", level=logging.ERROR)
        global_hook["puzzle"] = puzzle
        save_state()


def restart_state():
    resp = input('Are you sure you want to delete the state? type "yes" to reset or "no" to cancel: ')
    if resp != 'yes':
        return
    print("OK!")

    t = '_' + time.ctime().replace(' ', '_')
    shutil.move(STATE_FILENAME, STATE_FILENAME + t)
    load_state()

def load_state():
    """creates if does not exist"""
    # print("    Loading state")
    global state
    check_hook()
    if os.path.exists(STATE_FILENAME):
        with open(STATE_FILENAME, "rb") as f:
            state = pkl.load(f)
    else:
        state = State()  # this is where the state is first initialized
        # print("*** Creating state")
        save_state()


def save_state():
    check_hook()
    # print("    Saving state")
    with open(STATE_FILENAME, "wb") as f:
        pkl.dump(state, f)

    # To store snapshots of the notebook:
    if False:
        display(Javascript("IPython.notebook.save_notebook()"))
        t = '_' + time.ctime().replace(' ', '_')
        #command = f'jupyter nbconvert {IPYNP_FILE} --output {os.path.join(out_dir, IPYNP_FILE + t)}.html'
        #subprocess.call(command)
        time.sleep(1)
        shutil.copy(IPYNP_FILE, os.path.join(out_dir, IPYNP_FILE + t + '.ipynb'))


class PuzzleData:
    def __init__(self, src: str, num: int, name: str, part: str):
        self.num = num
        self.name = name
        self.part = part
        self.start_time = None
        self.solve_time = None
        self.give_up_time = None
        self.n_attempts = None
        self.src = src

    def __str__(self):
        return f"""
        num = {self.num} 
        name = {self.name} 
        part = {self.part} 
        start_time = {self.start_time}
        solve_time = {self.solve_time} 
        give_up_time ={self.give_up_time} 
        n_attempts   = {self.n_attempts}
        src         = {self.src}"""


class State:
    """
    state.cur points to the PuzzleData of state.puzzles that we are currently on. When done, state.cur is None.
    initially state.cur = self.puzzles[0] but its start_time is None.
    """

    def __init__(self):
        self.log = []  # redundant log
        self.puzzles = []
        for p in study_puzzles.get_puzzles():
            self.puzzles.append(PuzzleData(p["src"], len(self.puzzles), p["name"], p["part"]))
        self.cur = self.puzzles[0]

    def __str__(self):
        return f"cur: {self.cur}" + "\n" + f"log: [{len(self.log)}]" + "\n".join(str(l)[:100] for l in self.log[-5:])


def submit_feedback(text):
    log("Feedback: " + text)
    save_state()
    if LOCAL:
        print("This is the local version so we won't see this. But please send us an email!")
    else:
        print("Feedback logged, thank you!")


def notify_next():
    print("Run next_puzzle() when you are ready to begin the next puzzle.")


def cur_puzzle(reload_state=True):
    if reload_state:
        load_state()
    if state.cur is None:
        print(f"All parts complete!")
        print_solving_times()
        return

    if state.cur.start_time is None or state.cur.solve_time is not None or state.cur.give_up_time is not None:
        notify_next()
        return

    if time.time() - state.cur.start_time > _max_seconds:
        print('Time is up.')
        give_up()
        return

    reset_widgets()

    display(progress_bar)
    if state.cur.give_up_time is None and time.time() - state.cur.start_time > _max_seconds:
        print("Out of time.")

    print(f"{state.cur.name} ({state.cur.part})")  # (STUDY {_study_idx}/{NUM_STUDIES}):")
    print("============")
    print()
    print(state.cur.src)


with open("fireworks.gif", "rb") as f:
    _fireworks_image = f.read()


def reset_widgets():  # close existing widgets
    try:
        global progress_bar, fireworks_gif
        if progress_bar is not None:
            progress_bar.close()
        progress_bar = widgets.IntProgress(value=0, min=0, max=_max_seconds, description='Time:', bar_style='warning')
        if fireworks_gif is not None:
            fireworks_gif.close()
        fireworks_gif = widgets.Image(
            value=_fireworks_image,
            format='gif',
            width=100,
            height=200,
        )
    except Exception as e:
        logger.error("reset_widgets exception")


def print_solving_times():
    print("=" * 10)
    print('Check our "Programming Puzzles" paper (section 5.1) to see how difficult GPT-3 and others found each puzzle to be: https://arxiv.org/abs/2106.05784')
    print("=" * 10)
    print("Your solving times:")
    for i, puz in enumerate(state.puzzles):
        if i < 3:
            # Warmup.
            continue

        if state.cur is not None and state.cur.num < puz.num:
            return

        if puz.solve_time:
            elapsed = puz.solve_time - puz.start_time
            time_str = time.strftime("%M:%S", time.gmtime(elapsed))
            print(f"Puzzle {puz.num - 2}: {time_str} seconds")
        else:
            print(f"Puzzle {puz.num - 2}: Unsolved")


def check_finished_part():
    if state.cur is None:
        cur_puzzle(reload_state=False)
    elif state.cur is state.puzzles[-1]: # done!
        state.cur = None
        save_state()
        cur_puzzle(reload_state=False) # notifies that they are done
        return True
    else:
        if state.cur.part != state.puzzles[state.cur.num + 1].part:
            print(f"Finished {state.cur.part}!!!")
            print("Continue to the next part when you are ready.")
            if state.cur.part != "WARM UP":  # Warmup
                print_solving_times()
            else:
                print("You will get a summary of your solving times after each part.")
            state.cur = state.puzzles[state.cur.num + 1]
            save_state()
            return True
    return False


def give_up():
    # don't load state
    check_hook()
    if state.cur is None:
        cur_puzzle()  # notifies them that they are done
        return
    if state.cur.solve_time:
        print("Cannot give up since you already solved this puzzle.")
    else:
        if not state.cur.start_time:
            print("Cannot give up on a puzzle before you started it.")
        elif state.cur.give_up_time is None:
            elapsed = time.time() - state.cur.start_time
            if elapsed < _max_seconds:
                resp = input('Are you sure you want to give up on this puzzle? type "yes" to give up or "no" to keep trying: ')
                if resp != 'yes':
                    return
            state.cur.give_up_time = time.time()
            elapsed = state.cur.give_up_time - state.cur.start_time
            if last["num"] == state.cur.num:
                state.cur.n_attempts = last["n_attempts"]
            log(f"Gave up {state.cur.num} after {elapsed:.2f} seconds and {last['n_attempts']} attempts")
            reset_widgets()
            save_state()
            if check_finished_part():
                return

    notify_next()


def next_puzzle():
    load_state()
    if state.cur is None:  # already done
        cur_puzzle(reload_state=False)  # prints completion msg
        return


    if state.cur.start_time is not None:
        if state.cur.solve_time is None and state.cur.give_up_time is None:
            elapsed = time.time() - state.cur.start_time
            if elapsed < _max_seconds:
                print(f"You haven't solved this puzzle yet and time has not expired.")
                print(f"You have {_max_seconds - elapsed:.0f} seconds to continue trying to solve the puzzle.")
                print("You may type give_up() and then next_puzzle(), or cur_puzzle() to see the current puzzle.")
                return
            else:
                log(f"Time out {state.cur.num} after {elapsed:.2f} seconds and {last['n_attempts']} attempts")

        reset_widgets()
        if check_finished_part():
            return

        state.cur = state.puzzles[state.cur.num + 1]
    else:
        reset_widgets()


    state.cur.start_time = time.time()
    save_state()

    cur_puzzle(reload_state=False)


def check_type(x):  # TODO: update to multiple arguments, tuples, and functions
    desired_type_str = re.match(r"def puzzle\([^:]*: (.*)\)", state.cur.src).groups(1)[0].strip()

    def helper(obj, type_str):
        t = type(obj)
        if type_str == "int":
            success = t == int
        elif type_str == "str":
            success = t == str
        elif type_str == "float":
            success = t == float
        else:
            assert "[" in type_str and type_str[-1] == "]", f"Unknown type {type_str}"
            inner_type = type_str[type_str.index("[") + 1:-1].strip()
            success = False
            if type_str.startswith("List"):
                if t == list:
                    if not all(helper(y, inner_type) for y in obj):
                        return False
                    success = True
            elif type_str.startswith("Set"):
                if t == set:
                    if not all(helper(y, inner_type) for y in obj):
                        return False
                    success = True
            else:
                assert False, f"TODO: implement type checking for '{type_str}'"

        if not success:
            print(f"TypeError: puzzle expecting {type_str}, got type {t} in: {str(obj)[:50]}...")
        return success

    return helper(x, desired_type_str)


last = {  # previous evaluation, tracked so we don't load_state if called a million times in a loop
    "num": None,
    "time": None,
    "n_attempts": None,
    "func": None
}

class AlreadySolvedError(Exception):
    pass

def puzzle(solution):
    if state.cur is None or state.cur.num != last["num"] or time.time() - last["time"] > 10:
        load_state()  # if more than 10 seconds have elapsed since last attempt
        if state.cur is None:
            cur_puzzle()  # notify that all done
            return
        if state.cur.start_time is None:
            print("Haven't started.")
            notify_next()
            return
        if state.cur.num != last["num"]:
            loc = locals().copy()
            exec(state.cur.src, None, loc)
            last["func"] = loc["puzzle"]
            last["num"] = state.cur.num
            last["n_attempts"] = 0

    if state.cur.solve_time is not None:
        print(f"You've already solved {state.cur.name}")
        notify_next()
        raise AlreadySolvedError

    if state.cur.give_up_time is not None:
        print(f"You ran out of time for {state.cur.name}")
        notify_next()
        return

    last["n_attempts"] += 1
    last["time"] = time.time()

    if time.time() - state.cur.start_time > _max_seconds:
        print('Time is up.')
        give_up()
        return

    if not check_type(solution):
        return TypeError

    puzzle_ = last["func"]

    result = puzzle_(solution)  # exceptions happen

    last["time"] = time.time()
    elapsed = last["time"] - state.cur.start_time

    if result is True:
        reset_widgets()
        state.cur.solve_time = last["time"]
        state.cur.n_attempts = last["n_attempts"]  # only update when solved or give up
        time_str = time.strftime("%M:%S", time.gmtime(elapsed))
        display(Markdown(f'<span style="color: green">CORRECT in {time_str} sec.</span>'))
        log(f"Solved {state.cur.num} in {elapsed:.2f} seconds using {last['n_attempts']} attempts")
        save_state()
        display(fireworks_gif, widgets.Output())
        check_finished_part()

    return result


load_state()  # initialize
