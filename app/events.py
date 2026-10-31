"""Progress events, so the UI can show the agent working as it happens.

A listener is set per request with `listen`. Code anywhere below it (the main agent, the
sub-agents) reports progress with `emit` and `step` without passing callbacks around. Steps
nest: a step started while another is running (the deal agent's web searches inside
find_deals) carries that step's id as its parent.

A request can also be made `stoppable`: when the user presses Stop, the agents call
`check_stopped` before each next model call or tool, and it raises Stopped.
"""

import itertools
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

Listener = Callable[[dict], None]

_listener: ContextVar[Listener | None] = ContextVar("listener", default=None)
_parent: ContextVar[int | None] = ContextVar("parent", default=None)
_stop: ContextVar[threading.Event | None] = ContextVar("stop", default=None)
_ids = itertools.count(1)


class Stopped(Exception):
    """The user pressed Stop. Nothing from this turn is kept."""


@contextmanager
def stoppable(stop: threading.Event):
    token = _stop.set(stop)
    try:
        yield
    finally:
        _stop.reset(token)


def check_stopped() -> None:
    stop = _stop.get()
    if stop is not None and stop.is_set():
        raise Stopped()


@contextmanager
def listen(callback: Listener):
    token = _listener.set(callback)
    try:
        yield
    finally:
        _listener.reset(token)


def emit(type: str, **fields) -> None:
    callback = _listener.get()
    if callback:
        callback({"type": type, "parent": _parent.get(), **fields})


@contextmanager
def step(tool: str, tool_input: dict):
    """Report a tool call's start and end. Fill the yielded dict with status/output."""
    step_id = next(_ids)
    emit("step", id=step_id, tool=tool, input=tool_input)
    outcome: dict = {}
    token = _parent.set(step_id)
    try:
        yield outcome
    finally:
        _parent.reset(token)
        emit("step_done", id=step_id, tool=tool, **outcome)
