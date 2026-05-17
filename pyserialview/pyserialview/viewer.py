import logging
import threading
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Callable, Dict, Optional
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Minimum GUI refresh rate in milliseconds
MIN_REFRESH_RATE = 2500

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def view(
    producer_func: Callable[..., None],
    consumer_func: Callable[..., None],
    producer_config: Optional[Dict] = None,
    consumer_config: Optional[Dict] = None,
    refresh_rate: int = MIN_REFRESH_RATE
) -> None:
    """
    Runs a producer and a consumer concurrently in a Tkinter GUI with a Matplotlib plot.

    Args:
        producer_func: Function producing data. Runs in a background thread.
            Must accept 'lock' and keys from producer_config as kwargs.
        consumer_func: Function consuming/plotting data. Runs in the main GUI thread.
            Must accept 'axes', 'canvas', 'lock', and keys from consumer_config as kwargs.
        producer_config: Dictionary with configuration for producer function (mandatory).
        consumer_config: Dictionary with configuration for consumer function (mandatory).
        refresh_rate: Plot refresh rate in milliseconds (minimum 2500ms).

    Raises:
        ValueError: If either producer_config or consumer_config is None or missing mandatory keys.
    """

    # Validate configs
    if producer_config is None or consumer_config is None:
        raise ValueError("Both producer_config and consumer_config dictionaries must be provided.")
    
    refresh_rate = max(MIN_REFRESH_RATE, refresh_rate)

    # Single lock shared between producer and consumer
    root = tk.Tk()
    lock = threading.Lock()
    file_names = consumer_config.get("file_names", [])
    if isinstance(file_names, str):
        file_names = [file_names]
    num_csv = len(file_names)

    if num_csv == 0:
        raise FileNotFoundError("No CSV files found.")

    # GUI and Plot Setup
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = Figure(figsize=(10, 6), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=root)
    axes = [fig.add_subplot(num_csv, 1, i + 1) for i in range(num_csv)]
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    root.title(producer_config.get("title", "PySerialView"))
    root.geometry(producer_config.get("geometry", "850x650"))

    # --- Producer wrapper ---
    def _producer_wrapper(**kwargs):
        try:
            producer_func(**kwargs)
        except Exception as e:
            error_msg = f"Producer thread crashed: {str(e)}"
            logging.error(error_msg)
            os._exit(1)   # Kill the entire process immediately

    
    # --- Consumer wrapper ---
    def _consumer_wrapper():
        try:
            consumer_func(axes=axes, canvas=canvas, lock=lock, **consumer_config)
        except Exception as e:
            error_msg = f"Consumer thread crashed: {str(e)}"
            logging.error(error_msg)
            os._exit(1)   # Kill the entire process immediately

            return  

        root.after(refresh_rate, _consumer_wrapper)



    # Start producer thread
    producer_kwargs = {**producer_config, "lock": lock }
    thread = threading.Thread(target=_producer_wrapper, kwargs=producer_kwargs, daemon=True)
    thread.start()

    # Start consumer loop
    _consumer_wrapper()

    # Start Tkinter main loop
    root.mainloop()

