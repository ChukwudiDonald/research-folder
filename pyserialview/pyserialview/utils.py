import os
import re
import csv
import time
import threading
import serial
from datetime import datetime
import logging
import pandas as pd
from typing import Iterable
import matplotlib.pyplot as plt
import random 

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# ---------------- CSV Utilities ----------------
def initialize_csv_file(file_name: str, columns: list[str]) -> None:
    """Creates a new CSV file with headers."""
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)


def parse_csv_line(line: str, delimiter: str = ',') -> list[str]:
    """Splits a delimited string into non-empty, stripped values."""
    return [part.strip() for part in line.strip().split(delimiter) if part.strip()]


def append_row_to_csv(file_name: str, row: list, columns: list[str]) -> None:
    """Appends a row to a CSV, creating file with headers if missing."""
    file_exists = os.path.exists(file_name)
    with open(file_name, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(columns)
        writer.writerow(row)

def is_valid_csv_line(line: str) -> bool:
    fields = line.strip().split(",")
    for field in fields:
        field = field.strip()
        try:
            float(field)  # can be int, float, negative, scientific
        except ValueError:
            return False
    return True



# ---------------- CSV Logger ----------------
def csv_logger(
    port: str,
    num_columns: int,
    output_filename: str,
    baud_rate: int = 115200,
    wait_time: int = 5,
    desc: str = 'column',
    lock: threading.Lock | None = None
) -> None:
    """
    Reads numeric CSV-like data from a serial port and logs it to a file.
    """
    columns = ["time"] + [f"{desc} {i}" for i in range(num_columns)]
    initialize_csv_file(file_name=output_filename, columns=columns)
    pattern = re.compile(r'^[0-9eE.\-]+$')

    with serial.Serial(port, baud_rate) as ser:
        logging.info(f"Connected to {port}. Logging to '{output_filename}'")
        logging.info("Press Ctrl+C to stop.")

        while True:
            line = ser.readline().decode('utf-8')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            if is_valid_csv_line(line):
                values = parse_csv_line(line)
                if len(values) == num_columns:
                    row = [timestamp] + values
                    if lock:
                        with lock:
                            append_row_to_csv(file_name=output_filename, row=row, columns=columns)
                    else:
                        append_row_to_csv(file_name=output_filename, row=row, columns=columns)
                    logging.info(f"{timestamp} - Recorded: {values}")
                    time.sleep(wait_time)
                else:
                    logging.info(f"{timestamp} - Expected {num_columns} columns, got {len(values)}.")

            else:
                logging.info(f"{timestamp} - {line.strip()}")


# ---------------- Running Mean Consumer ----------------
def running_mean_plotter(
    lock: threading.Lock,
    axes: list,
    canvas: plt.FigureCanvasBase,
    file_names: list[str],
    nrows: int = 10,
    col_indices: int | slice | list[int] | tuple[int, int] | None = None,
    plot_kwargs: dict | None = None,
    *,
    x_values: Iterable | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None
) -> None:
    """
    Reads the last nrows of CSV data, calculates running mean, and plots it.
    Timestamp column is ignored in calculations.

    Args:
        plot_kwargs: Dictionary of kwargs passed directly to axes.plot()
        x_values: Iterable of x-axis values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    if plot_kwargs is None:
        plot_kwargs = {}

    for ax, file_name in zip(axes, file_names):
        with lock:
            df = pd.read_csv(file_name)

        df_cropped = crop_df(df, col_indices, nrows)
        mean_values = df_cropped.mean()

        ax.clear()


        # Use provided x_values or default to column indices
        x_vals = x_values if x_values is not None else range(len(mean_values))

        # --- Default plot aesthetics ---
        defaults = {
            'label': 'Mean',
            'color': 'red',
            'marker': 'o',
            'linestyle': '-',
            'linewidth': 0.75,
            'markersize': 2
        }
        final_kwargs = {**defaults, **plot_kwargs}

        ax.plot(x_vals, mean_values.values, **final_kwargs)

        # --- Labels and title ---
        ax.set_title((title or "Running Mean") + f" - {os.path.basename(file_name)}")
        ax.set_xlabel(xlabel or "Data Columns")
        ax.set_ylabel(ylabel or "Mean Value")
        ax.legend()

    canvas.draw()



def random_producer(lock: threading.Lock, 
                    output_filename: str, 
                    num_columns: int = 5, 
                    wait_time: float = 1.0) -> None:
    
    """
    Writes random CSV lines to a file at regular intervals.
    
    Args:
        lock: threading.Lock for safe file writing.
        output_filename: CSV file to write.
        num_columns: Number of random numeric columns per row.
        wait_time: Seconds to wait between writing rows.
    """

    # Initialize CSV with header
    columns = ["time"] + [f"col_{i}" for i in range(num_columns)]
    initialize_csv_file(file_name=output_filename, columns=columns)


    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        values = [round(random.uniform(0, 100), 2) for _ in range(num_columns)]
        row = [timestamp] + values

        if lock:
            with lock:
                append_row_to_csv(file_name=output_filename, row=row, columns=columns)
        else:
                append_row_to_csv(file_name=output_filename, row=row, columns=columns)

        logging.info(f"{timestamp} - {values}")
        time.sleep(wait_time)


def flask_producer(app, **run_kwargs):
    default_kwargs = {
        'host': '0.0.0.0',
        'port': 5000,
        'debug': False,
        'use_reloader': False
    }
    final_kwargs = {**default_kwargs, **run_kwargs}
    app.run(**final_kwargs)




def crop_df(df, col_indices, nrows):
    """
    Crop dataframe to specified columns and rows.
    
    Args:
        df: Input DataFrame
        col_indices: Column indices to select (int, slice, list, tuple)
        nrows: Number of rows to select from the end
        
    Returns:
        Cropped DataFrame
    """
    if col_indices is not None:
        if isinstance(col_indices, (list, tuple)):
            col_indices = slice(*col_indices)
        df = df.iloc[:, col_indices]
    
    return df.tail(nrows)