"""
Scheduler module for automatic 24/7 operation with daily 7am-5pm capture windows.
"""
import time
import logging
import threading
from datetime import datetime, time as dt_time
from typing import Callable, Optional


class DailyScheduler:
    """
    Scheduler that runs a function daily between specified hours.
    Default: 7am to 5pm (17:00) daily.
    """

    def __init__(self,
                 start_hour: int = 7,
                 start_minute: int = 0,
                 end_hour: int = 17,
                 end_minute: int = 0,
                 timezone_offset: int = 0):
        """
        Initialize the daily scheduler.

        Args:
            start_hour: Hour to start (24-hour format, default 7 for 7am)
            start_minute: Minute to start (default 0)
            end_hour: Hour to end (24-hour format, default 17 for 5pm)
            end_minute: Minute to end (default 0)
            timezone_offset: Timezone offset in hours (default 0 for local time)
        """
        self.start_time = dt_time(start_hour, start_minute)
        self.end_time = dt_time(end_hour, end_minute)
        self.timezone_offset = timezone_offset

        self.is_running = False
        self.is_capture_active = False
        self.capture_function: Optional[Callable] = None
        self.stop_function: Optional[Callable] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        logging.info(
            f"Scheduler initialized: {start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d}")

    def set_capture_function(self, capture_func: Callable, stop_func: Optional[Callable] = None):
        """
        Set the function to call when capture should start/stop.

        Args:
            capture_func: Function to call when capture should start
            stop_func: Optional function to call when capture should stop
        """
        self.capture_function = capture_func
        self.stop_function = stop_func

    def is_capture_time(self) -> bool:
        """
        Check if current time is within capture window.

        Returns:
            True if current time is within capture window
        """
        now = datetime.now()
        current_time = now.time()

        # Handle overnight periods (e.g., 11pm to 6am)
        if self.start_time > self.end_time:
            return current_time >= self.start_time or current_time <= self.end_time
        else:
            return self.start_time <= current_time <= self.end_time

    def _scheduler_loop(self):
        """Main scheduler loop that runs in a separate thread."""
        logging.info("Scheduler started")

        while not self._stop_event.is_set():
            try:
                current_capture_state = self.is_capture_time()

                # State transition: not capturing -> should capture
                if not self.is_capture_active and current_capture_state:
                    logging.info("Starting daily capture window")
                    self.is_capture_active = True
                    if self.capture_function:
                        try:
                            self.capture_function()
                        except Exception as e:
                            logging.error(f"Error starting capture: {e}")

                # State transition: capturing -> should stop
                elif self.is_capture_active and not current_capture_state:
                    logging.info("Stopping daily capture window")
                    self.is_capture_active = False
                    if self.stop_function:
                        try:
                            self.stop_function()
                        except Exception as e:
                            logging.error(f"Error stopping capture: {e}")

                # Sleep for 1 minute before checking again
                self._stop_event.wait(60)

            except Exception as e:
                logging.error(f"Scheduler error: {e}")
                self._stop_event.wait(60)

        logging.info("Scheduler stopped")

    def start(self):
        """Start the scheduler in a background thread."""
        if self.is_running:
            logging.warning("Scheduler is already running")
            return

        self.is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop, daemon=True)
        self._thread.start()
        logging.info("Scheduler started in background")

    def stop(self):
        """Stop the scheduler."""
        if not self.is_running:
            return

        logging.info("Stopping scheduler...")
        self.is_running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        # Stop capture if it's currently active
        if self.is_capture_active and self.stop_function:
            try:
                self.stop_function()
            except Exception as e:
                logging.error(f"Error stopping capture during shutdown: {e}")

        logging.info("Scheduler stopped")

    def get_status(self) -> dict:
        """
        Get current scheduler status.

        Returns:
            Dictionary with scheduler status information
        """
        return {
            "is_running": self.is_running,
            "is_capture_active": self.is_capture_active,
            "is_capture_time": self.is_capture_time(),
            "start_time": self.start_time.strftime("%H:%M"),
            "end_time": self.end_time.strftime("%H:%M"),
            "current_time": datetime.now().strftime("%H:%M:%S")
        }


def create_scheduler_from_config(config: dict) -> DailyScheduler:
    """
    Create a DailyScheduler from configuration.

    Args:
        config: Configuration dictionary with scheduler settings

    Returns:
        Configured DailyScheduler instance
    """
    scheduler_config = config.get("scheduler", {})

    return DailyScheduler(
        start_hour=scheduler_config.get("start_hour", 7),
        start_minute=scheduler_config.get("start_minute", 0),
        end_hour=scheduler_config.get("end_hour", 17),
        end_minute=scheduler_config.get("end_minute", 0),
        timezone_offset=scheduler_config.get("timezone_offset", 0)
    )
