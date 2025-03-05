from datetime import datetime

class LogDeduplicator:
    def __init__(self):
        self.last_message = None
        self.count = 0
        self.last_print_time = 0

    def log(self, message, logger_fn):
        current_time = datetime.now().timestamp()
        if message == self.last_message and current_time - self.last_print_time < 5:
            self.count += 1
        else:
            if self.count > 0:
                logger_fn(f"(Suppressed similar outputs {self.count} times)")
            logger_fn(message)
            self.last_message = message
            self.count = 0
            self.last_print_time = current_time 