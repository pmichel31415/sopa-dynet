#!/usr/bin/env python3

import sys


class Logger(object):
    """Utility to log to or stdout"""

    def __init__(self, file=None, verbose=True, flush=True):
        self.verbose = verbose
        self.file = file or sys.stdout
        self.flush = flush

    def __call__(self, msg):
        if self.verbose:
            print(msg, file=self.file)
            if self.flush:
                self.file.flush()
