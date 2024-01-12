# Copyright 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime to use in Kenning scenarios for CVNode."""

from kenning.core.runtime import Runtime


class CVNodeRuntime(Runtime):
    """
    Represents the runtime for CVNode scenarios.
    Utilizes the default implementation of the Runtime class.

    Any unused methods are not implemented,
    ensuring that abstract method errors are not raised.
    """

    def extract_output(self):
        raise NotImplementedError

    def prepare_model(self, input_data):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def load_input(self, X):
        raise NotImplementedError
