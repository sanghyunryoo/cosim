import os

import glfw
import numpy as np
from core.policy import build_policy
from core.reporter import Reporter
from envs.build import build_env
from PyQt5.QtCore import QObject, pyqtSignal


class Tester(QObject):
    # Signal emitted when the test is finished
    finished = pyqtSignal()
    # Signal emitted after each step (to notify the MainWindow)
    stepFinished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.user_command = None
        self._push_event = False
        self._render_window_patched = False
        self._stop = False 
        self._had_error = False 

    def load_config(self, config):
        self.config = config

    def load_policy(self, policy_path):
        self.policy_path = policy_path

    def init_user_command(self):
        """Initialize the user command array before starting the test."""
        self.user_command = np.zeros(self.config["observation"]["command_dim"])

    def receive_user_command(self):
        """Send the current user command value to the environment."""
        if self.user_command is None:
            self.init_user_command()
        self.env.receive_user_command(self.user_command)

    def update_command(self, index, value):
        """Called from the UI to update a specific index of the command."""
        if self.user_command is None:
            self.init_user_command()
        if index < self.config["observation"]["command_dim"]:
            self.user_command[index] = value

    def activate_push_event(self, push_vel):
        self._push_event = True
        self._push_vel = push_vel

    def deactivate_push_event(self):
        self._push_event = False

    def test(self):
        # Create environment, reporter, and policy
        report_path = os.path.join(os.path.dirname(self.policy_path), 'report.pdf')
        self.reporter = Reporter(report_path=report_path, config=self.config)
        self.policy = build_policy(self.config, policy_path=os.path.join(self.policy_path))

        self.env = build_env(self.config)
        state, info = self.env.reset()
        done = False
   
        # Main test loop
        while not done and not self._stop:
            # Apply command values updated from the UI to the environment
            self.receive_user_command()
            try:
                action = self.policy.get_action(state)
            except Exception as e:
                self.close()
                self._had_error = True
                raise RuntimeError(f"Failed to run inference with the selected ONNX policy: {self.policy_path}."
                                   "\n\nPlease ensure that you have selected a valid ONNX file."
                                   f"\n\nThe current state length (={state.shape[-1]}) may not match the input length expected by the ONNX policy"
                                   ", which could have caused this error.\n") from e

            # Trigger event
            if self._push_event:
                self.env.event(event="push", value=self._push_vel)

            # Render environment
            self.env.render()
            if not self._render_window_patched:
                self._patch_render_window()
                self._render_window_patched = True

            assert self.user_command is not None, "user_command must not be None."
            next_state, terminated, truncated, info = self.env.step(action)

            # Record info to reporter and emit stepFinished signal
            self.reporter.write_info(info)
            self.stepFinished.emit()

            done = terminated or truncated
            state = next_state

        # Finalize test and clean up
        if not self._had_error:
            self.reporter.generate_report()
        self.close()
        self.finished.emit()

    def stop(self):
        """Stop the test loop."""
        self._stop = True

    def close(self):
        """Attempt to close the environment."""
        try:
            self.env.close()
        except Exception:
            pass

    def _patch_render_window(self):
        """Replace window close behavior: pressing X will stop the test loop."""
        try:
            glfw.init()
            win = glfw.get_current_context()
            if not win:
                return

            def _on_close(w):
                self._stop = True
                glfw.set_window_should_close(w, False)

            glfw.set_window_close_callback(win, _on_close)

        except Exception as e:
            print(f"[WARN] close-button handler failed: {e}")