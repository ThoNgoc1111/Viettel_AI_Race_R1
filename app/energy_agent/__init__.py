"""energy_agent package entrypoint.

Try to import the full `RLAgent` implementation. If that fails (for example
when MATLAB's embedded Python doesn't have torch available), provide a
lightweight fallback `RLAgent` class that implements the minimal interface
used by the MATLAB ESInterface. The fallback returns safe deterministic
actions (max power) so simulations avoid KPI violations and produce
non-zero energy readings.
"""
try:
	from .rl_agent import RLAgent
except Exception as _exc:  # import-time failure (torch missing, etc.)
	import logging
	import numpy as _np

	class RLAgent:
		def __init__(self, n_cells, n_ues, max_time, log_file='rl_agent.log', use_gpu=False):
			self.n_cells = int(n_cells)
			self.n_ues = int(n_ues)
			self.max_time = int(max_time)
			self.action_dim = self.n_cells
			self.training_mode = False
			self.deterministic_actions = True
			self.logger = logging.getLogger('PPOAgentFallback')
			self.logger.setLevel(logging.INFO)
			if not self.logger.handlers:
				ch = logging.StreamHandler()
				fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
				ch.setFormatter(fmt)
				self.logger.addHandler(ch)
			self.logger.info('Fallback RLAgent initialized due to import error in rl_agent: using safe deterministic actions')

		def setup_logging(self, *args, **kwargs):
			pass

		def start_scenario(self):
			self.logger.info('Fallback agent: start_scenario')

		def end_scenario(self):
			self.logger.info('Fallback agent: end_scenario')

		def get_action(self, state):
			return _np.ones(self.action_dim, dtype=float)

		def update(self, *args, **kwargs):
			return

		def save_model(self, filepath=None):
			self.logger.info('Fallback agent: save_model called (no-op)')

		def load_model(self, filepath, eval_mode=False):
			self.logger.info(f'Fallback agent: load_model called for {filepath} (no-op)')

		def set_evaluation_mode(self):
			self.training_mode = False
			self.deterministic_actions = True

		def set_training_mode(self, training):
			self.training_mode = bool(training)

		def get_stats(self):
			return {'total_episodes': 0, 'total_steps': 0, 'avg_reward': 0.0, 'buffer_size': 0}

__all__ = ["RLAgent"]