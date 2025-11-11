import numpy as np
import os
from collections import deque
import logging
from datetime import datetime

from .transition import Transition, TransitionBuffer
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
from .models import Actor, Critic
from .state_normalizer import StateNormalizer


class RLAgent:
    def __init__(self, n_cells, n_ues, max_time, log_file='rl_agent.log', use_gpu=False):
        print("Initializing RL Agent")
        self.n_cells = n_cells
        self.n_ues = n_ues
        self.max_time = max_time
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # State dimensions
        self.state_dim = 17 + 14 + (n_cells * 12)
        self.action_dim = n_cells

        # Normalizer
        self.state_normalizer = StateNormalizer(self.state_dim, n_cells=n_cells)

        # Networks
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim=256).to(self.device)
        self.critic = Critic(self.state_dim, hidden_dim=256).to(self.device)

        # Use slightly more conservative learning rates to reduce training instability
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-4)

        # PPO hyperparameters
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.clip_epsilon = 0.2
        # Smaller number of epochs and larger batch size to stabilize updates
        self.ppo_epochs = 10
        self.batch_size = 256
        self.buffer_size = 4096

        # Experience buffer
        self.buffer = TransitionBuffer(self.buffer_size)

        # Training state
        self.training_mode = True
        # When True, actions are sampled (training/exploration). When False, deterministic actions are used.
        self.total_episodes = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = 0
        self.current_episode_reward = 0.0

        # Violation tracking
        self.violation_counts = {'drop': 0, 'latency': 0, 'cpu': 0, 'prb': 0}
        self.any_violation = False
        # When set, force deterministic (mean) actions even if training_mode toggles
        self.deterministic_actions = False

        self.setup_logging(log_file)
        self.logger.info(f"PPO Agent initialized: {n_cells} cells, {n_ues} UEs")

        # Auto-load pretrained model if MODEL_PATH env var is set (keeps runtime flexible)
        # current saved trained final model absolute path
        agent_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_name = 'final_model.pth' # <<< replace with final submission model name if different
        submission_model_path = os.path.join(agent_dir, model_name)

        agent_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- MODIFIED FOR TRAINING ---
        self.logger.warning("Agent is starting in FORCED TRAINING mode.")
        self.set_training_mode(True)
        # --- END OF MODIFICATION ---

        model_name = 'final_model.pth' # <<< replace with final submission model name if different
        submission_model_path = os.path.join(agent_dir, model_name)
        
        if os.path.isfile(submission_model_path):
            self.logger.info(f"Loading SUBMISSION model from {submission_model_path}")
        try:
        # Load model and set evaluation_mode=True
            self.load_model(submission_model_path, eval_mode=True)
        except Exception as e:
            self.logger.error(f"FATAL: Failed to load submission model: {e}")
        # If cannot load, still set eval mode
            self.set_evaluation_mode()
        else:
        self.logger.error(f"FATAL: SUBMISSION MODEL NOT FOUND at {submission_model_path}")
        # Nếu không tìm thấy model, vẫn set eval mode
            self.set_evaluation_mode()

    def setup_logging(self, log_file):
        self.logger = logging.getLogger('PPOAgent')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        self.logger.addHandler(fh); self.logger.addHandler(ch)

    def normalize_state(self, state):
        return self.state_normalizer.normalize(state)

    def start_scenario(self):
        self.total_episodes += 1
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        self.violation_counts = {k: 0 for k in self.violation_counts}
        self.any_violation = False
        self.logger.info(f"Starting episode {self.total_episodes}")

    def end_scenario(self):
        if self.any_violation:
            self.logger.warning(f"Episode {self.total_episodes} had KPI violations: {self.violation_counts}")
            # Reduce reward instead of wiping it out
            self.current_episode_reward *= 0.5

        self.episode_rewards.append(self.current_episode_reward)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        energy_used = getattr(self, 'total_energy_consumed', 0.0)
        self.logger.info(f"Episode {self.total_episodes} ended: Steps={self.episode_steps}, "
                        f"Reward={self.current_episode_reward:.2f}, Avg100={avg_reward:.2f}, "
                        f"Energy={energy_used:.3f} kWh, Violations={self.violation_counts}")

        if self.training_mode and len(self.buffer) >= self.batch_size:
            self.train()

        # Auto-save a checkpoint periodically so training runs inside containers
        # can persist models to a mounted folder. Save every 10 episodes.
        try:
            if self.training_mode and (self.total_episodes % 10 == 0):
                save_dir = '/app/energy_agent/models'
                # ensure directory exists if running inside container
                import os as _os
                if _os.path.isdir(save_dir):
                    save_path = _os.path.join(save_dir, f'ppo_autosave_ep{self.total_episodes}.pth')
                    self.save_model(save_path)
                    self.logger.info(f'Auto-saved model to {save_path}')
        except Exception as _:
            # Non-fatal if saving fails (e.g., not running in container or permission issues)
            pass


    # ---------- Safety Layer + Action ----------
    def get_action(self, state):
        raw_state = np.array(state).flatten()
        state = self.normalize_state(raw_state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_mean, action_logstd = self.actor(state_tensor)
            # If we're in training mode but deterministic_actions is requested, use mean.
            if self.training_mode and not getattr(self, 'deterministic_actions', False):
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            else:
                # deterministic evaluation or non-training -> use mean
                action = action_mean
                log_prob = torch.zeros(1).to(self.device)

        action = torch.clamp(action, 0.0, 1.0)
        action_np = action.cpu().numpy().flatten()

        # Safety layer
        drop_ratio = state[19]; lat_ratio = state[20]
        cpu_ratio = state[26]; prb_ratio = state[27]
        if drop_ratio > 0.9 or lat_ratio > 0.9:
            action_np = np.minimum(action_np + 0.05, 1.0)
        if cpu_ratio > 0.9 or prb_ratio > 0.9:
            prev = getattr(self, 'last_action', action_np)
            action_np = np.maximum(action_np, prev)

        # store raw / un-normalized state for reward calculations and diagnostics
        self.last_state = raw_state.copy()
        self.last_action = action_np.copy()
        # store scalar log-prob for stability
        try:
            self.last_log_prob = float(log_prob.cpu().numpy().flatten()[0])
        except Exception:
            # fallback if already scalar
            try:
                self.last_log_prob = float(log_prob)
            except Exception:
                self.last_log_prob = 0.0
        return action_np

    # ---------- Reward with Curriculum ----------
    def get_violation_penalty_scale(self):
        # Ramp penalty earlier so the agent prioritizes KPI avoidance sooner
        progress = min(1.0, self.total_episodes / 50.0)
        penalty_scale = 0.4 + 0.6 * progress
        if self.episode_steps == 0:
            self.logger.info(f"Episode {self.total_episodes}: penalty scale = {penalty_scale:.2f}")
        return penalty_scale

    def calculate_reward(self, prev_state, action, current_state):
        if prev_state is None:
            return 0.0
        prev = np.array(prev_state).flatten()
        curr = np.array(current_state).flatten()
        # energy (raw units expected at index 17)
        prev_energy = prev[17] if len(prev) > 17 else 0.0
        curr_energy = curr[17] if len(curr) > 17 else 0.0

        # Use normalized ratios for penalty/threshold reasoning (0..1)
        curr_norm = self.normalize_state(curr)
        drop_norm = curr_norm[19] if len(curr_norm) > 19 else 0.0
        lat_norm = curr_norm[20] if len(curr_norm) > 20 else 0.0
        cpu_norm = curr_norm[26] if len(curr_norm) > 26 else 0.0
        prb_norm = curr_norm[27] if len(curr_norm) > 27 else 0.0

        penalty_scale = self.get_violation_penalty_scale()

        # Hard violation if normalized KPI goes beyond near-1 (saturated/invalid)
        if drop_norm >= 0.995 or lat_norm >= 0.995 or cpu_norm >= 0.995 or prb_norm >= 0.995:
            self.any_violation = True
            return -100.0 * penalty_scale # Phạt nặng nếu vi phạm hoàn toàn

        # Increase reward (ví dụ: * 500.0)
        energy_delta = prev_energy - curr_energy
        energy_term = energy_delta * 500.0 

        # Use square reward functon (smoother but grows faster)
        # Penalize when exceeding 85%
        threshold = 0.85
        penalty = 0.0
        # Update better penalty coefficients
        penalty += 200.0 * (max(0.0, drop_norm - threshold) ** 2)
        penalty += 200.0 * (max(0.0, lat_norm - threshold) ** 2)
        # Phạt tài nguyên có thể nhẹ hơn, ví dụ 80.0
        penalty += 80.0 * (max(0.0, cpu_norm - threshold) ** 2)
        penalty += 80.0 * (max(0.0, prb_norm - threshold) ** 2)
        
        penalty *= penalty_scale

        # Smoothness: penalize big changes from previous action (if available)
        prev_action = getattr(self, 'last_action', action)
        action_smooth_pen = 5.0 * np.mean(np.abs(action - prev_action)) # keep small

        reward = energy_term - penalty - action_smooth_pen
        return float(np.clip(reward, -500.0, 500.0)) # Clip reward

    # ---------- Update ----------
    def update(self, state, action, next_state, done):
        if not self.training_mode:
            return
        reward = self.calculate_reward(state, action, next_state)
        self.episode_steps += 1
        self.total_steps += 1
        self.current_episode_reward += reward

        # Keep a record of the latest total energy reading (raw value at index 17)
        try:
            next_raw = np.array(next_state).flatten()
            if len(next_raw) > 17:
                # store as kWh/raw unit as passed by the simulation state
                self.total_energy_consumed = float(next_raw[17])
            else:
                # preserve previous if index missing
                self.total_energy_consumed = getattr(self, 'total_energy_consumed', 0.0)
        except Exception:
            self.total_energy_consumed = getattr(self, 'total_energy_consumed', 0.0)

        s = self.normalize_state(np.array(state).flatten())
        ns = self.normalize_state(np.array(next_state).flatten())
        a = np.array(action).flatten()

        with torch.no_grad():
            v = self.critic(torch.FloatTensor(s).unsqueeze(0).to(self.device)).cpu().numpy().item()

        transition = Transition(
            state=s, action=a, reward=reward, next_state=ns,
            done=done, log_prob=float(getattr(self, 'last_log_prob', 0.0)), value=v
        )
        self.buffer.add(transition)

        # Check KPI violations for counters/flag
        self.check_kpi_violations(next_state)

        # Optional: train incrementally
        if self.training_mode and len(self.buffer) >= self.batch_size:
            self.train()

    # ---------- KPI Violation Tracking ----------
    def check_kpi_violations(self, state):
        # Use normalized values for consistent thresholds
        s = np.array(state).flatten()
        s_norm = self.normalize_state(s)
        # Count a violation when normalized KPI passes a strict threshold (95%)
        if len(s_norm) > 19 and s_norm[19] > 0.95:
            self.violation_counts['drop'] += 1
            self.any_violation = True
        if len(s_norm) > 20 and s_norm[20] > 0.95:
            self.violation_counts['latency'] += 1
            self.any_violation = True
        if len(s_norm) > 26 and s_norm[26] > 0.95:
            self.violation_counts['cpu'] += 1
            self.any_violation = True
        if len(s_norm) > 27 and s_norm[27] > 0.95:
            self.violation_counts['prb'] += 1
            self.any_violation = True

    # ---------- KPI Snapshot Logging ----------
    def log_kpi_snapshot(self, state, step, timestep_seconds=1):
        if (step * timestep_seconds) % 10 != 0:
            return
        s = np.array(state).flatten()
        self.logger.info(
            f"KPI snapshot @ {step*timestep_seconds}s: "
            f"drop={s[19]:.3f}, lat={s[20]:.3f}, "
            f"cpu={s[26]:.3f}, prb={s[27]:.3f}, "
            f"energy={s[17]:.3f}"
        )

    # ---------- PPO Training ----------
    def compute_gae(self, rewards, values, next_values, dones):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            next_value = float(next_values[t]) if t < len(next_values) else 0.0
            delta = float(rewards[t]) + self.gamma * next_value * nonterminal - float(values[t])
            last_adv = delta + self.gamma * self.lambda_gae * nonterminal * last_adv
            advantages[t] = last_adv
        returns = advantages + values
        return advantages, returns

    def train(self):
        transitions = self.buffer.get_all()
        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions])
        old_log_probs = np.array([t.log_prob for t in transitions])
        values = np.array([t.value for t in transitions])

        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        with torch.no_grad():
            next_values = self.critic(next_states_tensor).cpu().numpy().flatten()

        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_idx = indices[start:end]

                batch_states = states_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_log_probs = old_log_probs_tensor[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]

                action_mean, action_logstd = self.actor(batch_states)
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                current_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(current_values, batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

        self.buffer.clear()
        self.logger.info(f"Training completed: Actor loss={actor_loss:.4f}, Critic loss={critic_loss:.4f}")

    # ---------- Save/Load Model ----------
    def save_model(self, filepath=None):
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ppo_model_{timestamp}.pth"
        
        # Get normalizer state
        normalizer_state = {}
        if hasattr(self, 'state_normalizer') and hasattr(self.state_normalizer, 'get_state'):
             normalizer_state = self.state_normalizer.get_state()
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'state_normalizer_state': normalizer_state,  # <-- ĐÃ THÊM VÀO
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath, eval_mode=False):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        # load normalizer state if available
        if hasattr(self, 'state_normalizer') and 'state_normalizer_state' in checkpoint and hasattr(self.state_normalizer, 'set_state'):
            if checkpoint['state_normalizer_state']: # Only load if not empty
                self.state_normalizer.set_state(checkpoint['state_normalizer_state'])
                self.logger.info("StateNormalizer state loaded.")
            else:
                self.logger.warning("StateNormalizer state was empty in checkpoint, not loading.")
        else:
            self.logger.warning("StateNormalizer state not found in checkpoint or set_state not available.")
        

        # load optimizers state if available, otherwise keep existing optimizers
        try:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        except Exception:
            self.logger.info("Optimizer state not found or incompatible; skipping optimizer state load.")
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.logger.info(f"Model loaded from {filepath}")
        if eval_mode:
            self.set_evaluation_mode()

    def set_evaluation_mode(self):
        """Configure agent for deterministic evaluation: disable training updates and use mean actions."""
        self.training_mode = False
        self.deterministic_actions = True
        self.actor.eval()
        self.critic.eval()
        self.logger.info("Agent set to evaluation mode: deterministic actions, training disabled.")

    def set_training_mode(self, training):
        self.training_mode = training
        self.actor.train(training)
        self.critic.train(training)
        self.logger.info(f"Training mode set to {training}")

    def get_stats(self):
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'avg_reward': avg_reward,
            'buffer_size': len(self.buffer),
            'training_mode': self.training_mode,
            'episode_steps': self.episode_steps,
            'current_episode_reward': self.current_episode_reward,
            'violations': self.violation_counts,
            'any_violation': self.any_violation
        }

