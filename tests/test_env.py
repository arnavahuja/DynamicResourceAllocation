import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment import config
from environment.cluster_env import CloudClusterEnv
from environment.job import Job
from environment.power_model import compute_power
from environment.server import Server
from environment.workload.synthetic import SyntheticWorkloadGenerator


class TestPowerModel:
    def test_idle_power(self):
        p = compute_power(0.0)
        assert abs(p - config.P_IDLE) < 1e-6

    def test_max_power(self):
        p = compute_power(1.0)
        assert abs(p - config.P_MAX) < 1e-6

    def test_monotonic(self):
        powers = [compute_power(u / 10.0) for u in range(11)]
        for i in range(len(powers) - 1):
            assert powers[i] <= powers[i + 1]


class TestServer:
    def test_assign_and_utilization(self):
        server = Server(server_id=0)
        job = Job(0, cpu_request=0.5, mem_request=0.3, duration=3, arrival_time=0, sla_deadline=6)
        assert server.can_fit(job)
        server.assign(job)
        assert abs(server.cpu_utilization - 0.5) < 1e-6
        assert abs(server.mem_utilization - 0.3) < 1e-6

    def test_cannot_overfit(self):
        server = Server(server_id=0)
        job1 = Job(0, cpu_request=0.7, mem_request=0.5, duration=3, arrival_time=0, sla_deadline=6)
        job2 = Job(1, cpu_request=0.5, mem_request=0.5, duration=3, arrival_time=0, sla_deadline=6)
        server.assign(job1)
        assert not server.can_fit(job2)

    def test_step_completes_jobs(self):
        server = Server(server_id=0)
        job = Job(0, cpu_request=0.5, mem_request=0.3, duration=1, arrival_time=0, sla_deadline=5)
        server.assign(job)
        completed = server.step()
        assert len(completed) == 1
        assert completed[0].job_id == 0
        assert abs(server.cpu_utilization) < 1e-6

    def test_reset(self):
        server = Server(server_id=0)
        job = Job(0, cpu_request=0.5, mem_request=0.3, duration=3, arrival_time=0, sla_deadline=6)
        server.assign(job)
        server.reset()
        assert abs(server.cpu_utilization) < 1e-6
        assert len(server.running_jobs) == 0


class TestSyntheticWorkload:
    def test_generates_jobs(self):
        gen = SyntheticWorkloadGenerator(arrival_rate=3.0, seed=42)
        gen.reset()
        jobs = gen.get_next_jobs(0)
        assert isinstance(jobs, list)
        assert all(isinstance(j, Job) for j in jobs)

    def test_exhaustion(self):
        gen = SyntheticWorkloadGenerator(max_jobs=5, arrival_rate=10.0, seed=42)
        gen.reset()
        total = 0
        for t in range(100):
            jobs = gen.get_next_jobs(t)
            total += len(jobs)
            if gen.is_exhausted():
                break
        assert total <= 5


class TestCloudClusterEnv:
    def test_reset_obs_shape(self):
        env = CloudClusterEnv()
        obs, info = env.reset()
        expected_dim = 4 * config.NUM_SERVERS + 4 * config.JOB_QUEUE_SIZE
        assert obs.shape == (expected_dim,)
        assert obs.dtype == np.float32

    def test_step_returns_correct_tuple(self):
        env = CloudClusterEnv()
        obs, info = env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_truncates(self):
        env = CloudClusterEnv(episode_length=10)
        env.reset()
        truncated = False
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if truncated or terminated:
                break
        assert truncated

    def test_info_keys(self):
        env = CloudClusterEnv()
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "power" in info
        assert "sla_violations" in info
        assert "jobs_completed" in info
        assert "queue_length" in info
        assert "step_power" in info

    def test_random_rollout(self):
        """Full random rollout should complete without errors."""
        env = CloudClusterEnv(episode_length=100)
        obs, info = env.reset()
        total_reward = 0.0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        assert info["timestep"] > 0
        assert isinstance(total_reward, float)

    def test_sla_violation_detection(self):
        """Jobs that wait too long should trigger SLA violations."""
        # Use a workload with jobs that have very short SLA deadlines
        gen = SyntheticWorkloadGenerator(
            arrival_rate=5.0,
            duration_range=(2, 3),
            sla_multiplier=1.0,  # deadline = duration, very tight
            seed=123,
        )
        env = CloudClusterEnv(
            num_servers=1,  # only 1 server to create congestion
            workload_generator=gen,
            episode_length=50,
        )
        env.reset()
        # Deliberately pick bad actions to cause violations
        for _ in range(50):
            _, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        # With 1 server and many arrivals, SLA violations are expected
        assert info["sla_violations"] > 0
