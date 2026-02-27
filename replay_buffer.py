"""
Optimized Replay Buffer for DreamerV3

Features:
- O(log n) episode sampling with cached probability distribution
- O(1) FIFO episode deletion using OrderedDict
- Background prefetch for zero-wait batch retrieval
- Async disk writes for non-blocking saves
"""

import collections
import io
import pathlib
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class EpisodeMeta:
    """Episode metadata for efficient indexing without loading full data"""
    key: str
    length: int  # Number of transitions (len(reward) - 1)
    timestamp: float
    file_path: Optional[pathlib.Path] = None


class SamplingCache:
    """
    Caches episode lengths and probability distribution for efficient sampling.

    Uses cumulative distribution + binary search for O(log n) sampling
    instead of O(n) probability reconstruction on each sample.
    """

    def __init__(self):
        self._keys: List[str] = []
        self._lengths: np.ndarray = np.array([], dtype=np.int64)
        self._probs: Optional[np.ndarray] = None
        self._cumsum: Optional[np.ndarray] = None
        self._dirty: bool = True
        self._key_to_idx: Dict[str, int] = {}
        self._removed_indices: set = set()  # Track removed indices for lazy cleanup

    def add_episode(self, key: str, length: int):
        """Add episode - O(1) operation, marks distribution as dirty"""
        self._keys.append(key)
        self._lengths = np.append(self._lengths, length)
        self._key_to_idx[key] = len(self._keys) - 1
        self._dirty = True

    def remove_episode(self, key: str):
        """Remove episode - marks for lazy cleanup"""
        if key in self._key_to_idx:
            idx = self._key_to_idx.pop(key)
            self._removed_indices.add(idx)
            self._dirty = True

    def _rebuild_if_dirty(self):
        """Lazily rebuild probability distribution when needed"""
        if not self._dirty:
            return

        # Compact arrays if too many removed
        if len(self._removed_indices) > len(self._keys) * 0.3:
            self._compact()

        # Build valid mask
        valid_mask = np.ones(len(self._lengths), dtype=bool)
        for idx in self._removed_indices:
            if idx < len(valid_mask):
                valid_mask[idx] = False

        valid_lengths = self._lengths[valid_mask]
        if len(valid_lengths) == 0:
            self._probs = np.array([])
            self._cumsum = np.array([])
            self._dirty = False
            return

        # Compute probability distribution weighted by length
        total = np.sum(valid_lengths)
        if total > 0:
            self._probs = valid_lengths.astype(np.float64) / total
            self._cumsum = np.cumsum(self._probs)
        else:
            self._probs = np.ones(len(valid_lengths)) / len(valid_lengths)
            self._cumsum = np.cumsum(self._probs)

        # Store valid indices for mapping
        self._valid_indices = np.where(valid_mask)[0]
        self._dirty = False

    def _compact(self):
        """Compact arrays by removing invalid entries"""
        valid_mask = np.ones(len(self._lengths), dtype=bool)
        for idx in self._removed_indices:
            if idx < len(valid_mask):
                valid_mask[idx] = False

        new_keys = []
        new_lengths = []
        new_key_to_idx = {}

        for i, (key, length) in enumerate(zip(self._keys, self._lengths)):
            if valid_mask[i]:
                new_key_to_idx[key] = len(new_keys)
                new_keys.append(key)
                new_lengths.append(length)

        self._keys = new_keys
        self._lengths = np.array(new_lengths, dtype=np.int64)
        self._key_to_idx = new_key_to_idx
        self._removed_indices.clear()

    def sample_episode_idx(self, rng: np.random.RandomState) -> Tuple[int, str]:
        """
        Sample episode using cumulative distribution + binary search - O(log n)

        Returns: (original_index, episode_key)
        """
        self._rebuild_if_dirty()

        if len(self._cumsum) == 0:
            raise ValueError("No episodes to sample from")

        u = rng.random()
        local_idx = np.searchsorted(self._cumsum, u)
        local_idx = min(local_idx, len(self._valid_indices) - 1)

        original_idx = self._valid_indices[local_idx]
        return original_idx, self._keys[original_idx]

    def __len__(self) -> int:
        return len(self._key_to_idx)


class EpisodeStore:
    """
    Efficient episode storage with O(1) FIFO deletion.

    Uses OrderedDict to maintain insertion order for FIFO eviction.
    """

    def __init__(self, max_steps: Optional[int] = None):
        self._episodes: collections.OrderedDict = collections.OrderedDict()
        self._max_steps = max_steps
        self._total_steps = 0

    def add(self, key: str, episode: Dict[str, np.ndarray]) -> List[str]:
        """
        Add episode, returns list of removed episode keys (FIFO eviction).

        Complexity: O(1) amortized for add, O(k) for removal where k = episodes removed
        """
        length = len(episode['reward']) - 1

        # Add to storage (moves to end if exists)
        if key in self._episodes:
            old_length = len(self._episodes[key]['reward']) - 1
            self._total_steps -= old_length

        self._episodes[key] = episode
        self._total_steps += length

        # FIFO eviction using OrderedDict's O(1) popitem
        removed = []
        while (self._max_steps is not None and
               self._total_steps > self._max_steps and
               len(self._episodes) > 1):
            oldest_key, oldest_ep = self._episodes.popitem(last=False)
            oldest_length = len(oldest_ep['reward']) - 1
            self._total_steps -= oldest_length
            removed.append(oldest_key)

        return removed

    def get(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        return self._episodes.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._episodes

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self):
        return iter(self._episodes)

    def keys(self):
        return self._episodes.keys()

    def values(self):
        return self._episodes.values()

    def items(self):
        return self._episodes.items()

    @property
    def total_steps(self) -> int:
        return self._total_steps


class AsyncEpisodeWriter:
    """Async disk writer - I/O doesn't block training"""

    def __init__(self, max_queue_size: int = 100):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._stopped = False
        self._thread.start()

    def _worker(self):
        while True:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:
                    break
                directory, ep_data, ep_id = item
                length = len(ep_data['reward'])
                filename = directory / f'{ep_id}-{length}.npz'
                with io.BytesIO() as f1:
                    np.savez_compressed(f1, **ep_data)
                    f1.seek(0)
                    with filename.open('wb') as f2:
                        f2.write(f1.read())
                self._queue.task_done()
            except queue.Empty:
                if self._stopped:
                    break
                continue
            except Exception as e:
                print(f"AsyncEpisodeWriter error: {e}")
                self._queue.task_done()

    def save(self, directory: pathlib.Path, episode: Dict[str, np.ndarray], ep_id: str):
        """Queue episode for async save"""
        try:
            self._queue.put_nowait((directory, episode, ep_id))
        except queue.Full:
            # Fallback to sync write
            length = len(episode['reward'])
            filename = directory / f'{ep_id}-{length}.npz'
            with io.BytesIO() as f1:
                np.savez_compressed(f1, **episode)
                f1.seek(0)
                with filename.open('wb') as f2:
                    f2.write(f1.read())

    def flush(self):
        """Wait for all queued writes to complete"""
        self._queue.join()

    def stop(self):
        """Stop background thread"""
        self._stopped = True
        self._queue.put(None)
        self._thread.join(timeout=5.0)


class PrefetchDataLoader:
    """
    Background thread prefetches batches, training gets data with zero wait.
    """

    def __init__(
        self,
        sample_fn,
        batch_size: int,
        batch_length: int,
        prefetch_batches: int = 4,
        seed: int = 0
    ):
        self._sample_fn = sample_fn
        self._batch_size = batch_size
        self._batch_length = batch_length
        self._queue = queue.Queue(maxsize=prefetch_batches)
        self._rng = np.random.RandomState(seed)

        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._worker.start()

    def _prefetch_worker(self):
        while not self._stop_event.is_set():
            try:
                batch = self._sample_batch()
                self._queue.put(batch, timeout=0.5)
            except queue.Full:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"Prefetch error: {e}")
                break

    def _sample_batch(self) -> Dict[str, np.ndarray]:
        batch_data = []
        for _ in range(self._batch_size):
            sequence = self._sample_fn(self._batch_length, self._rng)
            batch_data.append(sequence)

        return {
            key: np.stack([b[key] for b in batch_data], axis=0)
            for key in batch_data[0].keys()
        }

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        try:
            return self._queue.get(timeout=30.0)
        except queue.Empty:
            raise StopIteration("Prefetch queue timeout")

    def close(self):
        self._stop_event.set()
        self._worker.join(timeout=5.0)


class ReplayBuffer:
    """
    Optimized Replay Buffer for DreamerV3

    Features:
    - O(1) FIFO episode deletion using OrderedDict
    - O(log n) episode sampling with cached probability distribution
    - Background prefetch for zero-wait batch retrieval
    - Async disk writes for non-blocking saves

    Usage:
        buffer = ReplayBuffer(directory, max_steps=1000000, batch_size=16, batch_length=64)
        buffer.load_from_directory()

        # Get prefetch-enabled dataset iterator
        dataset = buffer.get_dataset()
        for batch in dataset:
            train(batch)

        # Add new episodes during training
        buffer.add_episode(ep_id, episode_data)
    """

    def __init__(
        self,
        directory: pathlib.Path,
        max_steps: Optional[int] = None,
        batch_size: int = 16,
        batch_length: int = 64,
        prefetch_batches: int = 4,
        seed: int = 0,
    ):
        self._directory = pathlib.Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

        self._max_steps = max_steps
        self._batch_size = batch_size
        self._batch_length = batch_length
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        # Core components
        self._store = EpisodeStore(max_steps)
        self._sampling_cache = SamplingCache()
        self._async_writer = AsyncEpisodeWriter()

        # Prefetch loader (initialized lazily)
        self._prefetch_loader: Optional[PrefetchDataLoader] = None
        self._prefetch_batches = prefetch_batches

    def add_episode(
        self,
        episode_id: str,
        episode: Dict[str, Any],
        save_to_disk: bool = True
    ) -> List[str]:
        """
        Add episode to buffer.

        Args:
            episode_id: Unique episode identifier
            episode: Dict of numpy arrays or lists
            save_to_disk: Whether to persist to disk

        Returns:
            List of removed episode IDs (due to FIFO eviction)
        """
        # Convert lists to arrays if needed
        ep_data = {}
        for k, v in episode.items():
            if k.startswith('log_'):
                continue
            ep_data[k] = np.array(v) if isinstance(v, list) else v

        # Add to store
        removed = self._store.add(episode_id, ep_data)

        # Update sampling cache
        length = len(ep_data['reward']) - 1
        self._sampling_cache.add_episode(episode_id, length)

        for key in removed:
            self._sampling_cache.remove_episode(key)

        # Async save
        if save_to_disk:
            self._async_writer.save(self._directory, ep_data, episode_id)

        return removed

    def load_from_directory(self, limit: Optional[int] = None, reverse: bool = True):
        """
        Load episodes from directory.

        Args:
            limit: Maximum total steps to load
            reverse: Load newest episodes first
        """
        total = 0
        filenames = sorted(self._directory.glob("*.npz"), reverse=reverse)

        for filepath in filenames:
            try:
                with filepath.open('rb') as f:
                    data = np.load(f)
                    episode = {k: data[k] for k in data.keys()}
            except Exception as e:
                print(f"Could not load episode {filepath}: {e}")
                continue

            # Parse episode ID from filename
            filename = filepath.stem
            parts = filename.rsplit('-', 1)
            ep_id = parts[0] if len(parts) == 2 else filename

            # Add to store and cache
            length = len(episode['reward']) - 1
            self._store._episodes[ep_id] = episode
            self._store._total_steps += length
            self._sampling_cache.add_episode(ep_id, length)

            total += length
            if limit and total >= limit:
                break

    def sample_sequence(
        self,
        length: int,
        rng: Optional[np.random.RandomState] = None
    ) -> Dict[str, np.ndarray]:
        """
        Sample a sequence of given length.

        May span multiple episodes, correctly sets is_first flags.
        Compatible with original sample_episodes behavior.
        """
        rng = rng or self._rng

        ret = None
        size = 0

        while size < length:
            # O(log n) episode sampling
            _, key = self._sampling_cache.sample_episode_idx(rng)
            episode = self._store.get(key)

            if episode is None:
                continue

            total = len(next(iter(episode.values())))
            if total < 2:
                continue

            if ret is None:
                # First segment
                index = int(rng.randint(0, total - 1))
                ret = {
                    k: v[index:min(index + length, total)].copy()
                    for k, v in episode.items()
                    if not k.startswith('log_')
                }
                if 'is_first' in ret:
                    ret['is_first'][0] = True
            else:
                # Append from beginning of new episode
                index = 0
                possible = length - size
                ret = {
                    k: np.concatenate([
                        ret[k],
                        v[index:min(index + possible, total)].copy()
                    ], axis=0)
                    for k, v in episode.items()
                    if not k.startswith('log_')
                }
                if 'is_first' in ret:
                    ret['is_first'][size] = True

            size = len(next(iter(ret.values())))

        return ret

    def get_dataset(self) -> PrefetchDataLoader:
        """
        Get prefetch-enabled dataset iterator.

        Usage:
            dataset = buffer.get_dataset()
            for batch in dataset:
                train(batch)
        """
        if self._prefetch_loader is not None:
            self._prefetch_loader.close()

        self._prefetch_loader = PrefetchDataLoader(
            sample_fn=self.sample_sequence,
            batch_size=self._batch_size,
            batch_length=self._batch_length,
            prefetch_batches=self._prefetch_batches,
            seed=self._seed
        )
        return self._prefetch_loader

    def as_dict(self) -> collections.OrderedDict:
        """
        Return OrderedDict for compatibility with existing simulate/make_dataset.

        This allows gradual migration - existing code can use buffer.as_dict()
        where it previously used train_eps directly.
        """
        return self._store._episodes

    @property
    def total_steps(self) -> int:
        return self._store.total_steps

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __getitem__(self, key: str) -> Dict[str, np.ndarray]:
        ep = self._store.get(key)
        if ep is None:
            raise KeyError(key)
        return ep

    def __setitem__(self, key: str, episode: Dict[str, Any]):
        self.add_episode(key, episode, save_to_disk=False)

    def keys(self):
        return self._store.keys()

    def values(self):
        return self._store.values()

    def items(self):
        return self._store.items()

    def pop(self, key: str, *args):
        """Remove and return episode"""
        if key in self._store._episodes:
            ep = self._store._episodes.pop(key)
            length = len(ep['reward']) - 1
            self._store._total_steps -= length
            self._sampling_cache.remove_episode(key)
            return ep
        if args:
            return args[0]
        raise KeyError(key)

    def popitem(self, last: bool = True):
        """Remove and return (key, episode) pair"""
        if len(self._store._episodes) == 0:
            raise KeyError("buffer is empty")
        key, ep = self._store._episodes.popitem(last=last)
        length = len(ep['reward']) - 1
        self._store._total_steps -= length
        self._sampling_cache.remove_episode(key)
        return key, ep

    def flush(self):
        """Wait for all async writes to complete"""
        self._async_writer.flush()

    def close(self):
        """Clean up resources"""
        if self._prefetch_loader:
            self._prefetch_loader.close()
        self._async_writer.flush()
        self._async_writer.stop()


# Compatibility functions for gradual migration

def create_replay_buffer(
    directory: pathlib.Path,
    config,
) -> ReplayBuffer:
    """
    Factory function to create ReplayBuffer from config.

    Usage:
        buffer = create_replay_buffer(config.traindir, config)
        buffer.load_from_directory(limit=config.dataset_size)
    """
    return ReplayBuffer(
        directory=directory,
        max_steps=getattr(config, 'dataset_size', None),
        batch_size=getattr(config, 'batch_size', 16),
        batch_length=getattr(config, 'batch_length', 64),
        prefetch_batches=4,
        seed=getattr(config, 'seed', 0),
    )
