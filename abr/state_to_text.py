from openai import OpenAI
import openai
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import Tuple
from global_constants import (
    PROMPT_FILE,
    STATE_SAVE_PATH,
    TEST_STATE_SAVE_PATH,
    NUM_TEST_STATES,
    STATE_DESCRIPTION_SAVE_PATH,
    MAX_NUM_STATES,
    LLM_MODEL,
    CONCEPTS_FILE,
    N_QUERY_TOGETHER,
    N_ACTIONS,
)
from multiprocessing.dummy import Pool
from time import sleep
import warnings

with open(PROMPT_FILE, "r") as f:
    PROMPT = "".join(f.readlines())

with open(CONCEPTS_FILE, "r") as f:
    CONCEPTS = "".join(f.readlines())

CUSTOM_INSTRUCTIONS = ("You are a computer scientist trying to gather "
                       "key information to use in an embedding model"
                       "to identify patterns."
                       "Be straight to the point and avoid unnecessary words.")

HISTORY_LEN = 10
FUTURE_CHUNK_LEN = 5
HISTORY_FEATURES = ["Selected Video Quality (SSIM dB), with a max of 25", 
                    "Selected Chunk Size (Mb), with a max of 3",
                    "Transmission Time of Chunk (seconds), with a max of 20", 
                    "Client Buffer (seconds), with a max of 15", 
                    "Quality of Experience, with a max of 5", 
                    "Stalling (seconds), with a max of 3"]
STATE_DIM = (HISTORY_LEN, len(HISTORY_FEATURES) + N_ACTIONS + N_ACTIONS)

MAX_QUALITY = 25
MAX_VIDEO_SIZE = 3
MIN_HISTORY_STATE = np.array([0, 0, 0, 0, -5, 0], dtype=np.float32)
MAX_HISTOR_STATE = np.array([MAX_QUALITY, MAX_VIDEO_SIZE, 20, 15, 5, 3], dtype=np.float32)


def unnormalize(obs: np.ndarray) -> Tuple[np.ndarray]:
    """Reverse normalization on a raw stored ABR state tensor.

    Parameters
    ----------
    obs : np.ndarray
        Normalized state array of shape ``(T, F)`` where the first
        ``MIN_HISTORY_STATE.shape[0]`` columns are historical features
        followed by future video size predictions and future SSIM
        (quality) predictions. The temporal dimension ``T`` should be
        at least ``HISTORY_LEN``; only the first ``FUTURE_CHUNK_LEN``
        rows are used for future predictions.

    Returns
    -------
    tuple of np.ndarray
        ``(history_data, throughput_history, ssim_data, video_sizes)``
        where:
        * history_data : denormalized historical features, shape ``(T, H)``
        * throughput_history : derived throughput Mbps, shape ``(T,)``
        * ssim_data : denormalized SSIM qualities, shape ``(FUTURE_CHUNK_LEN, N_ACTIONS)``
        * video_sizes : denormalized video sizes (Mb), shape ``(FUTURE_CHUNK_LEN, N_ACTIONS)``

    Notes
    -----
    Throughput is computed as size / download_time with invalid / zero
    times set to 0 and clipped to a max of 3 Mbps.
    """
    history_data = obs[:, :MIN_HISTORY_STATE.shape[0]]
    video_sizes = obs[0:FUTURE_CHUNK_LEN, 
                    MIN_HISTORY_STATE.shape[0]: MIN_HISTORY_STATE.shape[0] + N_ACTIONS]
    ssim_data = obs[0:FUTURE_CHUNK_LEN, 
                      MIN_HISTORY_STATE.shape[0] + N_ACTIONS: MIN_HISTORY_STATE.shape[0] + N_ACTIONS + N_ACTIONS]
    history_data = (history_data * (MAX_HISTOR_STATE - MIN_HISTORY_STATE)) + MIN_HISTORY_STATE
    with np.errstate(divide='ignore', invalid='ignore'):
        throughput_history = history_data[:, 1] / history_data[:, 2]
        throughput_history[history_data[:, 2] <= 0] = 0.
    throughput_history = np.clip(throughput_history, 0, 3)
    ssim_data = ssim_data * MAX_QUALITY
    video_sizes = video_sizes * MAX_VIDEO_SIZE
    return (history_data, throughput_history, ssim_data, video_sizes)


def state_to_str(state: np.ndarray) -> str:
    """Convert a single normalized state to a human-readable string.

    Builds a multi-line textual description enumerating historical
    features, derived throughput, and weighted upcoming video qualities
    and sizes. This textual representation is used as input to the LLM
    for generating higher-level natural language descriptions.

    Parameters
    ----------
    state : np.ndarray
        Normalized state array with shape consistent with ``STATE_DIM``.

    Returns
    -------
    str
        Human-readable multi-line description of the state.
    """
    state_description = []
    historical_data, throughput_history, ssim_data, video_sizes = unnormalize(state)
    for feature_idx, feature_name in enumerate(HISTORY_FEATURES):
        feature_description = f"{feature_name}: ["
        for val in historical_data.T[feature_idx]:
            feature_description += f"{val:.3f}, "
        feature_description += "]"
        state_description.append(feature_description)
    throughput_description = f"Network Throughput (Mbps), with a max of 3: ["
    for val in throughput_history:
        throughput_description += f"{val:.3f}, "
    throughput_description += "]"
    state_description.insert(3, throughput_description)
    for feature_name, feature_data in [
            ["Mean Upcoming Video Qualities (SSIM dB), with a max of 25", ssim_data],
            ["Mean Upcoming Video Sizes (Mb), with a max of 3", video_sizes]]:
        feature_description = f"{feature_name}: ["
        weights = 1. / (np.arange(feature_data.shape[1]) + 2)
        weights = weights / weights.sum()
        avg = np.average(feature_data, weights=weights, axis=1)
        for val in avg:
            feature_description += f"{val:.3f}, "
        feature_description += "]"
        state_description.append(feature_description)
    state_description = "\n\n\t\t\t".join(state_description)
    return state_description

def get_llm_description(state: np.ndarray, client: OpenAI) -> str:
    """Query the LLM for a natural language description of a state.

    Parameters
    ----------
    state : np.ndarray
        Normalized ABR state.
    client : OpenAI
        OpenAI client used to perform the chat completion request.

    Returns
    -------
    str
        Model generated description grounded on provided concepts
        and the structured state serialization.
    """
    prompt = PROMPT.format(concepts = CONCEPTS, state_data = state_to_str(state=state))
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": CUSTOM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        frequency_penalty=None,
        presence_penalty=None,
        
    )
    message = response.choices[0].message.content
    return message
    
def llm_wrapper(data: Tuple[int, np.ndarray, OpenAI]) -> Tuple[int, str]:
    """Wrapper for parallel LLM description generation with retry.

    Parameters
    ----------
    data : tuple
        ``(state_idx, state, client)`` where ``state_idx`` is an
        integer index, ``state`` is a normalized state array and
        ``client`` an OpenAI client instance.

    Returns
    -------
    tuple
        ``(state_idx, description)``. The description is either loaded
        from disk if already present or generated via the LLM.

    Notes
    -----
    Retries on rate limit / timeout errors with exponential-ish delays.
    """
    state_idx, state, client = data
    msg = None
    description_file = STATE_DESCRIPTION_SAVE_PATH / f"state_{state_idx:07d}.txt"
    if description_file.exists():
        msg = description_file.read_text().rstrip() 
    while msg is None:
        try:
            msg = get_llm_description(state=state, client=client)
            sleep(0.5)
        except (openai.RateLimitError, openai.APITimeoutError):
            sleep(1.5)
    return (state_idx, msg)



def load_dataset(
    n_samples: int = MAX_NUM_STATES,
    n_test_samples: int = NUM_TEST_STATES,
    disable_progress: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load stored states for trustee / model training.

    Reads pre-saved state/action pairs from ``STATE_SAVE_PATH`` and
    ``TEST_STATE_SAVE_PATH``. A lightweight wrapper so upper-level
    scripts don't need to parse raw traces.

    Parameters
    ----------
    n_samples : int, optional
        Maximum number of training states to load (default: MAX_NUM_STATES).
    n_test_samples : int, optional
        Maximum number of test states to load (default: NUM_TEST_STATES).
    disable_progress : bool, optional
        Unused; preserved for interface compatibility.

    Returns
    -------
    tuple
        ``((train_states, train_actions, train_trace_indices),
        (test_states, test_actions, test_trace_indices))`` where each is
        a numpy array. Shapes:

        * train_states : (N_train, *STATE_DIM)
        * train_actions : (N_train,)
        * train_trace_indices : (N_train,)
        * test_states : (N_test, *STATE_DIM)
        * test_actions : (N_test,)
        * test_trace_indices : (N_test,)
    """

    train_files = sorted(STATE_SAVE_PATH.glob("*.npz"))[:n_samples]
    states = np.zeros(((len(train_files),) + STATE_DIM), dtype=np.float32)
    actions = np.zeros(len(train_files), dtype=np.int64)
    train_trace_indices = np.zeros(len(train_files), dtype=np.int64)
    for idx, file in enumerate(train_files):
        data = np.load(file)
        states[idx] = data["state"]
        actions[idx] = int(data["action"])
        train_trace_indices[idx] = int(data.get("trace_idx", -1))

    test_files = sorted(TEST_STATE_SAVE_PATH.glob("*.npz"))[:n_test_samples]
    test_states = np.zeros(((len(test_files),) + STATE_DIM), dtype=np.float32)
    test_actions = np.zeros(len(test_files), dtype=np.int64)
    test_trace_indices = np.zeros(len(test_files), dtype=np.int64)
    for idx, file in enumerate(test_files):
        data = np.load(file)
        test_states[idx] = data["state"]
        test_actions[idx] = int(data["action"])
        test_trace_indices[idx] = int(data.get("trace_idx", -1))

    if len(train_files) < n_samples:
        warnings.warn(
            f"Only managed to find {len(train_files)} samples with current config."
        )
    if len(test_files) < n_test_samples:
        warnings.warn(
            f"Only managed to find {len(test_files)} test samples with current config."
        )

    return (
        (states, actions, train_trace_indices),
        (test_states, test_actions, test_trace_indices),
    )


def save_state_descriptions() -> None:
    """Generate and persist LLM state descriptions for training data.

    For each stored state file, queries the LLM (if a description file
    does not already exist) and saves the textual description under
    ``STATE_DESCRIPTION_SAVE_PATH`` using the pattern ``state_XXXXXXX.txt``.

    Notes
    -----
    Parallelizes requests with a thread pool sized by ``N_QUERY_TOGETHER``.
    """
    client = OpenAI()
    train_data, test_data = load_dataset(n_samples= MAX_NUM_STATES)
    states, actions, train_trace_indices = train_data
    pool = Pool(processes= N_QUERY_TOGETHER)
    inputs = [[state_idx, state, client] for state_idx, state in enumerate(states)]
    with tqdm(total=len(states), leave=False, desc="Querying LLM") as pbar:
        for state_idx, description in pool.imap_unordered(llm_wrapper, inputs):
            description_file = STATE_DESCRIPTION_SAVE_PATH / f"state_{state_idx:07d}.txt"
            description_file.parent.mkdir(parents=True, exist_ok=True)
            if not description_file.exists():
                with open(description_file, "w") as f:
                    print(description, file=f)
            pbar.update()
            
        
if __name__ == "__main__":
    save_state_descriptions()


