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
    STATE_DESCRIPTION_SAVE_PATH,
    MAX_NUM_STATES,
    LLM_MODEL,
    CONCEPTS_FILE,
    N_QUERY_TOGETHER,
    NUM_TEST_STATES,
)
from multiprocessing.dummy import Pool
from time import sleep


with open(PROMPT_FILE, "r") as f:
    PROMPT = "".join(f.readlines())

with open(CONCEPTS_FILE, "r") as f:
    CONCEPTS = "".join(f.readlines())

CUSTOM_INSTRUCTIONS = ("You are a computer scientist trying to gather "
                       "key information to use in an embedding model"
                       "to identify patterns."
                       "Be straight to the point and avoid unnecessary words.")

HISTORY_LEN = 10
FEATURES = ["Loss ratio (bytes lost / bytes sent; where 0 indicates no packet loss while 1 indicates all packets being dropped)", 
            "Latency ratio (average latency / minimum latency; where 1 indicates no increase in latency while a high number shows self-inflicting latency)",
            "Send ratio (sending rate / receiving rate; where 1 indicates all packets being received while a high number indicates a large number of sent data being lost in the network)",
            "Sent Latency Increase ((latecy of later half - latency of first half) / sending duration; where 0 indicates no changes in latency to client while a high number indicates increasing latency)"]
STATE_DIM = (HISTORY_LEN, len(FEATURES))

MIN_STATE = np.array([0., 1., 0., -1.])
MAX_STATE = np.array([1., 10000., 1000., 10.])
TEST_FRACTION = 0.3

def unnormalize(obs: np.ndarray) -> np.ndarray:
    """Reverse min-max style normalization on state tensor.

    Parameters
    ----------
    obs : np.ndarray
        Normalized state array with shape ``(T, F)`` consistent with
        ``STATE_DIM``.

    Returns
    -------
    np.ndarray
        Denormalized raw observation values in original scale.
    """
    raw_obs = (obs * (MAX_STATE - MIN_STATE)) + MIN_STATE
    return raw_obs



def state_to_str(state: np.ndarray) -> str:
    """Serialize a normalized congestion control state to text.

    Parameters
    ----------
    state : np.ndarray
        Normalized state array of shape ``STATE_DIM``.

    Returns
    -------
    str
        Multi-line description enumerating feature histories.
    """
    state_description = []
    raw_state = unnormalize(state)
    for feature_idx, feature_name in enumerate(FEATURES):
        feature_description = f"{feature_name}: ["
        for val in raw_state.T[feature_idx]:
            feature_description += f"{val:.2e}, "
        feature_description += "]"
        state_description.append(feature_description)
    state_description = "\n\n\t\t\t".join(state_description)
    return state_description

def get_llm_description(state: np.ndarray, client: OpenAI) -> str:
    """Query LLM for a natural language description of the state.

    Parameters
    ----------
    state : np.ndarray
        Normalized state array.
    client : OpenAI
        OpenAI client for chat completion API.

    Returns
    -------
    str
        Generated description referencing provided concepts and data.
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
    """Parallel-friendly wrapper returning (index, description).

    Parameters
    ----------
    data : tuple
        ``(state_idx, state, client)`` triplet.

    Returns
    -------
    tuple
        ``(state_idx, description)``; description loaded or generated.
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
    seed: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load stored states and actions for training / evaluation.

    Parameters
    ----------
    n_samples : int, optional
        Max number of training states to load.
    n_test_samples : int, optional
        Max number of test states to load.
    disable_progress : bool, optional
        Unused; retained for interface compatibility.
    seed : int, optional
        Unused; retained for backwards compatibility.

    Returns
    -------
    tuple
        ``(train_states, train_actions, test_states, test_actions)``.
    """

    train_files = sorted(STATE_SAVE_PATH.glob("*.npz"))[:n_samples]
    states = np.zeros(((len(train_files),) + STATE_DIM), dtype=np.float32)
    actions = np.zeros((len(train_files), ), dtype=np.int64)
    for idx, file in enumerate(train_files):
        data = np.load(file)
        states[idx] = data["state"]
        actions[idx] = data["action"]

    test_files = sorted(TEST_STATE_SAVE_PATH.glob("*.npz"))[:n_test_samples]
    test_states = np.zeros(((len(test_files),) + STATE_DIM), dtype=np.float32)
    test_actions = np.zeros((len(test_files), ), dtype=np.int64)
    for idx, file in enumerate(test_files):
        data = np.load(file)
        test_states[idx] = data["state"]
        test_actions[idx] = data["action"]

    if len(train_files) < n_samples:
        raise ValueError(
            (
                "Couldn't find sufficient train states with the given config. "
                f"Only found {len(train_files)}"
            )
        )
    if len(test_files) < n_test_samples:
        raise ValueError(
            (
                "Couldn't find sufficient test states with the given config. "
                f"Only found {len(test_files)}"
            )
        )

    return states, actions, test_states, test_actions


def save_state_descriptions() -> None:
    """Generate and persist LLM descriptions for each stored state."""
    client = OpenAI()
    states, actions, test_states, test_actions = load_dataset()
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


