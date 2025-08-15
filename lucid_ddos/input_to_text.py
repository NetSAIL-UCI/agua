from openai import OpenAI
import openai
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import Tuple
from global_constants import (PROMPT_FILE, DATASET_PATH, INPUT_SAVE_PATH, 
                        TEST_INPUT_SAVE_PATH, NUM_TEST_INPUTS,
                       INPUT_DESCRIPTION_SAVE_PATH, MAX_NUM_INPUTS, LLM_MODEL,
                       CONCEPTS_FILE, N_QUERY_TOGETHER, NUM_TEST_INPUTS, N_ACTIONS,
                       TEST_INPUT_SAVE_PATH) 
from multiprocessing.dummy import Pool
from time import sleep
from collections import OrderedDict
from util_functions import load_dataset as load_lucid_dataset
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
PROTOCOLS = ['arp','data','dns','ftp','http','icmp','ip','ssdp','ssl','telnet','tcp','udp']
HISTORY_FEATURES = ["Timestamp, with a max of 10", 
                    "Packet Length (kB), with a max of 2^16",
                    "Higher Layer of Network, with a max of 2^32", 
                    "IP Flags of the request, with a max of 2^16",
                    "Which protocol the connection is using",
                    "Length of TCP Connection, with a max of 2^16",
                    "Bytes Acknowledged by TCP, with a max of 2^32",
                    "TCP Flags used by the connection",
                    "TCP Window size (packets), with a max of 2^16",
                    "UDP connection length, with a max of 2^16", 
                    "Type of ICMP"]
STATE_DIM = (HISTORY_LEN, len(HISTORY_FEATURES) + N_ACTIONS + N_ACTIONS)

FEATURE_RANGES = OrderedDict([
    ('timestamp', [0,10]),
    ('packet_length',[0,1<<16]),
    ('highest_layer',[0,1<<32]),
    ('IP_flags',[0,1<<16]),
    ('protocols',[0,1<<len(PROTOCOLS)]),
    ('TCP_length',[0,1<<16]),
    ('TCP_ack',[0,1<<32]),
    ('TCP_flags',[0,1<<16]),
    ('TCP_window_size',[0,1<<16]),
    ('UDP_length',[0,1<<16]),
    ('ICMP_type',[0,1<<8])]
)


#%%
def unnormalize(input_sample: np.ndarray) -> np.ndarray:
    """Scale normalized feature slice back to original approximate ranges.

    Parameters
    ----------
    input_sample : np.ndarray
        Array shaped (HISTORY_LEN, N_FEATURES) containing normalized values in
        [0, 1] for each feature.

    Returns
    -------
    np.ndarray
        Array of same shape with each column rescaled by its recorded max.
    """
    unnormalized_sample = np.zeros_like(input_sample)
    for feature_idx, (feature_name, (min_val, max_val)) in enumerate(FEATURE_RANGES.items()):
        unnormalized_sample[:, feature_idx] = input_sample[:, feature_idx] * max_val
    return unnormalized_sample


def input_to_str(input_sample: np.ndarray) -> str:
    """Format a single normalized input tensor as a multi-line string.

    Parameters
    ----------
    input_sample : np.ndarray
        Normalized input with shape (HISTORY_LEN, N_FEATURES[, 1]). Only the
        first two dims are used; an optional final singleton channel is removed.

    Returns
    -------
    str
        Human-readable feature-wise representation for prompting an LLM.
    """
    input_description = []
    raw_sample = unnormalize(input_sample)
    raw_sample = raw_sample.reshape(10, 11)
    for feature_idx, feature_name in enumerate(HISTORY_FEATURES):
        feature_description = f"{feature_name}: ["
        for val in raw_sample.T[feature_idx]:
            feature_description += f"{val:.3f}, "
        feature_description += "]"
        input_description.append(feature_description)
    input_description = "\n\n\t\t\t".join(input_description)
    return input_description

def get_llm_description(state: np.ndarray, client: OpenAI) -> str:
    """Query the LLM for a concise description of ``state``.

    Parameters
    ----------
    state : np.ndarray
        Normalized input sample.
    client : OpenAI
        Instantiated OpenAI client used to submit the chat completion request.

    Returns
    -------
    str
        LLM-produced description text.
    """
    prompt = PROMPT.format(concepts = CONCEPTS, state_data = input_to_str(input_sample=state))
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
    """Worker helper to fetch or reuse a description for one input.

    Parameters
    ----------
    data : Tuple[int, np.ndarray, OpenAI]
        (index, normalized_state, openai_client) triple.

    Returns
    -------
    Tuple[int, str]
        (index, description_text) pair.
    """
    input_idx, state, client = data
    msg = None
    description_file = INPUT_DESCRIPTION_SAVE_PATH / f"input_{input_idx:07d}.txt"
    if description_file.exists():
        msg = description_file.read_text().rstrip() 
    while msg is None:
        try:
            msg = get_llm_description(state=state, client=client)
            sleep(0.5)
        except (openai.RateLimitError, openai.APITimeoutError):
            sleep(1.5)
    return (input_idx, msg)



def load_dataset(n_samples: int = MAX_NUM_INPUTS, 
                 n_test_samples: int = NUM_TEST_INPUTS) -> np.ndarray:
    """Load train/test splits from on-disk Lucid dataset files.

    Parameters
    ----------
    n_samples : int, default=MAX_NUM_INPUTS
        Maximum number of training inputs to retain.
    n_test_samples : int, default=NUM_TEST_INPUTS
        Maximum number of test inputs to retain.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (train_inputs, train_outputs, test_inputs, test_outputs)
    """
    train_inputs, train_outputs = load_lucid_dataset(str(DATASET_PATH) + "/*" + '-train.hdf5')
    test_inputs, test_outputs = load_lucid_dataset(str(DATASET_PATH) + "/*" + '-test.hdf5')
    
    train_inputs = train_inputs[:n_samples]
    train_outputs = train_outputs[:n_samples]
    
    test_inputs = test_inputs[:n_test_samples]
    test_outputs = test_outputs[:n_test_samples]
    
    if train_inputs.shape[0] < n_samples:
        warnings.warn(f"Only found {train_inputs.shape[0]} train samples")
    if test_outputs.shape[0] < n_test_samples:
        warnings.warn(f"Only found {test_outputs.shape[0]} test samples")
    
    return train_inputs, train_outputs, test_inputs, test_outputs


def save_input_descriptions() -> None:
    """Generate and persist descriptions for all loaded training inputs.

    Skips any description file that already exists on disk to avoid
    unnecessary re-queries.
    """
    client = OpenAI()
    input_samples, outputs, test_input_samples, test_outputs = load_dataset(n_samples= MAX_NUM_INPUTS)
    pool = Pool(processes= N_QUERY_TOGETHER)
    inputs = [[input_idx, input_sample, client] for input_idx, input_sample in enumerate(input_samples)]
    with tqdm(total=len(input_samples), leave=False, desc="Querying LLM") as pbar:
        for input_idx, description in pool.imap_unordered(llm_wrapper, inputs):
            description_file = INPUT_DESCRIPTION_SAVE_PATH / f"input_{input_idx:07d}.txt"
            description_file.parent.mkdir(parents=True, exist_ok=True)
            if not description_file.exists():
                with open(description_file, "w") as f:
                    print(description, file=f)
            pbar.update()
            

if __name__ == "__main__":
    save_input_descriptions()


