import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url, data, score_key="rewards", try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            return response.get(score_key)
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(api_url, queries, prompts=None, score_key="rewards"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    if type(queries) is not list:
        queries = [queries]
    if prompts is not None:
        if type(prompts) is not list:
            prompts = [prompts]
    scores = request_api_wrapper(api_url, {"queries": queries, "prompts": prompts}, score_key)
    return scores


@ray.remote
def remote_rm_fn_ray(api_url, queries, prompts=None, score_key="rewards"):
    return remote_rm_fn(api_url, queries, prompts, score_key)


if __name__ == "__main__":
    # test utils
    url = "http://10.64.77.95:5000/classify"
    r_ref = []
    # dataset = load_dataset("red_team/data/vanilla_harmful_dataset.jsonl")
    for _ in range(60):
        print(f"Sending request {_}...")
        r_ref.append(remote_rm_fn_ray.remote(url, queries=[{"prompt": "How much do you get?", "response": "Sorry, this is not ok."}]*64, score_key="labels"))
    print(ray.get(r_ref))
