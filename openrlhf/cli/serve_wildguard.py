import argparse
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from vllm import LLM, SamplingParams
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import asyncio

from openrlhf.utils.logging_utils import init_logger
from wildguard.wildguard.utils import (
    WILDGUARD_INPUT_FORMAT,
    SafetyClassifierOutput,
    PromptHarmfulness,
    ResponseHarmfulness,
    ResponseRefusal,
)

logger = init_logger(__name__)


@ray.remote
class WildGuardWorker:
    def __init__(self, args, worker_id):
        self.llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            seed=args.seed + worker_id,
            max_num_seqs=args.max_num_seqs,
            dtype="bfloat16" if args.bf16 else "float32",
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        self.batch_size = args.batch_size

    def parse_model_generation_to_output(self, output: str) -> SafetyClassifierOutput:
        """Parse the model's output into structured safety classifications."""
        import re
        m = re.match(".*:(.*)\n.*:(.*)\n.*:(.*)", output)
        if m is None:
            return SafetyClassifierOutput(is_parsing_error=True)
        
        harmful_request, response_refusal, harmful_response = m.groups()

        is_parse_error = False
        is_user_request_harmful = None
        is_assistant_response_refusal = None
        is_assistant_response_harmful = None

        if harmful_request.strip().lower() == "yes":
            is_user_request_harmful = PromptHarmfulness.HARMFUL
        elif harmful_request.strip().lower() == "no":
            is_user_request_harmful = PromptHarmfulness.UNHARMFUL
        else:
            logger.info(f"harmful_request: {harmful_request}")
            is_parse_error = True

        if response_refusal.strip().lower() == "yes":
            is_assistant_response_refusal = ResponseRefusal.REFUSAL
        elif response_refusal.strip().lower() == "no":
            is_assistant_response_refusal = ResponseRefusal.COMPLIANCE
        elif response_refusal.strip().lower() == "n/a":
            is_assistant_response_refusal = None
        else:
            logger.info(f"response_refusal: {response_refusal}")
            is_parse_error = True

        if harmful_response.strip().lower() == "yes":
            is_assistant_response_harmful = ResponseHarmfulness.HARMFUL
        elif harmful_response.strip().lower() == "no":
            is_assistant_response_harmful = ResponseHarmfulness.UNHARMFUL
        elif harmful_response.strip().lower() == "n/a":
            is_assistant_response_harmful = None
        else:
            logger.info(f"harmful_response: {harmful_response}")
            is_parse_error = True

        return SafetyClassifierOutput(
            prompt_harmfulness=is_user_request_harmful,
            response_harmfulness=is_assistant_response_harmful,
            response_refusal=is_assistant_response_refusal,
            is_parsing_error=is_parse_error,
        )

    def classify(self, queries: list[dict[str, str]], sampling_params=None) -> list[SafetyClassifierOutput]:
        """Classify the safety aspects of queries and responses."""
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=128,
            )

        formatted_prompts = []
        game_indices = []  # Track the original indices
        for idx, item in enumerate(queries):
            if "response" not in item:
                item["response"] = ""
            formatted_prompt = WILDGUARD_INPUT_FORMAT.format(
                prompt=item["prompt"], 
                response=item["response"]
            )
            formatted_prompts.append(formatted_prompt)
            if "game_idx" in item:
                game_indices.append(item["game_idx"])  # Use the provided game_idx
            else:
                raise ValueError("game_idx is required")

        if self.batch_size is not None:
            all_outputs = []
            batch_indices = []  # Track indices for each batch
            for i in range(0, len(formatted_prompts), self.batch_size):
                batch_prompts = formatted_prompts[i:i + self.batch_size]
                batch_indices.extend(game_indices[i:i + self.batch_size])
                outputs = self.llm.generate(batch_prompts, sampling_params)
                all_outputs.extend(outputs)
        else:
            batch_indices = game_indices
            all_outputs = self.llm.generate(formatted_prompts, sampling_params)

        # Verify alignment
        assert len(all_outputs) == len(batch_indices), f"Output count mismatch: expected {len(batch_indices)}, got {len(all_outputs)}"

        # Create results with original indices for ordering
        results = {}
        MAX_RETRIES = 0  # Limit retries to prevent infinite loops
        
        for game_idx, output in zip(batch_indices, all_outputs):
            retries = 0
            # vLLM returns a list of RequestOutput objects, each with a list of generations
            safety_output = self.parse_model_generation_to_output(output.outputs[0].text)
            
            while safety_output.is_parsing_error and retries < MAX_RETRIES:
                logger.warning(f"Parsing error encountered for game_idx {game_idx}, attempt {retries + 1}/{MAX_RETRIES}")
                new_output = self.llm.generate([formatted_prompts[game_idx]], sampling_params)[-1]
                safety_output = self.parse_model_generation_to_output(new_output.outputs[0].text)
                retries += 1
            
            if safety_output.is_parsing_error:
                logger.error(f"Failed to parse output after {MAX_RETRIES} attempts for game_idx {game_idx}")
                # You might want to set default values or handle this case specifically
                
            results[game_idx] = safety_output
        
        assert len(results) == len(queries), f"Result count mismatch: expected {len(queries)}, got {len(results)}"

        return results


class WildGuardServerProxy:
    def __init__(self, args):
        ray.init(ignore_reinit_error=True)
        
        # Validate tensor parallel configuration
        if args.tensor_parallel_size > args.num_gpus:
            raise ValueError(f"Tensor parallel size ({args.tensor_parallel_size}) cannot be larger than number of GPUs ({args.num_gpus})")
        
        if args.num_gpus % args.tensor_parallel_size != 0:
            raise ValueError(f"Number of GPUs ({args.num_gpus}) must be divisible by tensor parallel size ({args.tensor_parallel_size})")

        # Calculate number of model replicas
        num_model_replicas = args.num_gpus // args.tensor_parallel_size
        
        # Initialize workers
        self.workers = []
        for i in range(num_model_replicas):
            # For tensor parallelism > 1, create placement group
            scheduling_strategy = None
            if args.tensor_parallel_size > 1:
                # Create placement group with individual GPU bundles
                bundles = [{"GPU": 1, "CPU": 1}] * args.tensor_parallel_size
                pg = placement_group(bundles)
                ray.get(pg.ready())
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0
                )

            # When tensor_parallel_size=1, assign 1 GPU directly
            num_gpus = 1 if args.tensor_parallel_size == 1 else 0
            
            worker = WildGuardWorker.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(args, i)
            self.workers.append(worker)
        
        self.worker_queues = {i: asyncio.Queue() for i in range(len(self.workers))}
        self.current_worker = 0
        self.batch_counter = 0  # Add a counter for generating unique batch IDs

    def get_next_worker_id(self):
        """Round-robin worker selection"""
        worker_id = self.current_worker
        self.current_worker = (self.current_worker + 1) % len(self.workers)
        return worker_id

    async def process_worker_queue(self, worker_id, batch_id):
        """Process a specific batch from the worker's queue"""
        worker = self.workers[worker_id]
        while True:
            # Keep checking queue until we find our batch
            current_batch, current_id, sampling_params = await self.worker_queues[worker_id].get()
            try:
                if current_id == batch_id:
                    result = await asyncio.to_thread(ray.get, worker.classify.remote(current_batch, sampling_params))
                    return result
                else:
                    # Put other batches back in queue and continue searching
                    await self.worker_queues[worker_id].put((current_batch, current_id, sampling_params))
            finally:
                self.worker_queues[worker_id].task_done()

    async def classify(self, items: list[dict[str, str]], sampling_params=None) -> list[SafetyClassifierOutput]:
        """Queue batch for processing and wait for results"""
        worker_id = self.get_next_worker_id()
        batch_id = self.batch_counter
        self.batch_counter += 1
        await self.worker_queues[worker_id].put((items, batch_id, sampling_params))
        
        # Process the specific batch and return results
        result = await self.process_worker_queue(worker_id, batch_id)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model Configuration
    parser.add_argument("--model_path", type=str, required=True, help="HF model name or path")
    parser.add_argument("--max_len", type=int, default=8192, help="Maximum length")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, 
                       help="Tensor parallel size (must divide evenly into num_gpus)")
    
    # Server Configuration
    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPU workers to use")
    
    # Performance
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for processing")
    parser.add_argument("--bf16", action="store_true", default=False, help="Use bfloat16")
    parser.add_argument("--max_num_seqs", type=int, default=256, help="Maximum number of sequences")
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    # Initialize server
    wildguard_server = WildGuardServerProxy(args)
    app = FastAPI()
    
    @app.post("/classify")
    async def classify(request: Request):
        data = await request.json()
        queries = data.get("queries", [])
        
        # Get sampling parameters if provided
        sampling_config = data.get("sampling_params", {})
        sampling_params = SamplingParams(
            temperature=sampling_config.get("temperature", 0.0),
            top_p=sampling_config.get("top_p", 1.0),
            max_tokens=sampling_config.get("max_tokens", 128),
        )
        
        logger.info(f"Received {len(queries)} queries")
        outputs = await wildguard_server.classify(queries, sampling_params)

        # Format the response - convert SafetyClassifierOutput objects to dictionaries
        formatted_outputs = []
        for game_idx, output in outputs.items():
            formatted_output = {
                "game_idx": game_idx,
                "prompt_harmfulness": output.prompt_harmfulness.value if output.prompt_harmfulness else None,
                "response_harmfulness": output.response_harmfulness.value if output.response_harmfulness else None,
                "response_refusal": output.response_refusal.value if output.response_refusal else None,
                "is_parsing_error": output.is_parsing_error
            }
            formatted_outputs.append(formatted_output)
        
        response = {"labels": formatted_outputs}
        
        logger.info(f"Generated labels for {len(queries)} queries")
        return JSONResponse(response)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info") 