# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from omegaconf import DictConfig

import torch

from nemo_aligner.experimental.grpo.utils import parallel_state
from nemo_aligner.experimental.grpo.experience.interfaces import EnvironmentInterface
from nemo_aligner.experimental.grpo.experience.environments.metrics import calculate_pass_rate_per_prompt
from nemo_aligner.utils.distributed import broadcast_2d_tensor_within_mp

from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.inference.server.model import get_model
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import prefill_judgement


class MathEnvironment(EnvironmentInterface):
    def __init__(self, cfg: DictConfig):
        print(f"Started MathEnvironment client with config {cfg}")
        self.cfg = cfg

        host = os.getenv("SLURM_MASTER_NODE_HET_GROUP_0", "localhost")
        self.llm = get_model(host=host, **self.cfg.server)
        self.prompt = get_prompt(**self.cfg.prompt)

    def start_step(self, interactions, metadata):
        """
        metadata: List[Dict]. Needs to contain a "ground_truth" key, which is what
                              the grader will use to evaluate correctness.
        """
        if parallel_state.is_model_parallel_src_rank():
            # fold all interactions after the prompt together
            responses = [''.join(interaction[1:]) for interaction in interactions]

            data_points = []
            prefilled_judgements = []
            prefilled_indices = set()
            for idx, (response_metadata, response) in enumerate(zip(metadata, responses)):
                dp = {
                    "problem": response_metadata["problem"],
                    "expected_answer": response_metadata["expected_answer"],
                    "predicted_answer": extract_answer(response),
                }
                judgement = prefill_judgement(dp)
                if judgement is not None:
                    prefilled_judgements.append(judgement)
                    prefilled_indices.add(idx)
                else:  # cannot prefill, will send to an LLM
                    data_points.append(dp)

            judge_prompts = [self.prompt.fill(dp) for dp in data_points]
            if len(judge_prompts) > 0:
                generation_ids = self.llm.generate_async(prompts=judge_prompts, stop_phrases=self.prompt.stop_phrases)
            else:
                generation_ids = []

            return prefilled_judgements, prefilled_indices, generation_ids

        return None

    def finish_step(self, future):
        # gets the future result and also broadcasts within the current MP group
        results = None
        if future is not None:
            prefilled_judgements, prefilled_indices, generation_ids = future
            if generation_ids:
                outputs = self.llm.get_generations(generation_ids)
            else:
                outputs = []

            judgements = []
            prefilled_idx = 0
            generation_idx = 0
            # looping over all and selecting either prefilled or generated judgements
            for idx in range(len(outputs) + len(prefilled_judgements)):
                if idx in prefilled_indices:
                    judgements.append(prefilled_judgements[prefilled_idx])
                    prefilled_idx += 1
                else:
                    judgements.append(outputs[generation_idx]["generation"])
                    generation_idx += 1
            results = [is_correct_judgement(judgement) for judgement in judgements]

            # sharing across MP group
            results = torch.tensor(results, device=torch.cuda.current_device()).unsqueeze(1)
        results = broadcast_2d_tensor_within_mp(results)
        th_rewards = torch.tensor(results).squeeze(1)

        print('th rewards shape', th_rewards.shape)
        return None, None, th_rewards, torch.ones(th_rewards.shape[0],)

    def global_post_process_and_metrics(self, batch):
        """
        Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        """
        table = {
            "reward": batch["rewards"][0].item(),
            "prompt_sentence": batch["prompt_sentences"][0],
            "response_sentence": batch["response_sentences"][0],
            "expected_answer": batch["extra_verifier_info"][0]["expected_answer"],
        }
        batch["rewards"] = batch["rewards"] * batch["is_end"] # set a reward of 0 for any incorrectly ended sequences
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["response_lengths"] - batch["prompt_lengths"])[batch["rewards"] == 1].float().mean().item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics = {
            #"table": table, TODO @sahilj WIP
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(batch["text"], batch["rewards"]),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "response_lengths": batch["response_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "generation_lengths": (batch["response_lengths"] - batch["prompt_lengths"]).float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
