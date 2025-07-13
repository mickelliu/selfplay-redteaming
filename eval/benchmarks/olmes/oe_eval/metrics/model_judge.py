import logging
from itertools import groupby
from operator import itemgetter
from typing import List, Optional

from oe_eval.metrics.metric import Metric
from oe_eval.utilities.model_utils import load_judge_model

logger = logging.getLogger()


class ModelJudgeMetric(Metric):
    def __init__(
        self,
        model_config: dict,
        judge_config: dict,
        metric_names: List[str],
        scores_for_docs_dict: Optional[dict] = None,  # Used to pass along actual function
        deferred_metric: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metric_names = metric_names
        self.model = None
        if deferred_metric is None:
            # If non-API judge model is used, we'll defer metric calculation
            deferred_metric = (
                "model_type" in model_config and model_config["model_type"] != "litellm"
            )
        self.deferred_metric = deferred_metric

        if not self.deferred_metric:
            self.model = load_judge_model(model_config)
        self.model_config = model_config
        if scores_for_docs_dict is not None and isinstance(
            judge_config.get("scores_for_docs_fn"), str
        ):
            judge_config = judge_config.copy()
            judge_config["scores_for_docs_fn"] = scores_for_docs_dict[
                judge_config["scores_for_docs_fn"]
            ]
        self.judge_config = judge_config

    def process_one_doc(self, group_lst) -> dict:
        # Not used
        return {}

    def compute_for_docs(self, results_for_requests) -> None:
        if self.model is None:
            self.model = load_judge_model(self.model_config)

        self._scores_for_requests = results_for_requests
        self._scores_for_docs = []
        doc_lookup = {}
        for group_key_vals, group in groupby(
            sorted(self._scores_for_requests, key=itemgetter(*self.doc_groupby_keys)),
            itemgetter(*self.doc_groupby_keys),
        ):
            doc_id, native_id = group_key_vals
            group_lst = list(group)
            model_output = [res["model_resps"] for res in group_lst]
            doc_lookup[doc_id] = group_lst[0]["doc"]

            metrics_value: dict = {}

            # create doc/instance level output file data
            keys = ("doc_id", "native_id", "metrics", "model_output", "label")
            values = (doc_id, native_id, metrics_value, model_output, group_lst[0].get("label"))
            self._scores_for_docs.append(dict(zip(keys, values)))

        if "scores_for_docs_fn" in self.judge_config:
            scores_for_docs_fn = self.judge_config["scores_for_docs_fn"]
            logger.info(f"Running judge model on {len(self._scores_for_docs)} instances...")
            scores_for_docs_fn(
                model=self.model,
                scores_for_docs=self._scores_for_docs,
                doc_lookup=doc_lookup,
                generation_kwargs=self.judge_config.get("generation_kwargs", {}),
            )
            return

        # TODO: Support prompt fn etc more generally

        raise NotImplementedError
