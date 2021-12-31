from typing import Callable

import prometheus_fastapi_instrumentator as pinst
from prometheus_fastapi_instrumentator.metrics import Info
from prometheus_client import Histogram


# A dictionary mapping names to Prometheus FastAPI metrics
METRICS = {
    'request_size': pinst.metrics.request_size,
    'response_size': pinst.metrics.response_size,
    'requests': pinst.metrics.requests,
    'latency': pinst.metrics.latency,
}


def setup_prometheus_instrumentator(metrics=None):
    """
    Setup a Prometheus FastAPI instrumentator.

    :param metrics: A list of metric names.
    :return: An instrumentator with metrics.
    """
    # Setup Prometheus instrumentator
    if metrics is None:
        metrics = {'requests': {}}

    instrumentator = pinst.Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        inprogress_name="fastapi_inprogress",
        inprogress_labels=True,
    )

    # Add metrics to the instrumentator
    for metric in metrics.keys():
        instrumentator.add(
            METRICS[metric](
                should_include_handler=True,
                should_include_method=True,
                should_include_status=True,
                metric_namespace='fastapi',
                metric_subsystem='',
                **metrics[metric]
            )
        )

    return instrumentator


def model_output(
    metric_name: str = "model_output",
    metric_doc: str = "Output value of model",
    metric_namespace: str = "",
    metric_subsystem: str = "",
    buckets=(0, 1, 2), **kwargs
) -> Callable[[Info], None]:
    metric = Histogram(
        metric_name,
        metric_doc,
        buckets=buckets,
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == '/predict' and info.response and hasattr(info.response, "headers"):
            predicted_condition = info.response.headers.get("X-prediction")
            if predicted_condition is not None:
                metric.observe(float(predicted_condition))

    return instrumentation


METRICS['model_output'] = model_output
