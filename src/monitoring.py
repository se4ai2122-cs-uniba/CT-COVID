import prometheus_fastapi_instrumentator as pinst


# A dictionary mapping names to Prometheus FastAPI metrics
METRICS = {
    'request_size': pinst.metrics.request_size,
    'response_size': pinst.metrics.response_size,
    'requests': pinst.metrics.requests,
    'latency': pinst.metrics.latency
}


def setup_prometheus_instrumentator(metrics=['requests']):
    """
    Setup a Prometheus FastAPI instrumentator.

    :param metrics: A list of metric names.
    :return: An instrumentator with metrics.
    """
    # Setup Prometheus instrumentator
    instrumentator = pinst.Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        inprogress_name="fastapi_inprogress",
        inprogress_labels=True,
    )

    # Add metrics to the instrumentator
    instrumentator_metrics = [METRICS[name] for name in metrics]
    for metric_fn in instrumentator_metrics:
        instrumentator.add(
            metric_fn(
                should_include_handler=True,
                should_include_method=True,
                should_include_status=True,
                metric_namespace='fastapi',
                metric_subsystem='model'
            )
        )

    return instrumentator
