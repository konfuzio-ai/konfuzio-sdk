import pytest


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == 'call' and report.failed and call.excinfo is not None:
        exc_value = call.excinfo.value
        error_message = str(exc_value)
        if '502' in error_message or 'Read timed out' in error_message:
            # Modify the report to mark the test for rerun
            setattr(report, 'wasxfail', False)
            setattr(report, 'rerun', True)
            report.outcome = 'failed'


def pytest_addoption(parser):
    parser.addoption('--n_reruns', action='store', default=5, help='Number of times to rerun failed tests')


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    try:
        reruns = int(item.config.getoption('--n_reruns'))
    except TypeError:
        reruns = 0
    for i in range(reruns + 1):
        if i > 0:
            item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
        outcome = yield
        report = outcome.get_result()
        if not getattr(report, 'rerun', False):
            break
        if i == reruns:
            report.outcome = 'failed'
