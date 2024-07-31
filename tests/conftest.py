"""Configuration for pytest."""
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    if call.excinfo is not None:
        exc_type, exc_value, _ = call.excinfo
        if '502' in str(exc_value) or 'Read timed out' in str(exc_value):
            call.excinfo = None
            call._report_outcome = 'rerun'
            call.rerun = True


def pytest_addoption(parser):
    parser.addoption('--reruns', action='store', default=5, type=int, help='number of times to rerun failed tests')


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    max_reruns = item.config.getoption('--reruns')
    reruns = getattr(item, 'rerun', 0)
    _ = yield
    while reruns < max_reruns:
        if item._report_outcome == 'rerun':
            reruns += 1
            setattr(item, 'rerun', reruns)
            _ = yield
        else:
            break
