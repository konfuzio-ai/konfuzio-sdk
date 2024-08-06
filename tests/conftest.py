import logging

import pytest

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == 'call' and report.failed and call.excinfo is not None:
        exc_value = call.excinfo.value
        error_message = str(exc_value)

        logger.debug(f'Error message: {error_message}')

        if '502' in error_message or 'Read timed out' in error_message:
            setattr(report, 'wasxfail', False)
            setattr(report, 'rerun', True)
            report.outcome = 'failed'
        else:
            setattr(report, 'rerun', False)
