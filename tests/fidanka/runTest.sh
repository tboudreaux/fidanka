#!/bin/bash
#
export FIDANKA_TEST_ROOT_DIR=$(pwd)
coverage erase
coverage run -m pytest
coverage report
coverage html
