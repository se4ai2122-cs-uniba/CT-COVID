#!/bin/bash
cd test
for checkpoint in ct-train ct-valid ct-test
do
  great_expectations --v3-api checkpoint run $checkpoint
done
