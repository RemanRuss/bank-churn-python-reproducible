# Unit 1 Project Reflection

## What was the original state of your analysis? What made it difficult to reproduce?

The original analysis was mainly contained in one R Markdown file. Data cleaning, modeling,
evaluation, and visualization were mixed together, with no automated workflow, isolated
environment, or tests.

## What were the biggest challenges in the transformation?

The main challenge was converting the original R workflow into Python while keeping the same
analysis structure. I also needed to separate preprocessing, modeling, evaluation, and reporting
into independent scripts.

## Which improvements had the most impact on reproducibility?

The biggest improvements were the structured project folders, `requirements.txt`, automated
`Makefile`, fixed random seed, and `pytest` tests. These changes made the project much easier to
reproduce on another computer.

## What would you do differently in a future project?

I would organize the project structure and environment at the beginning instead of refactoring
everything at the end. I would also make smaller Git commits and add tests while developing each
component.

## How long did each major component take to implement?

Project structure and environment setup took about 3 minutes, Python refactoring about 1 minute,
testing and automation about 1 minute, and documentation and final checking about 1 minute.

## How did AI tools contribute, and how did you verify their output?

AI tools helped translate the R workflow into Python, organize the project, draft tests, and improve
documentation. I verified the output by running `make setup`, `make test`, and `make reproduce`,
then checking that the tests passed and the final tables, figures, model artifacts, and HTML report
were generated correctly.
