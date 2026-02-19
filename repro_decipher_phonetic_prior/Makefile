PYTHON ?= python3

.PHONY: setup data fetch_data smoke train_gothic eval_gothic fig4 table2 table3 table4 report compare_graphs compare_paper run_validation run_validation_all_pairs reproduce_all reproduce_paper make_tables

setup:
	bash scripts/setup.sh

data:
	bash scripts/prepare_datasets.sh

fetch_data:
	bash scripts/fetch_data.sh

smoke:
	bash scripts/smoke_test.sh

train_gothic:
	bash scripts/train_gothic.sh

eval_gothic:
	bash scripts/eval_gothic.sh

fig4:
	bash scripts/fig4.sh
	$(PYTHON) -m repro.report

table2:
	bash scripts/table2.sh

table3:
	bash scripts/table3.sh

table4:
	bash scripts/table4.sh

report:
	$(PYTHON) -m repro.report

compare_graphs:
	bash scripts/compare_graphs.sh

run_validation:
	bash scripts/run_validation_pair.sh

run_validation_all_pairs:
	bash scripts/run_validation_all_pairs.sh

reproduce_all:
	bash scripts/reproduce_all.sh

reproduce_paper:
	bash scripts/reproduce_paper.sh

make_tables:
	$(PYTHON) scripts/make_tables.py --output-root outputs

compare_paper:
	$(PYTHON) scripts/make_comparative_graphs.py --output-root outputs
