.PHONY: setup lint typecheck mock-api run-exp1 run-exp2 eval check-secrets clean

setup:
	pip install -e .
	cp -n .env.example .env 2>/dev/null || true
	@echo "Setup complete. Configure API keys only if you intend to run live models."

lint:
	ruff check .

typecheck:
	mypy agents/ mock_api/ eval_pipeline/ --ignore-missing-imports

mock-api:
	uvicorn mock_api.server:app --host 0.0.0.0 --port 8001 --reload

run-exp1:
	@echo "Full Exp1 replay requires controlled attack documents and raw cached search results, which are not in the public package."
	python scripts/run_exp1_v2.py --help

run-exp2:
	@echo "Full Exp2 replay requires controlled attack documents and raw cached search results, which are not in the public package."
	python scripts/run_exp2_v2.py --help


eval:
	@echo "Public results are already evaluated under results/aggregate_metrics and results/judge_labels."
	@echo "Use run_eval.py for newly generated raw experiment outputs, not sanitized JSONL files."
	python scripts/run_eval.py --help

check-secrets:
	@echo "Checking for leaked secrets..."
	@! grep -rn "sk-[a-zA-Z0-9]\{20,\}" --include="*.py" --include="*.yaml" --include="*.json" . \
		--exclude-dir=.git --exclude-dir=node_modules --exclude=.env --exclude=.env.example || \
		(echo "Potential OpenAI-style API keys found!" && exit 1)
	@! grep -rn "sk-ant-[a-zA-Z0-9]\{20,\}" --include="*.py" --include="*.yaml" --include="*.json" . \
		--exclude-dir=.git --exclude-dir=node_modules --exclude=.env --exclude=.env.example || \
		(echo "Potential Anthropic API keys found!" && exit 1)
	@echo "No obvious secrets found."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache
