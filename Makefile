.PHONY: run test-pubsub lint playground generate-traces grade

run:
	uv run python -m app.server

test-pubsub:
	curl -X POST -H "Content-Type: application/json" \
		-d '{"subscription": "projects/my-project/subscriptions/infosys-alert-sub", "message": {"data": "eyJ0aWNrZXIiOiAiSU5GWSIsICJldmVudCI6ICJkcm9wXzExIiwgImRldGFpbHMiOiAiSW5mb3N5cyBkcm9wcyAxMSUgYXQgMiBQTSJ9"}}' \
		http://localhost:8080/pubsub

lint:
	uvx ruff check app/ tests/

playground:
	uv run agents-cli playground

generate-traces:
	uv run python tests/eval/generate_traces.py

grade:
	uv run agents-cli eval grade --traces artifacts/traces/generated_traces.json --config tests/eval/eval_config.yaml
