# Local LLMOps usage monitoring

Resume Factory records model usage locally without storing prompts, evidence text, or draft
content in its usage tables. Application results remain under `.local/runs/`; SQLite and MLflow
files are also excluded from Git.

## Inspect a run

```bash
rf usage list --limit 20
rf usage show RUN_ID --group-by question
rf usage show RUN_ID --group-by team
rf usage show RUN_ID --group-by agent --json
rf usage compare RUN_A RUN_B --group-by model
```

Shared company, job, integration, and aggregate QA calls appear under `shared`; the system does
not invent a per-question allocation for them. Offline executions report zero tokens with
`usage_status=offline`. Missing provider metadata and unknown prices remain visible instead of
being treated as free calls.

## Enable MLflow

```dotenv
RF_ENABLE_MLFLOW=true
RF_MLFLOW_TRACKING_URI=sqlite:///.local/mlflow.db
```

```bash
rf run --input tests/fixtures/sample_application.json --offline --mode balanced
mlflow ui --backend-store-uri sqlite:///.local/mlflow.db --host 127.0.0.1 --port 5000
```

MLflow receives aggregate metrics and question/team/agent/model usage tables. It does not receive
prompts, evidence contents, or cover-letter text.

## Record human preference

```bash
rf feedback RUN_ID --decision revised --rating 4 --final-draft .local/final.json
```

When a canonical final-draft JSON is supplied, only its local path, SHA-256, and edit ratios are
stored. The final text is not copied into the telemetry database.

## Price catalog

`src/resume_factory/model_prices.json` contains versioned planning estimates. They are not provider
billing records. Verify the exact model IDs and rates before online use. An unknown model produces
`cost_complete=false` and `unknown_price_calls`; it never silently appears as a zero-cost model.
