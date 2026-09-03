# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Running tests:**
```bash
just test           # parallel run, excludes "fixme" marked tests (default)
just test-all       # all tests including slow ones
just test-changes   # only tests affected by recent code changes (pytest-testmon)
just test-failures  # rerun last failures only
pytest tests/test_file.py::test_function_name   # single test
pytest -m "cvxpy"   # by marker (gurobipy, pyomo, slow, scenario, stochastic, etc.)
pytest -n0          # serial (no parallelism, useful for debugging)
```

**Lint:**
```bash
just lint       # pre-commit on staged files
just lint-all   # pre-commit on all files
```

**Docs:**
```bash
just docs-fast   # serve locally with fast rebuild
just docs-rtd    # serve with ReadTheDocs config
```

**Dependency management:** `uv` (not pip). Add deps with `uv add`, sync with `uv sync`.

## Pytest markers

Defined in `pyproject.toml`. Key ones: `slow`, `fixme` (always skipped), `githubskip` (skipped in CI), `gurobipy`, `cvxpy`, `pyomo`, `scipy`, `json`, `polars`, `scenario`, `stochastic`, `robust`, `scalarization`, `nautilus`, `nimbus`, `ea`, `simulator_support`, `api`.

## Architecture

### Expression system (MathJSON)

All mathematical expressions are stored internally as **MathJSON** — a nested list/dict format, e.g. `["Add", "x_1", ["Multiply", 2, "x_2"]]`. Two input paths:

1. **Infix strings** (`"x_1 + 2*x_2"`) → converted by `InfixExpressionParser` (`problem/infix_parser.py`) → MathJSON
2. **MathJSON directly** as Python lists

`MathParser` in `problem/json_parser.py` converts MathJSON to solver-specific representations: polars expressions, Pyomo expressions, SymPy expressions, gurobipy linear expressions, or CVXPY expressions. This is the central hub enabling multi-backend support.

Pydantic field validator `parse_infix_to_func` on `func` fields (in `schema.py`) auto-converts infix strings to MathJSON at model construction time. Passing a **bare variable symbol string** (e.g. `func="x_1"`) will also be parsed — wrap it in a list if you need a literal MathJSON reference (e.g. `func=["Multiply", 1, "x_1"]`) to avoid backend-specific issues.

Tensor operations in MathJSON: `Extract` (include indices), `Exclude` (exclude indices), `Sum` (reduce), `At` (single index), `Tuple` (index range spec). Indices are 1-based (positive) or end-relative (negative).

### Problem definition (`problem/schema.py`)

`Problem` is a frozen Pydantic model. Key fields: `variables`, `constants`, `objectives`, `constraints`, `extra_funcs`, `scalarization_funcs`. Functions in these elements are MathJSON lists. Scalar and tensor variants exist for variables/constants (`Variable`/`TensorVariable`, `Constant`/`TensorConstant`).

Evaluators (`PolarsEvaluator`, `PyomoEvaluator`, `GurobipyEvaluator`, `CVXPYEvaluator`, `SympyEvaluator`) in `problem/*_evaluator.py` take a `Problem` and materialise it for a specific backend. Solvers in `tools/` wrap evaluators and return `SolverResults`.

### Solver selection

Use `guess_best_solver(problem)` from `desdeo.tools` to get a compatible solver class. `find_compatible_solvers(problem)` returns all options. Solver compatibility depends on `is_linear`, `is_convex`, `is_twice_differentiable` flags on problem elements.

For non-differentiable problems (e.g. NIMBUS/STOM nondiff scalarizations), use `NevergradGenericSolver` with `NGOpt` optimizer and budget ≥ 500, or `ScipyDeSolver`. `CMA` works well for small-dimensional smooth problems but fails on constrained nondiff problems.

### Scenario-based optimization (`problem/scenario.py`, `tools/scenarios.py`)

`ScenarioModel` holds a `base_problem` and **pools** of replaceable elements (constants, variables, objectives, constraints, etc.). Each `Scenario` maps element symbols to pool indices. `scenario_tree` (dict) defines the temporal/branching structure; `anticipation_stop` enforces non-anticipativity.

`build_combined_scenario_problem(scenario_model)` expands all scenarios into one large `Problem` with per-leaf copies of each element (symbols prefixed with leaf name). Returns `(combined_problem, symbol_maps)` where `symbol_maps` maps original symbol → leaf → combined symbol.

Aggregation functions (`add_expected_value`, `add_worst_case_robust`, `add_conditional_value_at_risk`, `add_weighted_scenarios`) take `(scenario_model, symbols, combined=..., symbol_maps=...)`. Pass pre-built `combined`+`symbol_maps` when applying multiple aggregations to the same problem.

`leaf_scenarios` on `ScenarioModel` returns a `MappingProxyType` (read-only). `with_base_problem(problem, validate=False)` returns a new model with updated base; pass `validate=True` to re-run Pydantic validators (safe when only adding elements, not removing variables).

### Scalarization (`tools/scalarization.py`, `tools/group_scalarization.py`)

`add_asf_diff` / `add_asf_nondiff`: Achievement Scalarizing Function (differentiable/nondiff variants).
`add_asf_partial_diff`: ASF using only a subset of objectives (reference point dict keys determine which objectives are used).
`add_epsilon_constraints`: Epsilon-constraint method.
All return `(new_problem, scalarization_symbol)`.

### Web API (`desdeo/api/`)

FastAPI application with SQLModel/SQLite (dev) or PostgreSQL (prod). DB models mirror the Pydantic problem schema via `from_pydantic()` helper in `api/models/problem.py` — auto-generates SQLModel classes from Pydantic models, handling union types and JSON columns. `ScenarioModelDB` is hand-written because `ScenarioModel`'s structure (FK references, relationship tables, polymorphic pools) can't be auto-generated.

Problem-specific side data that doesn't belong on the core (frozen, `extra="forbid"`) `Problem` schema — geographic data, generated descriptions, solver preference, anything a specific feature needs attached to a problem — goes through `ProblemMetaDataDB`'s polymorphic side-table pattern in `api/models/problem.py`: one table per metadata kind (e.g. `SiteSelectionMetaData`, `ForestProblemMetaData`), each with a `metadata_type` discriminator and an FK to `ProblemMetaDataDB`, aggregated via its `all_metadata` property and exposed through `ProblemMetaDataPublic`. This doubles as the signal the frontend uses to decide whether to render a specialised view for a problem (e.g. a map) instead of the generic one.

### Docs

Built with MkDocs + mkdocs-jupyter. Local config: `mkdocs.yml` (`execute: false`). RTD config: `mkdocs.rtd.yml` (`execute: true`, inherits from `mkdocs.yml`). Heavy notebooks that can't run on RTD should live in `notebooks/` (outside `docs/`) and be linked via GitHub raw URL from the `.md` page.

## WebUI (`webui/`)

SvelteKit 2 frontend (Svelte 5 runes mode) with TypeScript and Tailwind CSS 4.

### Commands

```bash
npm install                  # install dependencies
npm run dev                  # dev server at localhost:5173
npm run build                # production build
npm run preview              # preview production bundle
npm run generate:client      # regenerate OpenAPI client (requires API running at localhost:8000)
npm run lint                 # Prettier + ESLint
npm run format               # auto-format
npm run check                # svelte-check + type checking
npm run test:unit            # Vitest unit tests
npm run test:e2e             # Playwright e2e tests
npm run storybook            # component dev server at port 6006
```

**Environment variables:** `API_BASE_URL` (server-side, runtime — backend URL), `VITE_API_URL` (build-time, defaults to `/api`).

### Structure

```
src/
  routes/          # file-based routing; +page.server.ts for server logic, +page.svelte for UI
    /api/[...path] # catch-all proxy: forwards browser requests to backend with auth
  lib/
    gen/           # auto-generated OpenAPI client (Orval) — do not edit manually
    api/           # new-client.ts: custom fetch wrapper (dual-URL: server→direct, browser→/api)
    components/
      ui/          # shadcn-svelte headless components
      custom/      # domain-specific components
      visualizations/  # D3 / ECharts charts
    stores/        # Svelte stores (auth, methodSelection, uiState) with localStorage persistence
    helpers/       # utility functions
  hooks.server.ts  # handleFetch: automatic 401 → token refresh → retry
```

### Key patterns

- **API proxy**: all browser API calls go through `/api/[...path]` → SvelteKit forwards to backend. Avoids CORS, centralises auth.
- **Token refresh**: `hooks.server.ts` intercepts 401s, refreshes via `/auth/refresh`, retries. Tokens stored as HTTP-only cookies.
- **Generated client**: `orval.config.mjs` generates `src/lib/gen/` from OpenAPI spec. Run `generate:client` after backend changes.
- **Forms**: sveltekit-superforms + Zod for server-validated, type-safe forms.
- **State**: Svelte stores in `src/stores/` (auth) and `src/lib/stores/` (method selection, UI state).
- **Styling**: shadcn-svelte components (`components.json` configures aliases); formatting uses tabs, no semicolons, print width 100.
