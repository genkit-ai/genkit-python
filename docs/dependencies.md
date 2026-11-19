# Python Dependency Policy

CI installs from `uv.lock` (`uv sync --frozen`). Users don't get the lockfile; they resolve the ranges in `packages/*/pyproject.toml`. This page covers both.

## Upper bounds (caps)

Cap a published dependency at the next major (`<N+1`, or `<1` while 0.x) only if it meets one of two criteria:

- **Integration target:** the package exists to wrap it (`openai` in `genkit-openai`, `flask` in `genkit-flask`, `boto3` in `genkit-amazon-bedrock`).
- **Contract:** users write types or files against it and pass them to Genkit (`pydantic` models, `dotpromptz` `.prompt` files).

Leave everything else floor-only (`starlette`, `anyio`, `structlog`, `uvicorn`, ...). Capping internal deps blocks upstream security fixes and creates version conflicts with the rest of a user's app.

**Enforcement (`scripts/check_consistency.py`):**

- **`REQUIRED_CAPS`:** caps a dep in every package that uses it. CI fails on a missing cap or an unlisted one.
- **`SCOPED_CAPS`:** caps a dep in one package only. For example, `genkit-google-genai` caps `httpx<1` because `http_client_factory` returns an `httpx.AsyncClient`, while core `genkit` keeps `httpx` floor-only.
- **Known-bad version:** exclude it with `!=x.y.z`, not a cap.

## Lower bounds (floors)

A floor promises Genkit works at that version. When a PR uses a newer API, the floor must rise in the same PR.

- **CI gate:** `lowest-deps` in `python.yml` resolves direct deps at their floors (`uv lock --resolution lowest-direct`) and runs tests on Python 3.10.
- **Finding the floor:** binary-search PyPI between the declared floor and the locked version:

  ```bash
  bin/find_floor genkit-openai openai
  # => openai==1.64.0: FAIL (3 failed, 486 passed)
  #    openai==1.68.0: PASS (489 passed)
  #    genkit-openai: openai lowest passing 1.68.0 (declared floor 1.60.0)
  ```

  If that's too slow, use the locked version. A high floor fails loudly at install time; a low floor fails at runtime.

Don't bump floors just because a newer version is out. Only bump on purpose when:

- **Dropping support:** do it in a Genkit minor release and note it in the release notes.
- **Reachable CVE:** our code path hits the vulnerable code. Comment the reason next to the floor.

## Workspace-only deps

- **Dev and lint groups (root `pyproject.toml`):** cap at the next major. They never ship; caps only bound `uv lock --upgrade`.
- **Tests and samples:** declare every package you import. Don't rely on transitive pulls. Samples are workspace members, so keep their lists minimal.

## Lockfile updates & CI

Don't run `uv lock --upgrade` in feature PRs—it moves every uncapped dep at once and hides what broke.

- **Weekly minor/patch PR:** one grouped Dependabot PR after a 7-day cooldown. Merge if green. If red, fix our code rather than adding a cap.
- **Weekly major PRs:** one PR per dep so a broken major doesn't block the rest. Lockstep families (`opentelemetry-*`, `boto3` + `botocore`) stay grouped so they can resolve.
  - **Floor-only dep:** users already resolve the new major. Green: merge same day. Red: users are already broken—fix immediately.
  - **Capped dep:** widens the cap (`<3` -> `<4`). Check `packages/*/pyproject.toml`, skim the changelog for behavior mocked tests miss (exceptions, transport, defaults), merge, and release so users get the wider range.
  - **Contract dep (`pydantic`, `dotpromptz`):** widening affects user code. Merge deliberately with a release note.
  - **Red capped PR:** leave it open as the migration ticket; the cap protects users meanwhile.
- **Daily canary (`python-deps-canary.yml`):** tests the newest versions our bounds allow—what `pip install` gets today. Not a PR gate since upstream releases trigger it. Opens/updates one `py-deps-canary` issue while red and closes it when green; fix the code or add `!=x.y.z`, and cut a patch release if users are hit.
- **Security alerts:** Dependabot security PRs bypass cooldown and grouping. To clear a batch manually:

  ```bash
  # 1. Audit locked packages
  uv export --frozen --all-packages --all-groups --no-hashes --no-emit-workspace -o /tmp/req.txt
  uvx pip-audit -r /tmp/req.txt --no-deps --disable-pip

  # 2. Upgrade only the flagged packages within current bounds
  uv lock --upgrade-package urllib3 --upgrade-package cryptography
  ```

  If a package won't budge, find what holds it back with `uv tree --invert --package <name>` and upgrade that too.
