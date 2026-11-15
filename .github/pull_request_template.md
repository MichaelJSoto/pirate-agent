## Problem

What is wrong or missing, and how do we know? Link the code review finding or issue if there is one.

## Decision

What this change does, and why this approach. Name at least one alternative and why it was not
chosen.

## Testing

- [ ] New or changed behavior has tests (a bug fix includes a test that failed before the fix)
- [ ] `ruff format .`, `ruff check .`, `codespell` and `python -m pytest` pass locally
- [ ] Output shown below where it helps a reviewer (for example a training summary or an error message)

## Security

What input does this change read or trust? How is it validated? Write "None" if the change
touches no input, file, dependency or model loading.

- [ ] New dependencies are pinned and pass `pip-audit --strict -r requirements.txt`
- [ ] Models load only through `load_network()`

## Checklist

- [ ] Every new file has the header (programmer, date, version, intent, decision)
- [ ] README, CHANGELOG and configuration comments are updated if behavior changed
