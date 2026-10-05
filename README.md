# Eyes Over the Field: participant kit
AeroClub IIT Delhi · Tech FGC

Read `docs/PS.md` (the problem statement) and `docs/SPEC.md` (data formats and conventions).

## Setup (before the event)
```bash
pip install -r requirements.txt      # Python 3.9+
python run.py --part A --seed 1      # should print a score (0 for the unfilled template)
```

## Contents
| Path | What |
|---|---|
| `solution/solution.py` | Your code (template with TODOs). |
| `run.py` | Runs your solution in the simulator on practice worlds and prints the score. |
| `fgcsim/` | The simulator. Do not import it from your solution. |
| `docs/` | Problem statement and specification. |

Run `python run.py -h` for all options.

## Submitting
- One member zips the `solution/` folder, names it after the team (e.g. `SkyWatchers.zip`) and uploads it before the deadline announced at the briefing. Late submissions are not accepted.
- Only the last submission counts. Keep the class name `Solution` and its methods as in the template.
- Check first: `python run.py --part A B C --isolated` must run without errors.

Full rules: `docs/PS.md`, section 8.
