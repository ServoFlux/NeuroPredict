---
name: testing-archive-digitizer
description: Test the Archive MRI digitizer flow (/digitizer) end-to-end in the browser. Use when verifying that a photographed film-sheet image is reconstructed into a volume and runs the existing prediction. Also covers the general NeuroPredict scan-upload flow.
---

# Testing the Archive MRI Digitizer

The digitizer "assists" the website: it photographs a printed MRI **film sheet**
(a grid/contact-sheet of slices), the server rebuilds the slices into a 3D volume,
then runs the *existing* multimodal prediction. It does NOT add a new model signal.

## Prereqs
- Server running: `uvicorn webapp.main:app --port 8000` (model checkpoint must load).
- Sample film sheets generated. Create them with the simulator:
  `python iot/simulate_digitizer.py --etiology genetic --age 60 --no-post --save /home/ubuntu/film_samples/genetic_film.png`
  `python iot/simulate_digitizer.py --etiology no_wmd --no-post --save /home/ubuntu/film_samples/healthy_film.png`
- A public URL if testing remotely (cloudflared tunnel). Tunnels are temporary.
- Run `pytest -q` (expect all green) and `ruff check .` before browser testing.

## Browser test procedure
1. Open `/` → click the **Archive MRI digitizer** link → lands on `/digitizer`.
2. Click **Choose File**. The native GTK dialog opens. To set the path reliably,
   press `ctrl+l` and type the absolute path, then Enter. IMPORTANT: only press
   `ctrl+l` AFTER confirming the dialog is open — if the dialog did not open,
   `ctrl+l` goes to the browser address bar and you'll navigate away / open the
   image as a `file://` page. Always screenshot to confirm the dialog is up first.
3. The "Choose File" button shifts down when the "Last digitized capture" banner
   grows — re-screenshot and click its current position, don't reuse coordinates.
4. Fill questionnaire (Age + checkboxes). Submit **Digitize & predict**.
5. Verify the result page: file line shows `<name> (digitized film)`, the WMD
   probability, the Grad-CAM heatmap, and (if disease) the cause ranking + next steps.

## What to assert
- Disease yes/no detection: genetic/diseased film → high WMD (>70%); healthy film
  → low WMD (<25%), "No white matter disease", no cause section. This is reliable.
- Cause/etiology on digitized film: **may be unreliable**. Photographing film blurs
  the lesion *shape* that distinguishes causes, so a genetic film can come back as
  "Metabolic" even with NOTCH3/APOE checked (the questionnaire moves the result only
  a fraction of a point because the image dominates). If you need to demo a clean
  cause result, use the native scan-upload flow at `/` with a `.nii.gz` volume
  instead of the digitizer. Report cause-on-film as a known fidelity limitation,
  not a regression.

## Gotchas
- The `/digitizer` "Last digitized capture" banner reflects the most recent capture
  globally (in-memory `_latest_digitized`); it updates across requests — handy as an
  incidental check that a prior submission was processed.
- Default grid is cols=8, depth=64; the demo films are generated to match.

## Devin Secrets Needed
- None for local/synthetic testing.
- (Deferred) `HUGGINGFACE_TOKEN` (Write) only if deploying to a permanent HF Space.
