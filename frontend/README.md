# GHOSTSIGNAL — animated editorial frontend prototype

React + TypeScript + Vite frontend for GHOSTSIGNAL.

## Pages

- `#investigate` — investigation landing page and source selection.
- `#reports` — report archive preview.
- `#method` — five-stage investigation methodology.
- `#about` — product scope and operating principles.
- `#investigation` — animated investigation-in-progress view.
- `#results` — findings overview with Activity / Positive vs Negative chart modes.
- `#report` — full-report layout preview.

The four main navigation destinations are real client-side pages and can be opened directly through their hash URLs. During an active investigation, the header transitions from the main navigation into an investigation context with a live stage indicator and investigation ID; the main navigation returns on the results and informational pages.

## Run locally

```bash
npm install
npm run dev
```

## Visual and motion details

- Animated line-drawing wave graphic with pinned annotations on the home screen.
- Animated network nodes, connecting edges, community pulses and relationship bridges on the investigation screen.
- Animated header transition, page transitions, progress stages, metric entrances, chart line drawing and classification proportion bars.
- Content classification is displayed as proportion bars; the previous coloured tile/block grid has been removed.
- Responsive desktop and mobile layouts.

## Prototype limitations

- Investigation progress is simulated for the frontend demo. Replace it with backend status events.
- Metrics, activity, sentiment, network communities and observations are illustrative sample data.
- Reports page currently contains one local demo record; persistent report storage is not connected.
- Gemini report writing and server-side PDF generation are not connected.
- Keep API keys on the backend; never expose them in frontend code.

## Planned backend contract

Connect the investigation screen to backend status events and replace sample result data with the structured result object. The report screen should request a server-generated PDF and only enable download after the backend confirms the artifact is ready.
