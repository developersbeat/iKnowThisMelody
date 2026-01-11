# iKnowThisMelody 🎵
B2C app that finds **similar melodies/tunes** (not “same song like Shazam”), using **open-source only**:
- **Backend**: Python + FastAPI + PyTorch + FAISS
- **Web**: Next.js (TypeScript)
- **Mobile**: React Native (Expo, TypeScript)
- **Audio**: FFmpeg normalization + melody/pitch features + embeddings
- **Search**: FAISS vector index (+ optional DTW reranking)

> Goal: user records/uploads a 5–20s clip → app returns “Very close / Similar / Loose” melody matches.

---

## 0) Repo Layout (Monorepo)
iknowthismelody/
README.md
.gitignore
docker-compose.yml
Makefile
backend/
pyproject.toml
app/
main.py
api/
routes.py
deps.py
core/
config.py
logging.py
services/
audio_io.py
features.py
embeddings.py
index_store.py
search.py
models/
schemas.py
jobs/
queue.py
worker.py
db/
base.py
session.py
models.py
scripts/
build_index.py
import_dataset.py
tests/
test_health.py
data/
library/           # optional: audio library (local dev only)
index/             # faiss index + metadata
temp/              # uploaded clips (short-lived)
web/
package.json
next.config.js
src/
app/
components/
lib/
mobile/
package.json
app.json
src/
screens/
components/
lib/

---

## 1) What Codex Should Build (High-Level)

### Backend must provide:
- `POST /v1/search` → accepts audio clip; returns `{ job_id }` (async) OR `{ results }` (sync mode for MVP)
- `GET /v1/search/{job_id}` → returns status + results
- `GET /health` → ok
- `POST /v1/admin/build-index` → build FAISS index from local dataset (dev-only)
- `GET /v1/library/stats` → index stats (count, updatedAt)

### Web app must provide:
- Record audio (browser `MediaRecorder`)
- Upload to backend `/v1/search`
- Poll job endpoint (if async) and display results:
  - **Very Close Match**, **Similar Melody**, **Loose Similarity**
  - Confidence meter (0–100)
  - (Dev) show debug: pitch contour plot optional (no heavy charting required)

### Mobile app must provide:
- Record audio (Expo AV)
- Upload to backend `/v1/search`
- Poll and display results similar to web

---

## 2) Core Matching Approach (Open-Source Only)

We combine 2 signals:

1) **Embedding similarity** (fast retrieval)
- Use a free embedding model. MVP: start with a simple Torch audio embedding or OpenL3-like approach (open weights).
- Store embeddings per track segment in FAISS.

2) **Melody features rerank** (better “copy-paste tune” feel)
- Extract pitch contour (F0) using a free method:
  - MVP: `librosa.pyin` (pure Python) OR optional CREPE if available.
- Normalize contour (key-invariant, tempo-tolerant)
- Compute DTW distance (or simplified interval matching) to rerank top candidates.

**Final score** (MVP):
- Retrieve TopK from FAISS by cosine/inner product
- Rerank TopK with melody distance
- Convert to `confidence` (0–100)

---

## 3) Requirements & Dependencies

### System requirements
- Node.js 18+
- Python 3.11+
- FFmpeg installed (required)

#### macOS
```bash
brew install ffmpeg


sudo apt-get update && sudo apt-get install -y ffmpeg
```

4) Quick Start (Local Dev)

4.1 Clone & setup
```
git clone <YOUR_REPO_URL> iknowthismelody
cd iknowthismelody
```

4.2 Start backend (Python)

Codex should create a modern Python project using uv or poetry.
Prefer: uv + pyproject.toml.

Example (uv):
```
cd backend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt   # or `uv pip install -e .`
uvicorn app.main:app --reload --port 8000
```

Backend should run at:
	•	http://localhost:8000
	•	docs: http://localhost:8000/docs


  4.3 Start web (Next.js)
  ```
cd web
npm install
npm run dev
```
Web should run at:
	•	http://localhost:3000


  5) Environment Variables

Create files:

backend/.env

```
APP_ENV=dev
APP_HOST=0.0.0.0
APP_PORT=8000

# storage
DATA_DIR=./data
TEMP_DIR=./data/temp
INDEX_DIR=./data/index

# indexing
FAISS_INDEX_PATH=./data/index/faiss.index
FAISS_META_PATH=./data/index/meta.json

# optional DB (start with sqlite for MVP)
DATABASE_URL=sqlite:///./data/app.db

# CORS
CORS_ORIGINS=http://localhost:3000,exp://localhost:19000
```

web/.env.local
```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

mobile/.env (or app config)

```
EXPO_PUBLIC_API_BASE_URL=http://localhost:8000
```

6) Dataset & Indexing (Dev-Only for MVP)

6.1 How the MVP library works

For MVP, you’ll build an index from local audio files placed in:

```
backend/data/library/
  track_0001.mp3
  track_0002.wav
  ...
```


Codex must create a script:
	•	backend/app/scripts/build_index.py
that:

	1.	iterates over files in data/library
	2.	normalizes audio with ffmpeg to mono 16k WAV
	3.	splits into segments (e.g., 10s windows with 5s hop)
	4.	extracts:
	•	embeddings (for FAISS)
	•	melody features (for rerank, optionally stored in meta)
	5.	saves:
	•	FAISS index to data/index/faiss.index
	•	metadata JSON to data/index/meta.json

6.2 Build the index

Once backend runs:


```
python -m app.scripts.build_index --library ./data/library --out ./data/index
```

Or via API (dev):
```
curl -X POST http://localhost:8000/v1/admin/build-index
```

6.3 Index metadata shape (example)

meta.json should contain:

```
{
  "version": 1,
  "createdAt": "2026-01-12T00:00:00Z",
  "items": [
    {
      "id": "track_0001__seg_000",
      "trackId": "track_0001",
      "title": "Track 0001",
      "artist": "Unknown",
      "source": "local",
      "segmentStartSec": 0,
      "segmentDurationSec": 10,
      "melodyFeaturePath": "melody/track_0001__seg_000.npy"
    }
  ]
}

```

7) API Contract (Must Match in Web & Mobile)

7.1 POST /v1/search

multipart/form-data
	•	file: audio clip (.m4a, .wav, .mp3, .webm)
	•	optional mode: song_clip (default) | later hum
	•	optional top_k: integer default 10

Response (async MVP)

```
{ "job_id": "abc123" }
```
Response (sync option)
```
{
  "results": [
    {
      "trackId": "track_0001",
      "title": "Track 0001",
      "artist": "Unknown",
      "confidence": 92,
      "matchType": "very_close",
      "segmentStartSec": 15,
      "debug": { "faissScore": 0.83, "melodyDistance": 0.12 }
    }
  ]
}
```

7.2 GET /v1/search/{job_id}
```
{
  "status": "queued|processing|done|error",
  "results": [],
  "error": null
}
```

8) Scoring & Match Buckets (User-Facing)

Codex must implement bucketing:
	•	confidence >= 85 → very_close
	•	70–84 → similar
	•	55–69 → loose
	•	<55 → optionally hide or show as “maybe”

Confidence should be derived from a blend of:
	•	FAISS similarity (normalized)
	•	melody distance (inverted)

⸻

9) Backend Implementation Details (What to Build)

9.1 Audio normalization (required)

Implement services/audio_io.py:
	•	Save upload to TEMP_DIR with unique id
	•	Run ffmpeg to produce:
	•	mono
	•	16k sample rate
	•	wav PCM16
	•	Return path to normalized wav + duration

9.2 Embeddings

Implement services/embeddings.py:
	•	MVP embedding: simple Torch model:
	•	log-mel spectrogram + small CNN encoder (lightweight)
	•	OR use a lightweight open model with weights committed (if license allows)
	•	Output embedding dimension fixed (e.g., 256)

9.3 Melody features

Implement services/features.py:
	•	pitch contour extraction using librosa.pyin (default)
	•	normalize pitch:
	•	convert Hz to MIDI
	•	subtract median pitch (key invariance)
	•	downsample to fixed length (tempo tolerance)
	•	store as numpy array for each segment

9.4 FAISS index store

Implement services/index_store.py:
	•	load index at startup (if exists)
	•	add vectors & metadata
	•	search topK
	•	persist index to disk

9.5 Search pipeline

Implement services/search.py:
	•	normalize uploaded clip
	•	compute query embedding
	•	FAISS search topK (e.g., 50)
	•	load melody vectors for candidates
	•	compute DTW distance (fast DTW or librosa.sequence.dtw)
	•	rerank
	•	map to track-level results (group segments by track)
	•	return top 10 tracks with confidence

9.6 Async job queue (MVP-friendly)

Implement jobs/queue.py with a simple in-memory queue for dev:
	•	store job states in dict keyed by job_id
	•	worker thread/process consumes jobs
	•	production-ready alternative (later): Redis + RQ/Celery

For MVP, in-memory is OK.

⸻

10) Web App (Next.js) Requirements

Pages
	•	/ Home:
	•	record button (start/stop)
	•	file upload fallback
	•	“Search melody” CTA
	•	/results/[jobId] (or modal):
	•	polling status
	•	list results cards:
	•	title, artist
	•	confidence badge
	•	matchType label
	•	(dev) debug values

Components
	•	Recorder.tsx (MediaRecorder)
	•	Uploader.tsx (file input)
	•	ResultsList.tsx
	•	ConfidenceBar.tsx

UX
	•	show recording timer (max 20s)
	•	show “Processing…” loader
	•	show errors clearly (bad file, no index, etc.)

⸻

11) Mobile App (React Native Expo) Requirements

Screens
	•	HomeScreen
	•	record audio
	•	upload
	•	ResultsScreen
	•	polling
	•	results list

Components
	•	AudioRecorder using expo-av
	•	ResultsList
	•	ConfidenceBar

Notes
	•	handle permissions (mic)
	•	enforce max recording length (20s)

⸻

12) Docker (Optional but Helpful)

Create docker-compose.yml at repo root:
	•	backend service builds from backend/
	•	web service builds from web/
	•	expose ports 8000 and 3000
	•	mount backend/data as a volume for index persistence

For MVP, Docker is optional. But Codex should include it.

⸻

13) Commands (Makefile)

Create a Makefile with:
	•	make dev-backend
	•	make dev-web
	•	make dev-mobile
	•	make build-index
	•	make test

⸻

14) Safety & Legal Messaging (B2C)

Codex should include UI copy & disclaimers:
	•	Use “melody similarity” / “tune resemblance”
	•	Avoid: “plagiarism”, “copied”, “stolen”
	•	Disclaimer:
	•	“Results are algorithmic similarity estimates and not legal determinations.”

⸻

15) MVP Defaults & Constraints
	•	clip length: 5–20 seconds
	•	allowed formats: wav/mp3/m4a/webm
	•	index must exist; if not, return helpful error:
	•	“Library index not built. Add audio to backend/data/library and run build_index.”

⸻

16) Testing

Backend tests (pytest):
	•	health endpoint
	•	search endpoint returns 400 if no file
	•	search endpoint returns 503 if index missing
	•	build index script creates index files

⸻

17) What Codex Must Do Now (Task Checklist)

A) Initialize repo
	•	create monorepo folders: backend/ web/ mobile/
	•	add .gitignore, docker-compose.yml, Makefile

B) Backend
	•	FastAPI app with routers + config
	•	audio normalization via ffmpeg
	•	embedding + melody feature extraction
	•	FAISS index load/save
	•	search pipeline + rerank
	•	build index script
	•	optional async job system (in-memory)

C) Web
	•	Next.js TS app with recorder + upload + results UI
	•	environment-based API base URL
	•	polling logic

D) Mobile
	•	Expo RN TS app with audio recording + upload + results UI
	•	permissions + polling

E) Documentation
	•	ensure this README is accurate to the generated code
	•	include troubleshooting notes

⸻

18) Troubleshooting Notes (Must Include)
	•	If ffmpeg missing → backend error; show install steps
	•	If index missing → show how to build it
	•	If mobile can’t reach backend → use LAN IP, not localhost
	•	If pitch extraction fails on noisy audio → suggest longer clip or cleaner sample

⸻

19) Future Enhancements (Not Required for MVP)
	•	humming mode (query-by-humming)
	•	better pitch extractor (CREPE)
	•	move jobs to Redis queue
	•	swap FAISS for Milvus if massive scale
	•	user accounts + history + subscriptions

⸻

Done Definition ✅

When finished, I can:
	1.	Put audio files into backend/data/library
	2.	Run make build-index
	3.	Start backend + web + mobile
	4.	Record a melody clip and see ranked similar results

⸻

NOTE TO CODEX (Implementation Guidance)
	•	Prefer simple, working MVP over perfection.
	•	Keep libraries minimal.
	•	Make sure every service runs end-to-end with no missing imports.
	•	Use TypeScript everywhere in web/mobile.
	•	Don’t add paid services or proprietary APIs.
	•	Use clear error messages and consistent JSON contracts.

⸻



