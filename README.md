# AERIS-MK2

AERIS-MK2 is a Class XII AI educational project for aerospace telemetry. The
web platform supports private login, strict CSV validation, local descriptive
statistics and charts, optional Gemini explanations, and optional Neon
PostgreSQL history.

## Machine-learning method and syllabus status

The supplied workspace did not contain the actual CBSE curriculum or student
handbook files. The project requirements identify Linear Regression as a
covered method, but do not establish whether **residual-threshold anomaly
detection** is authorized. The implementation therefore isolates the
additional step and documents it as unresolved; do not describe the complete
forecasting-plus-threshold method as syllabus-compliant without teacher
approval.

The implemented predictor is a per-channel, multi-output
`sklearn.linear_model.LinearRegression` forecaster. For every supported NASA
channel, it uses the complete feature vector at record *t* to predict the
vector at *t+1*. Separate `StandardScaler` preprocessing is fitted to the
training partition only. The optional anomaly flag is a separate step: calculate
per-record multivariate residual RMSE, then flag values greater than the
channel-specific 99th percentile measured on validation data. These are
statistical warnings, not confirmed faults or diagnoses.

`scripts/train_model.py` trains only when explicitly invoked. It chronologically
partitions each supplied training sequence into contiguous 60% fit, 20%
validation, and 20% unused ranges; one-step pairs never cross partition
boundaries. The official, separate `Dataset/data/test` sequences are the final
held-out evaluation data. The split assumes each array's row order is temporal;
that assumption should be checked against authoritative dataset documentation.
This does not demonstrate that the supplied train/test sequences are
independent in collection time.

Held-out MSE, RMSE, and MAPE are regression-error metrics over scalar feature
predictions. MAPE omits zero targets and reports the omission count; even for
nonzero targets it can be unstable near zero. Binary precision, recall, F1, and
accuracy are calculated separately only for unique, length-matched anomaly
labels. The conflicting `P-2` rows and unlabeled `T-10` are excluded from
classification evaluation. The label intervals are interpreted as inclusive,
which is an explicit dataset assumption.

Training writes `models/aeris_linear_forecaster.joblib` and
`models/evaluation_report.json` atomically as individual files. The report
records the preprocessing/split configuration and library versions, and
contains a shared artifact ID so interrupted or mismatched output pairs are
rejected. Generated model files are ignored by Git. The app never trains at
startup: it loads the report for display and loads the saved model only when
inference is requested. Missing, incompatible, or corrupt artifacts produce an
explanation while descriptive analysis remains available.

The NASA training run completed successfully in the existing Python 3.12
environment for all **82 channels**. The evaluated test arrays produced
**14,984,705 scalar one-step predictions**:

| Held-out regression metric | Measured value |
| --- | ---: |
| MSE | 59.75389892464016 |
| RMSE | 7.730064613225439 |
| MAPE | 2694.8486136126294% |
| Zero targets excluded from MAPE | 14,339,040 |

MAPE's very large value is reported as measured and is sensitive to small
nonzero telemetry values; zero targets are excluded rather than assigned an
arbitrary percentage. Separately, residual-threshold flags were compared with
valid labels for 80 channels / 501,266 records: confusion counts were TP
15,702, TN 374,835, FP 63,969, FN 46,760; precision 0.1970855, recall
0.2513848, F1 0.2209480, and accuracy 0.7791013. These are label-based
classification metrics, not regression accuracy, and remain subject to the
unresolved syllabus caveat and label assumptions above.

Artifacts were saved at `models/aeris_linear_forecaster.joblib` and
`models/evaluation_report.json`. A fresh process loaded the model and
preprocessing artifacts and produced finite, correctly shaped predictions for
a sample from held-out test channel `A-1`.

Dataset inspection found 82 training and 82 testing `.npy` arrays, each with
25 or 55 features, plus 82 HDF5 model files and precomputed output arrays.
Those existing model artifacts are not loaded or claimed as models trained by
this project. `labeled_anomalies.csv` has 82 rows but 81 distinct channel IDs:
the two `P-2` entries contain different intervals, and `T-10` has no label
row. Resolve the duplicate before any label-based evaluation; do not evaluate
`T-10` against missing ground truth.

## Features

- Predefined private users with Werkzeug password hashes, Flask sessions,
  logout, and CSRF protection. There is no public registration or OAuth.
- CSV uploads limited to 5 MB, 100,000 rows, and 55 sensors.
- Local descriptive statistics: minimum, maximum, mean, median, population
  standard deviation, first/last values, net endpoint change, and histogram
  bins. These are descriptions, not anomaly-detection predictions; endpoints
  do not establish a sustained time trend.
- Responsive interactive line and distribution charts based on uploaded
  values. Chart.js is loaded from a CDN; if it is unavailable, the numeric
  results remain usable.
- Optional Gemini explanations from bounded local-statistics and, when
  available, computed model-output summaries.
  Gemini does not receive raw telemetry rows and is instructed not to invent
  model predictions, anomaly labels, metrics, units, or sustained temporal
  trends. Its summary includes only first/last values and net endpoint changes.
  If the key, SDK, or service is unavailable, local results still display.
- Optional Neon PostgreSQL history storing the filename, record count,
  selected model channel, compact held-out metrics, calculated sensor
  statistics, and Gemini insight text—not telemetry rows or uploaded values.
- Optional saved-artifact inference for a user-selected trained NASA channel.
  The upload remains descriptive-only unless the user chooses a channel and
  confirms that the ordered CSV columns map to that channel's feature order.
- Dataset inspection script that inventories files and reads `.npy` headers
  without loading whole arrays.

## Python environment and dependencies

Use the existing Python 3.12 environment. NumPy, scikit-learn, joblib, Flask,
pytest, psycopg, and the Gemini SDK were importable in the verified environment.
On Windows, Gunicorn is not required and is declared only for non-Windows
platforms. Do not reinstall or upgrade packages when the existing imports work.

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The application can be developed and tested without actual Gemini or Neon
credentials. No dependency installation or virtual-environment creation is
performed by the project code.

## Local configuration and login

Copy `.env.example` to `.env`. Generate a strong secret with the existing
environment:

```powershell
.\venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set a unique random `FLASK_SECRET_KEY` (at least 32 characters) and
configure at least one predefined username and password hash. Generate a
hash with the helper; it prompts without echoing the password:

```powershell
.\venv\Scripts\python.exe scripts\generate_password_hash.py
```

Set these values in `.env` (up to three user slots are supported):

```dotenv
AERIS_USER_1_USERNAME=operator
AERIS_USER_1_PASSWORD_HASH=paste-the-generated-hash
```

Configure a matching username and hash as a pair. Leave unused slots blank.
Plaintext passwords are not read from the environment or stored.

Start the development app:

```powershell
.\venv\Scripts\python.exe run.py
```

Open <http://127.0.0.1:5000>.

## Supported uploaded CSV format

The analysis endpoint accepts only UTF-8 CSV files using a strict wide format:

```csv
record_index,sensor_1,sensor_2
0,12.5,4.2
1,12.7,4.1
2,12.6,4.3
```

Requirements:

- The first header must be `record_index`.
- Sensor headers must be consecutive and ordered: `sensor_1`, `sensor_2`, and
  so on (up to `sensor_55`). Do not include other columns.
- Each record index must be a strictly increasing, non-negative integer.
- Every sensor value must be a finite number; blank values, NaN, and infinity
  are rejected. Missing values are not imputed.
- At least one data record is required. Rows with the wrong number of fields
  and duplicate headers are rejected.
- File size is limited to 5 MB and records to 100,000.

The same format is the model input contract only when the user explicitly
selects a channel and confirms that columns `sensor_1` through `sensor_n` are
in the exact feature order used for that NASA channel. A selected model also
requires exactly that channel's feature count and record indices increasing by
one. No sensor name or arbitrary telemetry file is automatically mapped to a
NASA channel; a successful mapping is the uploader's assertion, not verified
provenance or units. Invalid/incompatible mapping skips inference without
blocking descriptive analysis. Leave the channel blank for descriptive-only
results.

For a model-enabled result, the graph overlays the first selected feature with
its one-step predictions and residual-threshold flags. The table shows recent
first-feature predictions. The held-out evaluation block is clearly separate
from the current-upload output; the upload has no ground-truth label, so no
upload accuracy is calculated or claimed. Gemini receives bounded computed
summaries and may explain them; it does not generate predictions, scores, or
flags.

### Train locally

After the dependencies are available in the existing environment, explicitly
run:

```powershell
.\venv\Scripts\python.exe scripts\train_model.py
```

The script reads `Dataset\data\train`, `Dataset\data\test`, and
`Dataset\labeled_anomalies.csv`. It validates IDs, dimensions, numeric
integrity, and label alignment, and fails without writing success-shaped
results if prerequisites are missing. It does not modify source data or train
on Render. Override paths only when needed:

```powershell
.\venv\Scripts\python.exe scripts\train_model.py --dataset Dataset --output models
```

By default, artifacts are saved under the repository's ignored `models\`
directory. Set `MODEL_ARTIFACT_DIRECTORY` in `.env` to use another trusted,
readable artifact directory. The Flask app will not retrain. A deployment such
as Render must receive the model and matching report through a trusted,
explicit artifact deployment process; Render's ephemeral filesystem is not a
durable model store. Never accept or load user-supplied joblib files.

## Dataset inspection

The original `Dataset\` directory is excluded from Git and left unchanged.
Inspect formats and array headers with:

```powershell
.\venv\Scripts\python.exe scripts\inspect_dataset.py Dataset
```

Stream all train/test numeric values and audit label intervals/channel
alignment with:

```powershell
.\venv\Scripts\python.exe scripts\audit_dataset_values.py Dataset
```

The values audit reads one row at a time; it reports each channel's dimensions,
dtype, finite-value status, range, mean, zero count, and a first-row sample.
It lists label intervals and explicitly excludes ambiguous or unlabeled
channels from possible supervised evaluation. Neither inspection tool alters
the source data or loads all arrays into memory.

## Gemini configuration

Gemini is optional. Set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`; the
default model name is `gemini-2.5-flash`. The key is only read server-side.
Gemini receives at most eight channel summaries (counts and descriptive
statistics), plus at most five computed prediction examples and a short sample
of model-flagged indices when inference ran; it never receives uploaded rows.
It does not generate or replace local analysis, predictions, or model metrics.
Returned JSON is checked for the expected fields and size. Missing keys,
malformed responses, network failures, and quota/service errors leave the local
charts and statistics available.

Gemini text is generated content; review it against the displayed local
statistics before relying on it.

## Neon PostgreSQL history

History is optional. Put the Neon-provided PostgreSQL connection string in
`NEON_DATABASE_URL`. Initialize the schema explicitly (not during app startup).
The initializer also adds the descriptive-statistics column if an earlier
version of the project's table already exists:

```powershell
.\venv\Scripts\python.exe scripts\init_database.py
```

When a database is configured and reachable, a successful upload saves a
summary record. If configuration is absent or a database operation fails,
analysis results still display and the app reports that history was not saved.
Connection credentials are not included in user-facing errors or logs.

## Tests and validation

No packages were installed, and the current environment does not have the
declared Flask, pytest, psycopg, dotenv, or Gemini dependencies. Python syntax
can be checked with:

```powershell
.\venv\Scripts\python.exe -m compileall -q app scripts run.py tests
```

After dependencies are available, run unit and Flask workflow tests without
external credentials or a live database:

```powershell
.\venv\Scripts\python.exe -m pytest
```

The Gemini and Neon clients are mocked in tests. The dataset audit can be run
independently using the existing environment:

```powershell
.\venv\Scripts\python.exe scripts\audit_dataset_values.py Dataset
```

The complete pytest suite passed: **49 passed**. The end-to-end Flask test
client workflow also exercised login, a held-out NASA CSV upload, inference
from the saved artifact, result/evaluation rendering, and graceful behavior
with Gemini and Neon unset. Live Gemini and Neon connections were not tested;
the offline Gemini and database code paths are covered with mocks.

## Render deployment

For a future deployment, configure `FLASK_SECRET_KEY`,
`AERIS_USER_1_USERNAME`, `AERIS_USER_1_PASSWORD_HASH`, and
`NEON_DATABASE_URL` as Render environment variables; configure Gemini
variables only if insights are desired. Set `APP_ENV=production` and use
`gunicorn run:app` as the start command. Initialize the database as a separate
setup step. Render does not train a model. Supply the trusted model artifact
and its matching evaluation report using a deployment mechanism appropriate
for the service's filesystem and resource limits; inference loads only that
saved model, not the complete dataset. Keep the artifact outside the Git
repository and do not load an untrusted artifact.

This is a school-project prototype, not a certified aerospace, classified, or
mission-critical system.
