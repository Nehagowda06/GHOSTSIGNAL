# GHOSTSIGNAL --- Project Brief, Product Specification & Team Handbook

> **Purpose:** This is the single source of truth for every team member
> and every AI assistant used during the five-hour hackathon. Read it
> before designing, coding, or changing the project. If an AI assistant
> proposes something that conflicts with this document, follow this
> document unless the team explicitly agrees to revise it.

------------------------------------------------------------------------

## 1. Project identity

**Project name:** GHOSTSIGNAL\
**Product category:** Cross-Platform Digital Influence & Narrative
Intelligence\
**Working tagline:** Understand how information emerges, spreads,
evolves and gains influence across the internet.

### One-sentence description

GHOSTSIGNAL is an on-demand investigation tool that collects permitted
public information from selected online sources, analyses how content is
classified and how narratives appear or evolve, and produces an
evidence-backed report containing source references, timelines,
engagement indicators and potential coordination signals.

### The problem

Online information is fragmented across search results, product reviews,
news coverage and public discussions. Investigating a topic manually
requires locating content, comparing wording, establishing chronology,
inspecting engagement and distinguishing ordinary discussion from
patterns that warrant further examination.

GHOSTSIGNAL brings those steps into one workflow.

### Product purpose

Given an investigation topic, GHOSTSIGNAL should:

1.  Obtain real-world information from supported sources.
2.  Preserve the source and provenance of each record.
3.  Clean and normalize collected records.
4.  Classify content using a trained machine-learning model.
5.  Analyse similarities, dates and available engagement information.
6.  Present findings with evidence, caveats and source links.
7.  Generate an investigation report.

**The report is the product's main output.** GHOSTSIGNAL is not an
autonomous moderation or response system.

------------------------------------------------------------------------

## 2. Non-negotiable product boundaries

### GHOSTSIGNAL DOES

-   Accept a topic, query, product, brand, claim, event or URL for
    investigation.
-   Retrieve information through permitted APIs, approved connectors,
    RSS feeds, authorized datasets or user-provided public URLs.
-   Work with Google/web search results and Amazon review records for
    the initial MVP.
-   Normalize data from different sources into one common schema.
-   Classify collected content into defined categories.
-   Identify semantically similar content and repeated narrative
    patterns.
-   Reconstruct the earliest appearance **observed in the collected
    records**, not claim the true origin of a narrative.
-   Summarize available publication and engagement information.
-   Display source links and evidence.
-   Report uncertainty, source coverage and data limitations.

### GHOSTSIGNAL DOES NOT

-   Ban, suspend, report, message or contact accounts.
-   Remove, edit, suppress or manipulate online content.
-   Execute moderation actions or intervene on any platform.
-   Claim that similar content proves a coordinated campaign.
-   Claim that a post is fake, sponsored or paid for without evidence.
-   Claim that the earliest collected record is the actual origin of a
    narrative.
-   Claim to scan an entire platform when it only has access to search
    results, snippets, limited API responses or an uploaded dataset.
-   Invent missing dates, authors, ratings, views, likes, comments or
    other metrics.
-   Identify the same person across platforms based only on similar
    usernames or writing style.

**Important:** Coordination is a hypothesis supported by multiple
observable signals, not a conclusion derived from similarity alone.
Similar opinions can arise independently. Coordinated activity is not
automatically false or malicious.

------------------------------------------------------------------------

## 3. Five-hour MVP scope

The hackathon is **five hours**, not five days. The goal is a small,
working end-to-end demonstration---not a production-scale platform.

### Initial supported sources

  -----------------------------------------------------------------------
  Source                  MVP approach            Expected data
  ----------------------- ----------------------- -----------------------
  Google / web search     Google Search grounding Search result title,
                          or another permitted    snippet or retrieved
                          search integration      text, URL,
                                                  publisher/source and
                                                  available date

  Amazon                  Authorized review data, Review title/text,
                          a permitted export, or  rating, product
                          a clearly labelled real identifier/name, date
                          review dataset          and available
                                                  helpfulness metrics
  -----------------------------------------------------------------------

Do not assume a Google search result provides the full article body. Do
not assume an Amazon API provides unrestricted access to every public
review. Record what was actually retrieved.

### Out of scope for this hackathon

-   Reddit, X, Instagram, Facebook and other social connectors.
-   Live television broadcast ingestion.
-   Full news archive ingestion.
-   Real-time continuous monitoring.
-   Complex graph neural networks.
-   Deep-learning model training from scratch.
-   Multi-agent orchestration.
-   Production authentication, billing and multi-tenancy.
-   Autonomous platform actions.

The architecture may allow later expansion, but do not spend MVP time
implementing future connectors.

### Primary demo workflow

1.  User opens GHOSTSIGNAL.
2.  User enters an investigation query, for example a product, brand or
    public claim.
3.  User selects Google/web and/or Amazon.
4.  The application obtains available records.
5.  Records are normalized and cleaned.
6.  The trained model classifies the content.
7.  Analysis calculates similarity, chronology and available engagement
    indicators.
8.  The interface displays records, classifications, visual summaries
    and limitations.
9.  User generates or views an evidence-backed report.

If one source is unavailable, the application must explain that
limitation and continue with the available source when possible.

------------------------------------------------------------------------

## 4. Core motto and implementation pipeline

The team must build around this exact sequence:

**OBTAIN REAL-WORLD DATA → TRAIN OUR MODEL → OBTAIN RESULTS → DISPLAY
THEM PROPERLY WITH CLASSIFICATION**

### Stage A --- Obtain real-world data

-   Collect from permitted Google/web search and Amazon review sources.
-   Keep raw records unchanged in the raw-data area.
-   Preserve source URLs, source type and retrieval metadata.
-   Do not fabricate real-world records or metrics.
-   Keep a clear distinction between live-collected data and
    fallback/test data.

### Stage B --- Prepare and train the model

-   Clean and normalize records.
-   Create a labelled training dataset.
-   Train a lightweight text classifier.
-   Evaluate it on held-out labelled examples.
-   Save the fitted model and vectorizer for reuse.

### Stage C --- Obtain analytical results

-   Predict a content category for each record.
-   Calculate model confidence/probability where supported.
-   Group similar text or identify repeated wording.
-   Build a chronological view from valid dates.
-   Summarize engagement metrics that are actually available.
-   Flag potential patterns for human interpretation.

### Stage D --- Display results

-   Show the investigation query and source coverage.
-   Show records with predicted categories and links.
-   Show publication timeline where dates exist.
-   Show engagement metrics without mixing incompatible measures.
-   Show similarity/coordination indicators with cautious wording.
-   Produce a concise report with evidence and limitations.

------------------------------------------------------------------------

## 5. Classification taxonomy

The initial model uses four categories. Labels must be consistent across
preprocessing, training, backend and frontend.

  ----------------------------------------------------------------------------
  Canonical label        Display name      Meaning           Examples
  ---------------------- ----------------- ----------------- -----------------
  `news_reporting`       News / Reporting  Informational     News article,
                                           reporting,        factual report,
                                           article or news   event coverage
                                           coverage          

  `customer_review`      Customer Review   First-person or   Product review,
                                           evaluative        rating
                                           feedback about a  explanation,
                                           product/service   purchase
                                                             experience

  `promotional`          Promotional       Content whose     Product
                                           primary purpose   promotion,
                                           appears to be     explicit
                                           marketing or      advertisement,
                                           promotion         promotional copy

  `organic_discussion`   Organic           General           Forum-style
                         Discussion        discussion,       discussion,
                                           opinion, question question,
                                           or personal       personal opinion
                                           account not       
                                           clearly fitting   
                                           another category  
  ----------------------------------------------------------------------------

### Classification rules

-   These are **content categories**, not truth labels.
-   `promotional` does not mean covert, paid or deceptive.
-   `organic_discussion` does not mean verified organic reach or genuine
    authorship.
-   `news_reporting` does not mean the article is accurate or unbiased.
-   `customer_review` does not mean the review is authentic or verified.
-   When evidence is insufficient, the model should be allowed to return
    `uncertain` or a low-confidence result rather than forcing a
    confident category.
-   Do not add more categories during the five-hour MVP without team
    agreement.

------------------------------------------------------------------------

## 6. Data contract --- shared record schema

This contract is mandatory. All source collectors, preprocessing code,
model code and backend integration must use the same field names.

  -----------------------------------------------------------------------------------
  Field               Type                         Required? Description
  ------------------- ---------------- --------------------- ------------------------
  `id`                string                             Yes Stable unique record
                                                             identifier

  `source`            string                             Yes Canonical source name:
                                                             `google` or `amazon`

  `content`           string                             Yes Main text used for
                                                             analysis

  `title`             string/null                         No Article or review title

  `url`               string/null                         No Original source URL

  `published_at`      ISO-8601                            No Original
                      string/null                            publication/review date,
                                                             if available

  `retrieved_at`      ISO-8601 string                    Yes Time the record was
                                                             collected

  `author`            string/null                         No Public author/display
                                                             name, if available and
                                                             appropriate

  `product`           string/null                         No Product name or
                                                             identifier for Amazon
                                                             records

  `rating`            number/null                         No Amazon rating, if
                                                             available

  `likes`             integer/null                        No Like/helpful count only
                                                             if the source exposes it

  `comments`          integer/null                        No Comment count only if
                                                             available

  `shares`            integer/null                        No Share count only if
                                                             available

  `views`             integer/null                        No View count only if
                                                             available

  `label`             string/null                         No Ground-truth training
                                                             label; null for
                                                             unlabelled data

  `predicted_label`   string/null                         No Model prediction

  `confidence`        number/null                         No Model
                                                             confidence/probability
                                                             if available

  `metadata`          object                              No Source-specific extra
                                                             fields
  -----------------------------------------------------------------------------------

### Data contract rules

1.  Use `null` for unavailable values. Never substitute zero for
    unknown.
2.  Dates must be parseable ISO-8601 values and preserve timezone
    information where known.
3.  Keep the original URL wherever available.
4.  `retrieved_at` is not the same as `published_at`.
5.  A search snippet must not be represented as a full article.
6.  Keep source-specific metrics in their original meaning.
7.  Do not add or rename shared fields without notifying the team.
8.  Keep raw source responses separately from normalized records.

### Suggested formats

-   Raw collection: JSON or JSONL.
-   Normalized records: CSV or JSONL.
-   Training data: CSV.
-   Model artifacts: Joblib files.
-   Final report: HTML and/or PDF.

------------------------------------------------------------------------

## 7. Fallback and error-handling policy

The MVP must be resilient to unavailable APIs, missing fields and small
datasets.

### Collection fallbacks

-   If Google collection fails, show the source error and allow an
    Amazon-only investigation.
-   If Amazon data is unavailable, show the source error and allow a
    Google-only investigation.
-   If both sources fail, show a clear failure state. Do not create fake
    results and label them as live data.
-   If only snippets are available, label them as snippets.
-   If a source returns no publication date, leave the date blank and
    exclude the record from date-based calculations.
-   If a source does not expose engagement, show "Not available."

### Preprocessing fallbacks

-   Missing optional columns should be created with null values.
-   Empty content should be rejected or excluded from model inference,
    with a count recorded.
-   Duplicate records should be detected using stable source IDs or
    URL/content fingerprints.
-   Malformed dates should become null, not arbitrary dates.
-   Missing metrics should remain null.
-   If a source file is absent, preprocessing should continue with the
    available source and report the missing input.
-   Preserve the raw input; never overwrite it with cleaned data.

### Model fallbacks

-   If there are too few labelled examples, report that model training
    is not reliable.
-   If a class has too few examples, do not claim robust performance for
    that class.
-   If training fails, show a clear error and do not silently substitute
    hard-coded predictions.
-   If a prediction is low-confidence, mark it uncertain or require
    review.
-   A fallback sample dataset may be used for development only if it is
    clearly identified as sample/test data and never presented as
    live-collected evidence.

------------------------------------------------------------------------

## 8. Machine-learning approach

### MVP model

Use a lightweight and explainable text-classification pipeline:

1.  Text input.
2.  TF-IDF vectorization.
3.  Logistic Regression classifier.
4.  Predicted class.
5.  Class probability/confidence where available.

### Why this approach

-   Fast to train on a laptop.
-   Suitable for a small labelled text dataset.
-   Easy to serialize and reload.
-   Easy to explain to judges.
-   Establishes a real, trainable ML component without training a
    language model from scratch.

### Training requirements

-   Training examples must have one of the four canonical labels.
-   Keep a separate test split when the data volume permits.
-   Use a fixed random seed for reproducibility.
-   Report accuracy, precision, recall and F1-score.
-   Prefer macro-F1 in addition to accuracy if classes are imbalanced.
-   Avoid data leakage: near-duplicate records or examples from the same
    source item should not be split across train and test.
-   Save the fitted vectorizer and classifier together or save both with
    clear versioning.
-   Record training data size, class distribution and evaluation
    results.

### Do not claim

-   That a small hackathon model understands intent perfectly.
-   That a model's class prediction proves content is paid promotion.
-   That high accuracy on a tiny or unrepresentative test set
    generalizes to the internet.
-   That model confidence is a calibrated probability unless calibration
    has been evaluated.

------------------------------------------------------------------------

## 9. Analysis requirements

These analyses should be simple, transparent and achievable in the MVP.

### 9.1 Similarity analysis

-   Compare content using TF-IDF cosine similarity or another
    lightweight method.
-   Identify pairs/groups of text that are unusually similar.
-   Display matched records and their source links.
-   Make thresholds configurable.
-   Avoid describing similar text as proof of coordination.

### 9.2 Timeline analysis

-   Sort records by valid `published_at`.
-   Display the earliest observed record in the collected dataset.
-   Show later appearances and publication dates.
-   Exclude records with unknown or invalid dates from chronological
    claims.
-   State that the earliest observed item is not necessarily the true
    origin.

### 9.3 Engagement analysis

-   Display metrics that are actually available.
-   Keep Google/web and Amazon metrics separate.
-   Do not add ratings, likes, views, helpful votes and comments into a
    single invented engagement total.
-   Clearly distinguish unavailable metrics from zero.
-   Explain the date range and number of records used.

### 9.4 Potential coordination indicators

For the MVP, use observable indicators such as: - High textual
similarity. - Repeated phrases or templates. - Close publication timing,
if reliable timestamps exist. - Repeated URLs or source references.

Output should use language such as: - "Similar wording observed." -
"Repeated narrative pattern detected." - "Potential synchronization
signal." - "Insufficient evidence to determine coordination."

Do not infer a sponsor, campaign operator, political affiliation,
identity or intent.

------------------------------------------------------------------------

## 10. Report specification

The generated report is the central deliverable.

### Required sections

1.  **Investigation overview:** query/topic, selected sources,
    collection time, date range where known, number of records collected
    and analysed.
2.  **Executive summary:** main observations, predicted content-category
    distribution and important caveats.
3.  **Source coverage:** responding sources, available fields, whether
    records are full text/snippets/imported reviews, and known coverage
    limitations.
4.  **Classification results:** counts and proportions by category,
    individual record classifications, confidence/uncertainty where
    available.
5.  **Timeline:** earliest observed publication in collected data,
    subsequent records, missing-date count, and caveat that observed
    chronology is not proof of actual origin.
6.  **Similarity findings:** similar content groups, similarity values
    or thresholds if calculated, and links to underlying records.
7.  **Engagement:** available source-specific metrics, missing metrics
    and interpretation limitations.
8.  **Potential coordination signals:** observed indicators, alternative
    explanations and explicit uncertainty statement.
9.  **Conclusion and limitations:** what the evidence supports, what
    cannot be concluded, source coverage and collection limitations.

Every substantive finding should be traceable to one or more collected
records.

------------------------------------------------------------------------

## 11. Team division --- four teammates

The team has four members. Names can be added by the team, but role
numbering and ownership below must remain clear.

### Teammate 1 --- Data collection

**Primary ownership:** `src/collectors/`

Responsibilities:

-   Implement Google/web search collection using the selected permitted
    integration.
-   Implement Amazon review ingestion through an authorized source,
    permitted export or clearly labelled dataset.
-   Obtain real-world records for the selected investigation query.
-   Preserve source URLs and provenance.
-   Capture publication/review dates and available metrics.
-   Save raw records without modifying original content.
-   Handle missing values, API errors, rate limits and partial results.
-   Produce a short README explaining setup, required credentials, data
    fields and limitations.

Deliverables:

-   `src/collectors/google_collector.py`
-   `src/collectors/amazon_collector.py`
-   Raw JSON/JSONL/CSV outputs in agreed raw-data directories.
-   Example invocation or test command.
-   Source-access notes.

Acceptance criteria:

-   At least one source returns real records during the demo, if access
    is available.
-   Each collected record has a source identifier and retrieval
    timestamp.
-   Original URLs are preserved where available.
-   Errors and unavailable fields are reported honestly.
-   No unauthorized scraping or access-control bypass.

### Teammate 2 --- Machine-learning model

**Primary ownership:** `src/model/` and model artifacts in `models/`

Responsibilities:

-   Agree on the canonical four classification labels.
-   Consume the normalized labelled training CSV from Teammate 4.
-   Validate input schema and class distribution.
-   Implement TF-IDF + Logistic Regression.
-   Split data for evaluation where sample size permits.
-   Calculate accuracy, precision, recall and F1-score.
-   Save the model and vectorizer.
-   Implement a simple prediction function for backend use.
-   Document model limitations and expected input/output.

Deliverables:

-   `src/model/train.py`
-   `src/model/classifier.py`
-   `src/model/evaluate.py`
-   Trained model artifact(s) in `models/`.
-   Evaluation summary.
-   Prediction interface agreed with Teammate 3.

Acceptance criteria:

-   The model can be trained from the agreed CSV.
-   The saved model can be reloaded.
-   New normalized text can be classified.
-   Output uses only canonical labels or an explicit uncertain state.
-   Evaluation results are genuine and reproducible.
-   No hard-coded predictions presented as model output.

### Teammate 3 --- Backend, integration and deployment

**Primary ownership:** `src/services/`, `src/analysis/`,
`src/reporting/`, application integration and deployment.

Responsibilities:

-   Establish the end-to-end investigation pipeline.
-   Define and implement the shared invocation interface.
-   Connect collector outputs to preprocessing outputs.
-   Connect normalized records to the trained model.
-   Integrate similarity, timeline and engagement analysis.
-   Coordinate the report generator.
-   Provide the frontend with a predictable result object or function
    interface.
-   Handle exceptions and partial-source failures.
-   Run end-to-end tests.
-   Deploy only after the local flow works.

Deliverables:

-   Investigation orchestration/service.
-   Analysis modules or working integrated equivalents.
-   Report generation.
-   Frontend integration interface.
-   Working local end-to-end run.
-   Deployed application if time permits.

Acceptance criteria:

-   A user query can pass through available collection, preprocessing,
    model and analysis stages.
-   Partial source failure does not crash the entire application when
    another source is usable.
-   Results are returned in the agreed schema.
-   Report findings link back to evidence.
-   Deployment does not take priority over a functioning local MVP.

### Teammate 4 --- Data preprocessing, fallbacks and frontend

**Primary ownership:** `src/preprocessing/` and `frontend/`

Responsibilities --- preprocessing:

-   Own the common normalized evidence schema with input from the team.
-   Read raw Google and Amazon records.
-   Clean text and remove irrelevant markup/noise.
-   Normalize dates and source fields.
-   Deduplicate records.
-   Preserve provenance and source URLs.
-   Handle missing fields, invalid dates, empty text and missing source
    files.
-   Produce normalized records.
-   Prepare the labelled training CSV for Teammate 2.
-   Document fallback behaviour and data-quality counts.

Responsibilities --- frontend:

-   Build the investigation interface independently of backend
    completion.
-   Use mock JSON matching the agreed result schema during development.
-   Provide topic/query input and source selection.
-   Display source availability and coverage.
-   Display classification labels and confidence/uncertainty.
-   Display timeline and source-specific engagement.
-   Display similar-content findings and evidence links.
-   Display report summary and download/view controls.
-   Replace mock data with the real backend interface during
    integration.

Deliverables:

-   `src/preprocessing/cleaner.py`
-   `src/preprocessing/normalizer.py`
-   Normalized data output and labelled training CSV.
-   Data-quality/fallback summary.
-   `frontend/` application files and components.
-   Mock response JSON for frontend development.
-   Frontend integration notes for Teammate 3.

Acceptance criteria:

-   Both source formats are converted to the same schema.
-   Missing optional values remain null.
-   Duplicate and invalid records are handled predictably.
-   Raw data is never overwritten.
-   Training CSV uses canonical labels.
-   Frontend runs independently using mock data.
-   Frontend can consume the agreed backend response without redesigning
    the entire UI.

------------------------------------------------------------------------

## 12. Ownership and file boundaries

  -----------------------------------------------------------------------
  Path                    Owner                   Notes
  ----------------------- ----------------------- -----------------------
  `src/collectors/`       Teammate 1              Source acquisition

  `data/raw/`             Teammate 1              Raw records; preserve
                                                  originals

  `src/preprocessing/`    Teammate 4              Cleaning and
                                                  normalization

  `data/processed/`       Teammate 4              Normalized records

  `data/training/`        Teammate 4 + Teammate 2 Labelled training data

  `src/model/`            Teammate 2              Training and inference

  `models/`               Teammate 2              Saved model artifacts

  `src/analysis/`         Teammate 3              Analysis pipeline

  `src/services/`         Teammate 3              Orchestration

  `src/reporting/`        Teammate 3              Report generation

  `frontend/`             Teammate 4              UI

  `app.py` / entry point  Teammate 3, coordinated Integration entry point
                          with Teammate 4         

  `tests/`                Each owner              Tests for owned modules
  -----------------------------------------------------------------------

Do not edit another teammate's owned files without coordination. Shared
schema and interface changes must be communicated immediately.

------------------------------------------------------------------------

## 13. GitHub workflow

Use one shared GitHub repository and independent feature branches.

### Branches

-   `main` --- stable integrated version only.
-   `feature/data-collection` --- Teammate 1.
-   `feature/ml-model` --- Teammate 2.
-   `feature/backend-integration` --- Teammate 3.
-   `feature/preprocessing` --- Teammate 4.
-   `feature/frontend` --- Teammate 4.

Teammate 4 may maintain two branches so frontend work can be merged
independently from preprocessing.

### Basic workflow

Clone the repository:

``` bash
git clone <REPOSITORY_URL>
cd GHOSTSIGNAL
```

Create a branch:

``` bash
git checkout -b feature/your-task
```

Commit focused work:

``` bash
git add .
git commit -m "Describe the completed change"
git push -u origin feature/your-task
```

Open a Pull Request into `main`. Review changed files and ensure the
application still runs before merging.

### Synchronize with main

``` bash
git checkout main
git pull origin main
git checkout feature/your-task
git merge main
```

Resolve conflicts carefully. Do not overwrite another teammate's work to
make a merge pass.

### Git rules

-   Never commit `.env`, API keys, tokens or passwords.
-   Do not commit virtual environments or `node_modules`.
-   Avoid committing large raw datasets or model artifacts unless the
    team explicitly decides they are required and safe to share.
-   Use small, meaningful commits.
-   Do not rename shared files or change function signatures without
    telling the team.
-   Merge working modules early rather than waiting until the last few
    minutes.

------------------------------------------------------------------------

## 14. Shared interfaces between teammates

The team should agree on these interfaces before parallel implementation
begins.

### Collector output

Each collector should return a list of raw records or save them in a
documented format. Teammate 1 must document the exact raw fields
returned.

### Preprocessing output

Teammate 4 should expose a function conceptually equivalent to:

``` python
def preprocess_records(records):
    """
    Input: list of raw source records.
    Output: normalized records and data-quality summary.
    """
    return normalized_records, quality_summary
```

The exact function may vary, but its arguments and return structure must
be agreed before implementation.

### Model output

Teammate 2 should expose a function conceptually equivalent to:

``` python
def predict_records(records):
    """
    Input: normalized records.
    Output: records enriched with predicted_label and confidence.
    """
    return predictions
```

The model should be loadable from saved artifacts. It should not retrain
on every investigation.

### Backend result

Teammate 3 should return a predictable investigation result object
containing:

``` json
{
  "query": "example investigation",
  "sources_requested": ["google", "amazon"],
  "sources_available": ["google"],
  "coverage": {},
  "records": [],
  "classification_summary": {},
  "timeline": [],
  "similarity_findings": [],
  "engagement_summary": {},
  "coordination_indicators": [],
  "limitations": [],
  "report": {}
}
```

This is a structural example, not a claim that these values already
exist. Empty arrays and objects are valid when no evidence is available.

### Frontend input

Teammate 4 should build the frontend against a mock object with the same
structure. The frontend must not depend on the backend's internal
implementation.

------------------------------------------------------------------------

## 15. Five-hour execution schedule

The schedule is a coordination guide. Parallel work should begin
immediately.

  ------------------------------------------------------------------------------------
  Time           Teammate 1      Teammate 2     Teammate 3       Teammate 4
  -------------- --------------- -------------- ---------------- ---------------------
  0:00--0:20     Verify source   Set up model   Set up project   Agree schema; create
                 access and raw  environment    entry point and  preprocessing and
                 schema          and labels     interfaces       frontend skeleton

  0:20--1:15     Collect real    Prepare        Build            Normalize data as it
                 Google/Amazon   training       orchestration    arrives; build UI
                 records         pipeline and   against          with mock data
                                 validation     placeholder      
                                                modules          

  1:15--2:15     Validate        Train and      Implement        Finish
                 collection and  evaluate on    timeline,        cleaning/fallbacks;
                 provenance      labelled data  similarity and   deliver training CSV;
                                                engagement       continue UI
                                                integration      

  2:15--3:15     Fix collection  Save model and Connect real     Finish main frontend
                 issues and      prediction     collector,       views
                 support         interface      preprocessing    
                 integration                    and model        
                                                modules          

  3:15--4:15     Support         Verify         Complete         Connect frontend to
                 collection      inference and  end-to-end       backend
                 fixes           evaluation     pipeline and     
                                                report           

  4:15--5:00     Support final   Support model  Run              Test display,
                 fixes           fixes          local/deployed   evidence links and
                                                demo and fix     report
                                                integration      
  ------------------------------------------------------------------------------------

### Time-critical rule

By approximately the midpoint, the team should have a working vertical
slice, even if limited to one source and a small number of records. Do
not wait for every planned feature before testing the entire pipeline.

------------------------------------------------------------------------

## 16. Definition of done

The MVP is ready to demonstrate when:

-   [ ] The user can enter an investigation query.
-   [ ] At least one real source can return records, subject to access
    availability.
-   [ ] The source and original URL are preserved where available.
-   [ ] Records are normalized into the shared schema.
-   [ ] The model has been trained on labelled examples.
-   [ ] Evaluation metrics are calculated and disclosed.
-   [ ] New records can be classified by the saved model.
-   [ ] The UI displays records and their classifications.
-   [ ] Available dates can be shown in a timeline.
-   [ ] Similar content can be surfaced with evidence links.
-   [ ] Available engagement information is shown without mixing
    incompatible metrics.
-   [ ] Source failures and missing data are disclosed.
-   [ ] A report or structured investigation summary can be generated.
-   [ ] The demo does not present mock data as live evidence.
-   [ ] The application runs locally end to end.

Deployment is desirable, but a reliable local demonstration is more
important than a broken deployed application.

------------------------------------------------------------------------

## 17. Instructions for AI assistants used by team members

Every teammate may use a different AI model or coding assistant. Each
assistant must be given this document as project context.

### Mandatory instructions for AI assistants

1.  Read this project brief before suggesting architecture or writing
    code.
2.  Preserve the GHOSTSIGNAL product definition and report-only purpose.
3.  Do not expand the five-hour MVP into an enterprise platform.
4.  Do not introduce extra frameworks, databases, services or model
    families unless necessary and approved.
5.  Respect the teammate ownership table.
6.  Respect the shared data schema and canonical classification labels.
7.  Do not rename files or change interfaces without warning the team.
8.  Do not fabricate source records, publication dates, engagement
    metrics, model scores or evaluation results.
9.  Clearly distinguish live data, imported data, fallback data and mock
    data.
10. Do not claim full-platform access when only partial search/API data
    is available.
11. Use authorized APIs, approved integrations, permitted exports or
    public feeds. Do not bypass access controls or scrape in violation
    of source terms.
12. Keep credentials in environment variables; never hard-code secrets
    or commit `.env`.
13. Prefer the smallest working implementation.
14. Write code that integrates with the existing project rather than
    replacing the project structure.
15. Include a short explanation of files changed, dependencies added,
    commands to run and assumptions made.
16. If a requirement is unclear, ask the teammate before making a major
    architectural decision.
17. Do not implement platform interventions, automated moderation or
    account actions.
18. Treat classification as a content-category prediction, not a
    determination of truth, authenticity, sponsorship or intent.
19. Treat coordination indicators as hypotheses requiring evidence and
    alternative explanations.
20. Prioritize a working end-to-end demo over speculative features.

### Prompt teammates can give their AI assistant

> You are helping build GHOSTSIGNAL, a five-hour hackathon MVP. First
> read `GHOSTSIGNAL_PROJECT_BRIEF.md` in full. Treat it as the source of
> truth for product scope, architecture, data schema, classification
> labels, team ownership and interfaces. I am Teammate \[number\],
> responsible for \[assigned workstream\]. Only work within my assigned
> ownership unless I explicitly ask otherwise. Preserve existing
> interfaces and coordinate any shared-schema changes. Build the
> simplest working implementation that fits the current project. Do not
> invent data, metrics or results. Explain the exact files you will
> create or modify before making broad changes. Keep the application
> compatible with the other teammates' work.

------------------------------------------------------------------------

## 18. Future roadmap --- not part of the five-hour build

Only after the MVP works should the team consider:

-   Reddit and X connectors.
-   YouTube news coverage and available transcripts.
-   More digital news publishers and television-channel sources.
-   Multilingual narrative analysis.
-   Better semantic embeddings and clustering.
-   Temporal burst detection.
-   Network and community analysis.
-   More robust coordination scoring.
-   Better report export and evidence visualization.
-   Persistent database and investigation history.
-   User accounts, permissions and production deployment.

These are future extensions, not hackathon acceptance requirements.

------------------------------------------------------------------------

## 19. Final team principle

**Build the smallest honest version of GHOSTSIGNAL that proves the
central idea.**

A small model trained on real, labelled examples, operating on real
collected evidence and producing transparent classifications is more
valuable than a complex architecture that cannot complete an
investigation.

The team should optimize for:

1.  Real evidence.
2.  Reproducible preprocessing.
3.  A working trained classifier.
4.  Traceable analytical results.
5.  Clear presentation.
6.  Honest limitations.

**GHOSTSIGNAL investigates information. It does not intervene in the
platforms where that information appears.**
