/* Shared interactive visuals for index.html and play.html.
   UMLSVIS.pipeline(el)      the five-step walkthrough from a medical question to the answer
   UMLSVIS.umls(el, view)    what UMLS is and what its tables contain
   UMLSVIS.finding(id)       the chart that goes with each finding (returns HTML) */
(function () {
'use strict';
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');

/* ------------------------------------------------------------------ */
/* 1. The walkthrough: question, entity linking, subgraph, injection, answer */
const STEPS = [
  { key: 'q', label: 'Question', stage: 'q', title: 'An exam question about a patient',
    note: 'Medical QA benchmarks such as MedQA ask questions like this one, taken from medical licensing exams. The model must pick one option.' },
  { key: 'nel', label: 'Entity linking', stage: 'nel', title: 'Step 1. Find the medical concepts',
    note: 'An entity linker spots medical terms in the question and in the options and maps each one to a UMLS concept ID. Three of them are shown here.', breaks: [3, 4, 5] },
  { key: 'ksr', label: 'Subgraph retrieval', stage: 'ksr', title: 'Step 2. Collect the facts that connect them',
    note: 'Starting from the linked concepts, the system walks UMLS relations and keeps the paths that connect question and options: a small knowledge graph for this question.', breaks: [5, 6, 7] },
  { key: 'ki', label: 'Knowledge injection', stage: 'ki', title: 'Step 3. Give the facts to the model',
    note: 'The facts reach the model in one of three ways: written into the prompt (shown here), encoded by a graph neural network, or used as training data.', breaks: [8, 9] },
  { key: 'a', label: 'Answer', stage: 'a', title: 'The model answers',
    note: 'With the evidence, the model picks B) Celecoxib. Every arrow in this pipeline hides a construction choice, and the papers rarely report them.' },
];
function pipelineHTML() {
  return `<div class="uvp" data-step="0">
  <div class="uvp-tabs" role="tablist">${STEPS.map((s, i) => `<button role="tab" class="uvp-tab uv-${s.stage}" data-i="${i}" aria-selected="${i === 0}"><span class="n">${i ? i : ''}</span>${s.label}</button>`).join('')}</div>
  <div class="uvp-body">
    <div class="uvp-stage">
      <div class="uvp-q">
        <p class="uvp-text"><span class="sp sp-ra">A patient with <mark class="m-ra">rheumatoid arthritis (RA)</mark></span> requires selective inhibition of <mark class="m-cox">COX-2</mark>. Which drug is most appropriate?</p>
        <ol class="uvp-opts" type="A"><li>Acetaminophen</li><li class="o-b"><mark class="m-cel">Celecoxib</mark></li><li>Ketorolac</li><li>Aspirin</li></ol>
        <div class="uvp-chips"><span class="chip c-ra">rheumatoid arthritis <b>C0003873</b></span><span class="chip c-cox">COX-2 <b>C0387583</b></span><span class="chip c-cel">Celecoxib <b>C0538927</b></span></div>
      </div>
      <svg class="uvp-graph" viewBox="0 0 520 210" role="img" aria-label="Subgraph: Celecoxib inhibits COX-2, Celecoxib may treat rheumatoid arthritis">
        <defs><marker id="uvArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>
        <g class="e e1"><path d="M300 104 L 170 52" marker-end="url(#uvArrow)"/><text x="230" y="64">inhibits</text></g>
        <g class="e e2"><path d="M300 116 L 170 166" marker-end="url(#uvArrow)"/><text x="226" y="160">may treat</text></g>
        <g class="nd nq"><rect x="40" y="30" width="130" height="40" rx="4"/><text x="105" y="55">COX-2</text></g>
        <g class="nd nq"><rect x="20" y="146" width="150" height="40" rx="4"/><text x="95" y="171">Rheumatoid arthritis</text></g>
        <g class="nd na"><rect x="300" y="90" width="120" height="40" rx="4"/><text x="360" y="115">Celecoxib</text></g>
        <g class="nd nx"><rect x="440" y="20" width="76" height="30" rx="4"/><text x="478" y="40">Acetamin.</text></g>
        <g class="nd nx"><rect x="440" y="96" width="76" height="30" rx="4"/><text x="478" y="116">Ketorolac</text></g>
        <g class="nd nx"><rect x="440" y="170" width="76" height="30" rx="4"/><text x="478" y="190">Aspirin</text></g>
        <text class="lg lg-q" x="20" y="16">question concepts</text><text class="lg lg-a" x="300" y="80">answer option B</text><text class="lg lg-x" x="440" y="12">other options</text>
      </svg>
      <div class="uvp-prompt" aria-label="Prompt given to the model">
        <p class="pl"><span class="sys">You are a medical expert tasked with answering multiple-choice questions. Use the question stem and, when present, the provided UMLS subgraph as evidence. Select the single best answer.</span></p>
        <p class="pl ev"><b>UMLS evidence:</b> Celecoxib inhibits COX-2. Celecoxib may treat rheumatoid arthritis.</p>
        <p class="pl"><b>Question:</b> A patient with rheumatoid arthritis (RA) requires selective inhibition of COX-2. Which drug is most appropriate?</p>
        <p class="pl"><b>Options:</b> A) Acetaminophen B) Celecoxib C) Ketorolac D) Aspirin</p>
      </div>
      <div class="uvp-answer"><div class="llm"><span>Language model</span></div><div class="arrow"></div><div class="out"><span class="lab">Answer</span><strong>B) Celecoxib</strong></div></div>
    </div>
    <div class="uvp-side">
      <h4 class="uvp-title"></h4>
      <p class="uvp-note"></p>
      <p class="uvp-breaks"></p>
      <div class="uvp-ctrl"><button class="uvp-prev" aria-label="Previous step">Back</button><button class="uvp-next">Next step</button></div>
    </div>
  </div>
</div>`;
}
function pipeline(el, opts = {}) {
  el.innerHTML = pipelineHTML();
  const root = el.querySelector('.uvp'), tabs = [...root.querySelectorAll('.uvp-tab')];
  let step = 0, timer = null;
  function set(i) {
    step = Math.max(0, Math.min(STEPS.length - 1, i)); root.dataset.step = step;
    tabs.forEach((t, j) => { t.setAttribute('aria-selected', String(j === step)); t.classList.toggle('done', j < step); });
    const s = STEPS[step];
    root.querySelector('.uvp-title').textContent = s.title;
    root.querySelector('.uvp-note').textContent = s.note;
    const br = root.querySelector('.uvp-breaks');
    br.innerHTML = s.breaks ? `Where it can break: ${s.breaks.map(n => opts.link ? `<a href="${opts.link(n)}">finding ${n}</a>` : `finding ${n}`).join(', ')}.` : '';
    root.querySelector('.uvp-prev').disabled = step === 0;
    root.querySelector('.uvp-next').textContent = step === STEPS.length - 1 ? 'Start again' : 'Next step';
    if (opts.onStep) opts.onStep(step);
  }
  tabs.forEach(t => t.addEventListener('click', () => { stop(); set(+t.dataset.i); }));
  root.querySelector('.uvp-prev').addEventListener('click', () => { stop(); set(step - 1); });
  root.querySelector('.uvp-next').addEventListener('click', () => { stop(); set(step === STEPS.length - 1 ? 0 : step + 1); });
  function stop() { if (timer) { clearInterval(timer); timer = null; } }
  if (opts.autoplay) timer = setInterval(() => set(step === STEPS.length - 1 ? 0 : step + 1), opts.autoplay);
  if (opts.hideControls) root.classList.add('noctrl');
  set(opts.start || 0);
  return { set, get step() { return step; }, stop, count: STEPS.length };
}

/* ------------------------------------------------------------------ */
/* 2. What UMLS is and what its tables contain */
const TABLES = [
  { id: 'MRCONSO', desc: 'names and synonyms from each vocabulary', what: 'Names', use: 100, cols: ['Concept', 'Source vocabulary', 'Name'], rows: [['C0014848', 'SNOMEDCT_US', 'Achalasia'], ['C0014848', 'ICD10CM', 'Achalasia of cardia'], ['C0014848', 'MSH', 'Cardiospasm'], ['C0014848', 'MSHCHI', '\u98df\u7ba1\u5931\u5f1b\u7f13\u75c7']] },
  { id: 'MRREL', desc: 'relations between concepts', what: 'Relations', use: 84, cols: ['Concept', 'Relation', 'Related concept'], rows: [['C0014848', 'is_a', 'Esophageal motility disorders'], ['C0014848', 'has_finding_site', 'Esophagogastric junction'], ['C0014848', 'has_contro_drug', 'Benztropine']] },
  { id: 'MRSTY', desc: 'the category, such as Disease or Syndrome', what: 'Semantic types', use: 20, cols: ['Concept', 'Type ID', 'Semantic type'], rows: [['C0014848', 'T047', 'Disease or Syndrome']] },
  { id: 'MRDEF', desc: 'text definitions', what: 'Definitions', use: 8, cols: ['Concept', 'Source', 'Definition'], rows: [['C0014848', 'MSH', 'A motility disorder of the esophagus in which the lower esophageal sphincter fails to relax during swallowing ...']] },
  { id: 'MRHIER', desc: 'parent and child paths', what: 'Hierarchies', use: 0, cols: ['Concept', 'Path from the root'], rows: [['C0014848', 'Digestive system diseases > Gastrointestinal diseases > Esophageal diseases > Esophageal motility disorders > Esophageal achalasia']] },
  { id: 'MRSAT', desc: 'codes and other attributes', what: 'Attributes', use: 0, cols: ['Concept', 'Attribute', 'Value'], rows: [['C0014848', 'ICD-10-CM code', 'K22.0'], ['C0014848', 'MeSH ID', 'D004941'], ['C0014848', 'SNOMED CT ID', '45564002']] },
];
function tableHTML(t) {
  return `<table class="uvt"><caption><code>${t.id}</code> ${t.what}</caption><thead><tr>${t.cols.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>${t.rows.map(r => `<tr>${r.map((c, j) => `<td${j === 0 ? ' class="cui"' : ''}>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
const UMLS_DB = `<div class="uvu-db">
  <div class="uvu-src main"><h5>Metathesaurus</h5><p>23 relational tables. Six describe concepts (names, relations, semantic types, definitions, hierarchies, attributes); 17 serve infrastructure and versioning.</p></div>
  <div class="uvu-src"><h5>Semantic Network</h5><p>About 130 semantic types, such as Disease or Syndrome, linked by 54 semantic relations.</p></div>
  <div class="uvu-src"><h5>SPECIALIST Lexicon</h5><p>Over one million English entries with spelling variants, used by tools that find medical terms in text.</p></div>
</div>
<div class="uvu-stats"><div><b>190</b><span>vocabularies joined, from SNOMED CT and ICD-10-CM to MeSH and RxNorm</span></div><div><b>3.49M</b><span>concepts, each with an ID (CUI)</span></div><div><b>17.39M</b><span>names for those concepts</span></div><div><b>29</b><span>languages</span></div></div>
<p class="uvu-foot">Release 2025AB. NLM ships two releases a year as pipe-separated files meant to be loaded into a relational database.</p>`;
const UMLS_GRAPH = `<svg class="uvu-graph" viewBox="0 0 540 190" role="img" aria-label="Rows of the relation table drawn as a graph around esophageal achalasia">
  <defs><marker id="uvArrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>
  <g class="e"><path d="M205 80 L 140 45" marker-end="url(#uvArrow2)"/><text x="120" y="78">is a</text></g>
  <g class="e"><path d="M335 80 L 400 45" marker-end="url(#uvArrow2)"/><text x="376" y="80">finding site</text></g>
  <g class="e"><path d="M270 118 L 270 148" marker-end="url(#uvArrow2)"/><text x="280" y="138">contraindicated drug</text></g>
  <g class="nd nc"><rect x="190" y="78" width="160" height="40" rx="4"/><text x="270" y="103">Esophageal achalasia</text></g>
  <g class="nd"><rect x="10" y="12" width="200" height="34" rx="4"/><text x="110" y="34">Esophageal motility disorders</text></g>
  <g class="nd"><rect x="330" y="12" width="200" height="34" rx="4"/><text x="430" y="34">Esophagogastric junction</text></g>
  <g class="nd"><rect x="200" y="150" width="140" height="34" rx="4"/><text x="270" y="172">Benztropine</text></g>
</svg>`;
function umls(el, view = 'tabs') {
  if (view === 'db') { el.innerHTML = `<div class="uvu">${UMLS_DB}</div>`; return; }
  if (view === 'rows') { el.innerHTML = `<div class="uvu uvu-rows">${TABLES.slice(0, 4).map(tableHTML).join('')}<p class="uvu-foot">Two more concept tables hold hierarchies (MRHIER) and attributes (MRSAT); the other 17 tables serve infrastructure and versioning.</p></div>`; return; }
  if (view === 'graph') { el.innerHTML = `<div class="uvu">${UMLS_GRAPH}<p class="uvu-foot">Each row of the relation table becomes an edge (head, relation, tail). 88% of the reviewed papers use UMLS this way.</p></div>`; return; }
  // interactive explorer for the website
  el.innerHTML = `<div class="uvu uvx">${UMLS_DB}
    <div class="uvx-pick" role="tablist" aria-label="UMLS tables">${TABLES.map((t, i) => `<button role="tab" aria-selected="${i === 0}" data-i="${i}"><code>${t.id}</code><span>${t.what}</span></button>`).join('')}<button role="tab" aria-selected="false" data-i="g"><code>As a graph</code><span>How papers use it</span></button></div>
    <div class="uvx-view"></div></div>`;
  const view_ = el.querySelector('.uvx-view'), btns = [...el.querySelectorAll('.uvx-pick button')];
  function show(i) {
    btns.forEach(b => b.setAttribute('aria-selected', String(b.dataset.i === String(i))));
    if (i === 'g') { view_.innerHTML = UMLS_GRAPH + `<p class="uvu-foot">Each row of the relation table becomes an edge (head, relation, tail). 88% of the reviewed papers turn UMLS into such a graph.</p>`; return; }
    const t = TABLES[i];
    view_.innerHTML = tableHTML(t) + `<div class="uvx-use"><span>Used by the reviewed papers</span><div class="trk"><i style="width:${t.use}%"></i></div><b>${t.use}%</b></div>`;
  }
  btns.forEach(b => b.addEventListener('click', () => show(b.dataset.i === 'g' ? 'g' : +b.dataset.i)));
  show(0);
}

/* ------------------------------------------------------------------ */
/* 3. One chart per finding */
function bars(rows, { max, unit = '', log = false, fmt = (v) => v } = {}) {
  const m = max || Math.max(...rows.map(r => r[1]));
  const w = (v) => log ? Math.max(2, Math.log10(Math.max(1, v)) / Math.log10(m) * 100) : (v / m * 100);
  return `<div class="uvb">${rows.map(([label, v, cls = '', note = '']) => `<div class="uvb-row ${cls}"><span class="uvb-l">${label}</span><div class="uvb-t"><i style="--w:${w(v).toFixed(1)}%"></i></div><span class="uvb-v">${fmt(v)}${unit}${note ? ` <em>${note}</em>` : ''}</span></div>`).join('')}</div>`;
}
const fmtInt = (v) => v.toLocaleString('en-US');
const VIS = {
  release: () => `<div class="uvv"><p class="uvv-h">The 25 reviewed papers, by whether they name their UMLS release</p><div class="uvdocs">${Array.from({ length: 25 }, (_, i) => i < 4 ? `<span class="doc on"><b>${['2020AA', '2021AA', '2022AB', '2023AA'][i]}</b></span>` : '<span class="doc">?</span>').join('')}</div><p class="uvv-k"><span class="key on"></span> names its release <span class="key"></span> does not say</p></div>`,
  tables: () => `<div class="uvv"><p class="uvv-h">Share of the 25 papers that use each concept table</p>${bars([['Names <code>MRCONSO</code>', 100], ['Relations <code>MRREL</code>', 84], ['Semantic types <code>MRSTY</code>', 20], ['Definitions <code>MRDEF</code>', 8], ['Hierarchies <code>MRHIER</code>', 0], ['Attributes <code>MRSAT</code>', 0]], { max: 100, unit: '%' })}</div>`,
  linkers: () => `<div class="uvv two"><div><p class="uvv-h">Concepts linked per MedQA question (same UMLS release)</p>${bars([['MetaMap', 116.5, 'k1'], ['scispaCy', 57.7, 'k2']], { max: 120 })}</div><div><p class="uvv-h">Share of links judged correct, 40 answer options checked by hand</p>${bars([['MetaMap', 35.6, 'k1', '40 of 40 options found'], ['scispaCy', 46.7, 'k2', '36 of 40 options found']], { max: 100, unit: '%' })}</div></div>`,
  'release-swap': () => `<div class="uvv two"><div><p class="uvv-h">Concepts linked per MedQA question, scispaCy, by UMLS release</p>${bars([['2020AA', 57.67], ['2021AA', 71.76], ['2022AB', 67.18], ['2023AA', 67.53]], { max: 80, fmt: (v) => v.toFixed(1) })}</div><div><p class="uvv-h">Same, 2023AA, keeping only some vocabularies</p>${bars([['All vocabularies', 67.53], ['Level-0 only', 62.63], ['DrugBank and DDB', 0.99, 'k3']], { max: 80, fmt: (v) => v.toFixed(1) })}</div></div>`,
  'answer-side': () => `<div class="uvv"><p class="uvv-h">MedQA test split: share of questions and of answer options with no linked concept</p><div class="uvpair"><span></span><span class="h">questions</span><span class="h">answer options</span></div>${[['QA-GNN', 3.3, 58.8], ['GreaseLM', 3.3, 59.0], ['DRAGON', 0.1, 24.5], ['MetaMap (ours)', 0, 0.9]].map(([n, q, a]) => `<div class="uvpair"><span class="n">${n}</span>${bars([['', q, 'k2']], { max: 100, unit: '%' })}${bars([['', a, 'k3']], { max: 100, unit: '%' })}</div>`).join('')}</div>`,
  retrieval: () => `<div class="uvv"><p class="uvv-h">Concepts collected around a PubMedQA question (log scale)</p>${bars([['Full UMLS, 2 hops', 3755, 'k1'], ['Full UMLS, 5 hops', 1336309, 'k3'], ['DRAGON graph, 2 hops', 193, 'k2'], ['DRAGON graph, 5 hops', 46719, 'k2'], ['Node budget of QA-GNN', 200, 'k4']], { max: 1336309, log: true, fmt: fmtInt })}</div>`,
  fallback: () => `<div class="uvv"><p class="uvv-h">What the graph contains when linking finds nothing</p><svg class="uvph" viewBox="0 0 520 120" role="img" aria-label="An unlinked question is replaced by Serine and an unlinked answer option by Ethanol"><g class="nd nq"><rect x="10" y="10" width="160" height="38" rx="4"/><text x="90" y="34">question: nothing linked</text></g><g class="nd na"><rect x="10" y="72" width="160" height="38" rx="4"/><text x="90" y="96">option: nothing linked</text></g><g class="e"><path d="M175 29 L 300 29" marker-end="url(#uvArrow3)"/></g><g class="e"><path d="M175 91 L 300 91" marker-end="url(#uvArrow3)"/></g><g class="nd nph"><rect x="305" y="10" width="200" height="38" rx="4"/><text x="405" y="34">Serine, an amino acid</text></g><g class="nd nph"><rect x="305" y="72" width="200" height="38" rx="4"/><text x="405" y="96">Ethanol</text></g><defs><marker id="uvArrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs></svg></div>`,
  retraining: () => `<div class="uvv"><p class="uvv-h">QA-GNN test accuracy on MedQA under seven UMLS constructions (mean of three runs; the axis starts at 36%)</p>${bars([['2020AA (as published)', 38.1], ['2021AA', 37.6], ['2022AB', 38.7], ['2023AA', 38.0], ['Level-0 only', 37.5], ['DrugBank and DDB', 38.5], ['Empty graph', 37.4, 'k3']].map(([l, v, c]) => [l, v - 36, c]), { max: 4, fmt: (v) => (v + 36).toFixed(1), unit: '%' })}<p class="uvv-k">All seven runs fall within 1.3 points, a spread larger than the gains reported in this line of work (+0.8 and +0.5).</p></div>`,
  readers: () => `<div class="uvv"><p class="uvv-h">Three ways to write the same UMLS evidence into the prompt</p><div class="uvfmt"><div><b>Plain relations</b><code>Celecoxib inhibits COX-2. Celecoxib may treat RA. ...</code><span>all relations around the linked concepts; wins for 9 of 10 models on PubMedQA</span></div><div><b>Question-to-answer paths</b><code>RA, may treat, Celecoxib</code><span>only the relations on paths between question and options</span></div><div><b>Enriched paths</b><code>the same paths + each concept's definition and semantic type</code><span>helps most for most models on MedQA and MedMCQA</span></div></div></div>`,
};
function finding(anchor) { return VIS[anchor] ? VIS[anchor]() : ''; }
function animate(scope) { requestAnimationFrame(() => requestAnimationFrame(() => scope.querySelectorAll('.uvb-t i').forEach(i => i.classList.add('go')))); }

window.UMLSVIS = { pipeline, umls, finding, animate, TABLES, STEPS };
})();
