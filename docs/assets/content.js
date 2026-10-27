/* Shared content for the classic page (index.html) and the game (play.html).
   Every number comes from the paper. Edit the text here and both pages update. */
(function () {
'use strict';

const PAPER = {
  title: 'Are We Done with UMLS Injection into Language Models for Medical Question Answering?',
  shortTitle: 'Are We Done with UMLS Injection?',
  pdf: 'http://arxiv.org/abs/2610.33801',
  code: 'https://github.com/disi-unibo-nlp/are-we-done-with-umls',
  group: 'https://disi-unibo-nlp.github.io/',
  authors: [
    { name: 'Yiran Zeng', letter: 'U' },
    { name: 'Giacomo Frisoni', letter: 'M' },
    { name: 'Lorenzo Molfetta', letter: 'L' },
    { name: 'Gianluca Moro', letter: 'S' },
  ],
  affiliation: 'Department of Computer Science and Engineering, University of Bologna, Italy',
  bibtex: `@misc{zeng2026umls,
  title  = {Are We Done with {UMLS} Injection into Language Models
            for Medical Question Answering?},
  author = {Yiran Zeng and Giacomo Frisoni and
            Lorenzo Molfetta and Gianluca Moro},
  year   = {2026},
  note   = {Manuscript}
}`,
};

/* Example question used in the story and on the classic page (Fig. 4 of the paper). */
const EXAMPLE = {
  question: 'A patient with rheumatoid arthritis (RA) requires selective inhibition of COX-2. Which drug is most appropriate?',
  options: ['Acetaminophen', 'Celecoxib', 'Ketorolac', 'Aspirin'],
  answer: 1,
  links: [['RA', 'C0003873'], ['COX-2', 'C0387583'], ['Celecoxib', 'C0538927']],
  facts: [['Celecoxib', 'inhibits', 'COX-2'], ['Celecoxib', 'may treat', 'RA']],
};

/* Example concept (Fig. 3 of the paper). */
const CONCEPT = {
  name: 'Esophageal achalasia', cui: 'C0014848',
  names: ['Achalasia', 'Achalasia of cardia', 'Cardiospasm'],
  type: 'Disease or Syndrome',
  definition: 'A motility disorder of the esophagus in which the lower esophageal sphincter fails to relax during swallowing.',
  relations: [['is a', 'Esophageal motility disorders'], ['finding site', 'Esophagogastric junction'], ['contraindicated drug', 'Benztropine']],
};

/* The opening story of the game. Each step is one line of dialogue; some show a card. */
const STORY = [
  { scene: 'lab', who: 'Narrator', text: 'Bologna, late at night. Four researchers in a lab for medical natural language processing are teaching AI language models to answer medical questions.' },
  { scene: 'lab', who: 'Narrator', card: 'exam', text: 'Their test bench: questions written like medical licensing exams, with a short patient case and four or five options. The model must pick the right one.' },
  { scene: 'lab', who: 'Narrator', text: 'A language model answers from what it absorbed during training. That knowledge is hidden in its parameters: hard to inspect and costly to update. In medicine, a wrong answer can hurt someone.' },
  { scene: 'lab', who: 'Narrator', card: 'umls-db', text: 'So many teams connect the model to UMLS, the Unified Medical Language System of the US National Library of Medicine. UMLS is not a chatbot or a website: it is a large database of medical terminology.' },
  { scene: 'lab', who: 'Narrator', card: 'umls-rows', text: 'Its core, the Metathesaurus, is a set of relational tables. Every medical concept has an ID, and its names, relations, type and definition are rows spread over different tables.' },
  { scene: 'lab', who: 'Narrator', card: 'umls-graph', text: 'Most systems turn the relation table into a knowledge graph: every row becomes an arrow between two concepts.' },
  { scene: 'lab', who: 'Narrator', card: 'pipeline', step: 0, text: 'Here is how a system uses that graph to answer an exam question. It starts from the question and its four options.' },
  { scene: 'lab', who: 'Narrator', card: 'pipeline', step: 1, text: 'Step 1, entity linking: a tool finds the medical terms in the text and maps each one to its UMLS concept ID.' },
  { scene: 'lab', who: 'Narrator', card: 'pipeline', step: 2, text: 'Step 2, subgraph retrieval: starting from those concepts, it walks UMLS relations and keeps the facts that connect question and options.' },
  { scene: 'lab', who: 'Narrator', card: 'pipeline', step: 3, text: 'Step 3, knowledge injection: the facts are handed to the model, here written into its prompt as evidence.' },
  { scene: 'lab', who: 'Narrator', card: 'pipeline', step: 4, text: 'The model reads the question, the options and the evidence, and answers B, Celecoxib.' },
  { scene: 'lab', who: 'Narrator', text: 'Dozens of papers report that UMLS helps this way. The gains are modest, though: in the reference systems of this line of work, less than one accuracy point.' },
  { scene: 'walk', who: 'The team', text: 'Do those gains really come from UMLS? Let us review every study from 2020 to 2025, 25 in total, and rebuild their choices one at a time.' },
  { scene: 'archive', who: 'The team', rattle: true, text: 'Wait. Did the closet with the UMLS label just move?' },
  { scene: 'archive', who: 'Narrator', text: 'Every field has skeletons in its closet. Open this one, catch what escapes, and help the community see what is really inside.' },
];

/* The nine findings. `quiz` is used by both pages; `game` tells the game which mini-game to run. */
const SKELS = [
  { id: 'release', name: 'The Phantom Release', room: 'RELEASE SHELF', anchor: 'release', ref: 'Section 4 and Figure 2',
    brief: 'UMLS changes over time. The National Library of Medicine publishes a new release twice a year, in May (for example 2025AA) and in November (2025AB), adding, merging and retiring vocabularies and concepts. A paper that uses UMLS should say which release it used, just as you would cite the edition of a textbook.',
    quiz: { q: 'Of the 25 papers we reviewed, how many say which UMLS release they used?', options: ['21 of 25', '13 of 25', '4 of 25'], answer: 2 },
    game: { type: 'quiz' },
    stamp: '84%', stampLabel: 'of the papers never say which release they used',
    title: 'Most papers never say which UMLS they used',
    text: 'Only 4 of the 25 papers name their release, and those four use four different ones: 2020AA, 2021AA, 2022AB and 2023AA. These releases range from about 182 to 220 vocabularies and from 3.4 to 4.7 million concepts.',
    why: 'Two systems reporting different scores may have been given different knowledge, and neither paper records enough to tell which.' },

  { id: 'tables', name: 'The Two-Table Skeleton', room: 'DATABASE ROOM', anchor: 'tables', ref: 'Section 4 and Figure 3',
    brief: 'The Metathesaurus has 23 tables. Six of them describe concepts, as in the achalasia example of the story: names, relations, semantic types, definitions, hierarchies and attributes. Together they tell a model what a concept is and how it fits into the rest of medicine.',
    quiz: { q: 'UMLS ships 23 tables. How many do almost all reviewed papers use?', options: ['2', '6', 'All 23'], answer: 0 },
    game: { type: 'tables', prompt: 'Which concept tables do you think most papers give to the model? Select the ones you expect, then check.' },
    stamp: '2 of 23', stampLabel: 'tables used by almost every paper',
    title: 'UMLS shrinks to names and relations',
    text: 'Names are used in all 25 papers and relations in 84%. Semantic types appear in 20% of the papers, definitions in 8%, hierarchies and attributes in none. Most papers also keep a single source vocabulary, or as few as 4 kinds of relation.',
    why: 'Facts that live in different vocabularies, such as a drug\u2019s dosage in RxNorm and its drug class in ATC, can then never be chained in one line of reasoning.' },

  { id: 'linker', name: 'The Linker Lottery', room: 'LINKING BENCH', anchor: 'linkers', ref: 'Section 5.1, Figure 5 and Table 1',
    brief: 'Step 1 is entity linking: a tool finds the medical terms in a question and maps each one to a UMLS concept. Anything it misses is invisible to every later step. Most reviewed systems use a tool called scispaCy; MetaMap is the main alternative.',
    quiz: { q: 'On MedQA questions, MetaMap links about 116 concepts per question. How many does scispaCy link, with the same UMLS release?', options: ['About 29', 'About 58', 'About 116'], answer: 1 },
    game: { type: 'link', prompt: 'Be the entity linker. Select the words of this question that should map to a medical concept. Find the two key ones.' },
    stamp: '116 vs 58', stampLabel: 'concepts per MedQA question, MetaMap against scispaCy',
    title: 'The linker decides what UMLS can offer',
    text: 'On the same MedQA questions and the same UMLS release, MetaMap links 116.5 concepts per question on average and scispaCy 57.7: almost one concept per word. In our manual check of 219 links on 40 answer options, 35.6% of MetaMap\u2019s links were correct against 46.7% for scispaCy; the rest were generic or wrong. Yet MetaMap found a correct concept in all 40 options, scispaCy in 36.',
    why: 'The choice of linker alone can halve or double the UMLS evidence a model receives, and none of the reviewed papers reports how much its linker covers.' },

  { id: 'swap', name: 'The Shapeshifter', room: 'RELEASE SWAP', anchor: 'release-swap', ref: 'Section 5.1 and Table 2',
    brief: 'Now keep the linker fixed and change only the UMLS release it looks up, like replacing a dictionary with a newer edition. The questions stay exactly the same.',
    quiz: { q: 'For how many MedQA questions does the set of linked concepts change when only the release changes?', options: ['About 10%', 'About half', 'More than 99%'], answer: 2 },
    game: { type: 'quiz' },
    stamp: '99.4%', stampLabel: 'of the MedQA questions get different concepts',
    title: 'Same linker, different UMLS, different evidence',
    text: 'Moving scispaCy from UMLS 2020AA to 2021AA raises the concepts per MedQA question from 57.7 to 71.8, and across releases the linked set changes for at least 99.4% of the questions. Restricting UMLS to just two vocabularies, DrugBank and the Diseases Database, as the reference systems QA-GNN and GreaseLM do, leaves about one concept per question in our runs.',
    why: 'More than 98% of the evidence is gone before the model reads anything, and the papers still describe this as using UMLS.' },

  { id: 'answers', name: 'The Missing Answer', room: 'EXAM HALL', anchor: 'answer-side', ref: 'Section 5.1 and Table 10',
    brief: 'Graph-based systems look for chains in UMLS that connect the concepts of the question to those of each answer option, such as Celecoxib, may treat, rheumatoid arthritis. Both ends must be linked. But answer options are short fragments, like \u201cAtrophy\u201d or \u201cProtect against viral infections\u201d, that give the linker little context.',
    quiz: { q: 'In the released MedQA test data of QA-GNN, how many answer options link no concept at all?', options: ['About 3%', 'About 25%', 'About 59%'], answer: 2 },
    game: { type: 'quiz' },
    stamp: '58.8%', stampLabel: 'of the answer options link no concept',
    title: 'The answers never enter UMLS',
    text: 'Only 3.3% of the questions link nothing, but 58.8% of the answer options do. GreaseLM has the same gap, and DRAGON, with a broader vocabulary, still misses 24.5%. In our runs with MetaMap in place of scispaCy, about 1% of the options stay unlinked.',
    why: 'For most options no question-to-answer chain can exist, so the core idea of these systems silently fails on the majority of the benchmark.' },

  { id: 'hydra', name: 'The Hydra Subgraph', room: 'GRAPH ROOM', anchor: 'retrieval', ref: 'Section 5.2, Figure 6 and Table 3',
    brief: 'Step 2 collects evidence by walking along UMLS relations, starting from the linked concepts. Each step is a hop: one hop reaches the direct neighbors, two hops reach their neighbors, and so on. UMLS is dense, so every hop multiplies what you collect.',
    quiz: { q: 'Starting from the concepts of a PubMedQA question, how many concepts do you collect on average after five hops through the full UMLS?', options: ['About 4 thousand', 'About 50 thousand', 'More than 1.3 million'], answer: 2 },
    game: { type: 'quiz' },
    stamp: '1.3M', stampLabel: 'concepts collected after five hops',
    title: 'The evidence explodes, then gets cut',
    text: 'The collected evidence grows from 3,755 concepts at two hops to 1,336,309 at five. So most systems stop at two hops and then trim the graph to 200 concepts. In QA-GNN, GreaseLM and DRAGON the trim applies to 70 to 73% of the examples, and only about 1.5 concepts from each answer option survive it.',
    why: 'The answer side, already the weakest after linking, is cut a second time before the model sees it.' },

  { id: 'placeholders', name: 'Serine and Ethanol', twins: true, room: 'CHEMISTRY SHELF', anchor: 'fallback', ref: 'Section 5.2',
    brief: 'Sometimes linking finds nothing at all in a question or in an answer option. The model still expects a graph, so the code has to put something there. None of the reviewed papers says what. We read the released code.',
    quiz: { q: 'What do QA-GNN and GreaseLM put in the graph when a question links no concept?', options: ['Nothing', 'Serine, an amino acid', 'A random concept'], answer: 1 },
    game: { type: 'quiz' },
    stamp: 'Serine', stampLabel: 'stands in for any question that links nothing',
    title: 'Silent placeholders replace missing evidence',
    text: 'An unlinked question gets Serine, an amino acid. An unlinked answer option gets Ethanol. Neither concept is chosen for any principled reason, and because so many answer options link nothing, the Ethanol placeholder is used for most of them.',
    why: 'The model then reasons over a graph that has nothing to do with the question, and no paper reports it.' },

  { id: 'empty', name: 'The Empty Graph', room: 'TRAINING ROOM', anchor: 'retraining', ref: 'Section 5.3 and Table 4',
    brief: 'Step 3 gives the evidence to the model. QA-GNN, the reference system of this line of work, reads the graph with a graph neural network. Its paper reports 38.0% accuracy on MedQA, 0.8 points above the same language model without the graph. We retrained it under seven UMLS constructions, three times each.',
    quiz: { q: 'QA-GNN reaches 38.0% on MedQA with UMLS. What does it score when its graph holds only the two placeholders?', options: ['About 30%', 'About 34%', 'About 37%'], answer: 2 },
    game: { type: 'quiz' },
    stamp: '37.4%', stampLabel: 'QA-GNN accuracy with an empty graph',
    title: 'The gains hide in the noise',
    text: 'Across four releases and two vocabulary subsets, QA-GNN scores between 37.5% and 38.7%. With an empty graph that holds only the two placeholders, it still scores 37.4%. The published gains of this line of work, +0.8 points for QA-GNN and +0.5 more for GreaseLM, are no larger than the change from downloading another release.',
    why: 'GreaseLM does lose accuracy without a graph (36.7%), so whether UMLS helps depends on the system, and only controls like this one can tell.' },

  { id: 'readers', name: 'The Reader\u2019s Riddle', room: 'READING ROOM', anchor: 'readers', ref: 'Section 6, Figures 7 and 8',
    brief: 'Today\u2019s large language models can read UMLS evidence written as text in the prompt. We tested 10 models from 5 families on three benchmarks with three formats of evidence: plain relations around the concepts, only the paths that link question and answer, and paths enriched with definitions and semantic types.',
    quiz: { q: 'Is one evidence format the best for every model and every benchmark?', options: ['Yes, enriched paths always win', 'No, the winner changes', 'No, evidence never helps'], answer: 1 },
    game: { type: 'quiz' },
    stamp: '9 of 10', stampLabel: 'models prefer plain relations on PubMedQA',
    title: 'How UMLS is written into the prompt changes the result',
    text: 'Enriched paths help most for most models on MedQA and MedMCQA, while plain relations win for 9 of 10 models on PubMedQA. The winner changes again when the models reason step by step, and path length matters too: in the Qwen3.5 family, the 9B model works best with three-hop paths and the 35B model with four.',
    why: 'Papers should report the evidence format as carefully as the model, because it can change the result they obtain.' },
];

const CHECKLIST = [
  ['UMLS release', 'Which release was used, for example 2023AA.'],
  ['Sources and filters', 'Which vocabularies were kept and which concepts or relations were filtered out.'],
  ['Concept metadata', 'Which fields reach the model, such as definitions and semantic types.'],
  ['Entity linker', 'Which tool and which non-default settings.'],
  ['Linking coverage', 'How many questions and answer options link nothing.'],
  ['Connectivity rule', 'Which concepts are connected when collecting evidence.'],
  ['Depth and size', 'How many hops, and how large the collected subgraph is.'],
  ['Pruning', 'The rule and the node budget used to trim the subgraph.'],
  ['Fallback', 'What the system does when linking finds nothing.'],
];

const VERDICT = {
  title: 'Are we done with UMLS injection? Maybe not.',
  text: 'Releases, vocabulary filters, linkers, retrieval rules and fallbacks reshape UMLS before any model reads it. Only 11 of the 25 studies share their code, and only 6 let us recover how UMLS was built. UMLS has not yet been tested in a way that isolates its contribution.',
};

/* 3x5 pixel font and the toy blocks of the author line. */
const FONT = {
  A:'010101111101101', B:'110101110101110', C:'011100100100011', D:'110101101101110', E:'111100110100111',
  F:'111100110100100', G:'011100101101011', H:'101101111101101', I:'111010010010111', J:'001001001101010',
  K:'101101110101101', L:'100100100100111', M:'101111111101101', N:'110101101101101', O:'010101101101010',
  P:'110101110100100', Q:'010101101110011', R:'110101110101101', S:'011100010001110', T:'111010010010010',
  U:'101101101101111', V:'101101101101010', W:'101101111111101', X:'101101010101101', Y:'101101010010010',
  Z:'111001010100111', '0':'111101101101111', '1':'010110010010111', '2':'110001010100111', '3':'110001010001110',
  '4':'101101111001001', '5':'111100110001110', '6':'011100111101111', '7':'111001010010010', '8':'111101111101111',
  '9':'111101111001110', '.':'000000000000010', '!':'010010010000010', '?':'110001010000010', '-':'000000111000000',
  ':':'000010000010000', "'":'010010000000000', '%':'101001010100101', '/':'001001010100100', '+':'000010111010000',
  '=':'000111000111000', ',':'000000000010100', '>':'100010001010100', '<':'001010100010001', ' ':'000000000000000',
};
function blockSVG(letter) {
  const g = FONT[letter] || FONT[' '];
  let r = '';
  for (let i = 0; i < 15; i++) if (g[i] === '1') r += `<rect x="${3 + (i % 3)}" y="${4 + Math.floor(i / 3)}" width="1" height="1" fill="#4a3226"/>`;
  return `<svg viewBox="0 0 12 12" shape-rendering="crispEdges" aria-hidden="true"><rect x="1" y="0" width="11" height="2" fill="#dcc7b8"/><rect x="10" y="1" width="2" height="10" fill="#b79482"/><rect x="0" y="2" width="10" height="10" fill="#6f5141"/><rect x="1" y="3" width="8" height="8" fill="#f5efdf"/><rect x="1" y="3" width="8" height="1" fill="#b79482"/><rect x="1" y="10" width="8" height="1" fill="#b79482"/><rect x="1" y="3" width="1" height="8" fill="#b79482"/><rect x="8" y="3" width="1" height="8" fill="#b79482"/>${r}</svg>`;
}

window.UMLS = { PAPER, EXAMPLE, CONCEPT, STORY, SKELS, CHECKLIST, VERDICT, FONT, blockSVG };
})();
