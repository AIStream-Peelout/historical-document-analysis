"""Phase D of ja_ar_v1: dataset rows from gate v2 pairs (private dataset on the NAS + manifest + generated funnel).

A pair becomes a row when, in this order:

1. gate v2 accepted it (``decision == "accept"``);
2. it wins its photograph: among editions of the same text on one photograph (``dup_rank`` 0, 1, ...) the first one
   that clears every later stage is taken, the others are recorded as shadowed (no source priority);
3. every side it carries is novel (``hold-out v2`` tier of the pair's photograph: ``novel_strict`` or, when allowed,
   ``novel_page_level``; ``previously_trained`` is never built, ``uncertain_history`` goes to the review queue file);
4. no side is held out for evaluation (benchmarks, sealed sets, the registered eval draw, registry val splits);
5. its approval covers exactly the answer about to be written (:func:`final_answer`, :mod:`.approval`): the pair
   was approved, by the assembly's rule or by judges, with this photograph, rotation, crop, source text and final
   answer, and no unresolved line stands next to its text. A pair without such an approval is held
   (``no_approval``, ``approval_void``, ``unresolved_lines``); nothing changes a text after its approval;
6. its answer keeps at least ``MIN_LETTERS`` letters, carries none of the hygiene characters and fits the token
   budget of the training notebooks.

The answer of a pair is made in one place (:func:`final_answer`: the clean_v2 text rules, then the sign
normalisations every earlier build applied) and before adjudication, so judges see the text a row would hold. No
line is removed there on a reader's evidence: a line the character model finds unlikely is flagged
(``scrambled_candidate``) and stays. Until 2026-11-22 such a line was dropped when it also aligned worse than both
its neighbours; that removed 234 lines from 167 rows of genizah_novel_v2 after their approval, 199 of them lines
the reads did support (integrity audit of that date).

Rotation and crop remedies are applied to a derived copy of the photograph (``W2/images_derived``); every other row
points at the stored photograph. Rows use the KTIV ``FEATURES`` schema with the script-neutral prompt; split ``val``
(3 % of document groups, by hash) beside ``train_page``. A dataset holding legacy-store (FJP) or Russian National
Library material is private: README + PRIVATE.md say so, and nothing here pushes or publishes anything.

Example::

    PYTHONPATH=. .venv/bin/python -m src.finetuning.qwen_hebrew.ja_ar_build.build_rows \\
        --pairs W2/gate/run12/pairs.jsonl --line-evidence W2/gate/run12/line_evidence.jsonl \\
        --candidates W1/stages/stage1_images.jsonl --verdicts W2/novelty/verdicts_w1_draw.jsonl \\
        --out /Volumes/home/studio_offload/datasets/genizah_ja_ar_v1 [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from PIL import Image

from src.finetuning.qwen_hebrew.ja_ar_build import approval, assign, clean_v2, textnorm
from src.finetuning.qwen_hebrew.novel_build import quality as nb_quality

SECTION = "ja_ar_page"
#: Candidate source -> ``label_source`` of its rows (as in novel_build).
LABEL_SOURCES = {"pgp": "pgp_edition", "fjp": "legacy_store_diplomatic", "fjp_scrape": "legacy_store_scrape",
                 "ktiv": "ktiv_nli", "literature": "scanned_edition", "lit_run2": "run2_edition",
                 "apd_v3": "apd_edition", "aodeh_v2_1": "book_edition",
                 "fgp_targeted": "legacy_store_edition", "fgp_oct30_session": "legacy_store_edition",
                 "fjp_rnl_zip_1111": "legacy_store_edition", "book_edition": "book_edition"}
LEGACY_STORE_SOURCES = frozenset({"fjp", "fjp_scrape", "fgp_targeted", "fgp_oct30_session", "fjp_rnl_zip_1111"})
#: Candidate source -> the source part of a row stem, where the candidate's own source name must not reach a row
#: (stems travel with the dataset; the legacy store is never named outside this repository's private notes).
STEM_SOURCES = {"fgp_targeted": "lse", "fgp_oct30_session": "lse", "fjp_rnl_zip_1111": "lse"}
#: Gate remedy classes built by default. The blind audit of gate v2 outputs (2026-11-12, audit2, 80 pairs) found plain
#: placements complete in 36 of 43 judged pairs (as-is, kept margin blocks, kept edge lines; no wrong text) and the
#: repaired ones (trims, re-pairs, merges, crops) complete in 15 of 37: those are held in a repair queue for now.
DEFAULT_REMEDY_CLASSES = ("as_is", "margin_block", "edge_kept")
#: Tiers built by default (page-level novel rows carry their tier; the mixture decides whether to draw them).
DEFAULT_TIERS = ("novel_strict", "novel_page_level")
MIN_LETTERS = 30
#: A single-line pair under this many letters is a fragment of a text, not a page (blind audit 2026-11-12).
MIN_SINGLE_LINE_LETTERS = 80
#: Mean letters per line above this: the text is flattened (line breaks lost); held for re-lining (novel_v1 rule).
MAX_MEAN_LINE_LETTERS = 80
HYGIENE_CHARS = "␣&#$_{}<>\\"
#: Question + answer token budget of the training notebooks (context 12,288 - image 6,835 - 200 headroom).
TOKEN_BUDGET = 12288 - 6835 - 200
VAL_SHARE = 0.03
OUT_NAME = re.compile(r"^genizah_(?:ja_ar|novel)_v\d+")
PRIVATE_BANNER = ("**PRIVATE. Never publish, push or release this dataset or anything derived from it. It holds "
                  "legacy-store material and may hold Russian National Library material (training only).**")


ANGLE_PAIR = re.compile(r"<[^<>\n]*>")


def drop_lone_angles(text: str) -> Tuple[str, int]:
    """Drop a ``<`` or ``>`` that is not part of an angle pair on its line (merchants' fraction signs such as
    ``ב'>'`` that the earlier fraction rule leaves when a geresh follows the sign).

    :param text: Text.
    :return: ``(text, signs dropped)``.
    """
    out: List[str] = []
    dropped = 0
    for line in text.split("\n"):
        paired = {i for m in ANGLE_PAIR.finditer(line) for i in (m.start(), m.end() - 1)}
        kept = []
        for i, ch in enumerate(line):
            if ch in "<>" and i not in paired:
                dropped += 1
                continue
            kept.append(ch)
        out.append("".join(kept))
    return "\n".join(out), dropped


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Records of a JSON-lines file.

    :param path: File.
    :return: Records.
    """
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def slug(text: str) -> str:
    """File-name-safe slug.

    :param text: Any text.
    :return: Lower-case ASCII slug.
    """
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80] or "x"


def remedy_class(pair: Mapping[str, Any]) -> str:
    """The gate remedy class of a pair (the most invasive remedy wins).

    :param pair: Pair record.
    :return: ``assemble_split`` / ``assemble_crop`` / ``assemble`` (pairs of :mod:`.assemble`) / ``merge`` / ``crop`` / ``re_pair`` /
        ``trim_moved`` / ``trim_edge`` / ``margin_block`` / ``edge_kept`` / ``as_is``.
    """
    rem = set(pair.get("remedies") or [])
    flags = set(pair.get("flags") or [])
    if "assemble" in rem:
        return "assemble_split" if "split" in rem else "assemble_crop" if "crop" in rem else "assemble"
    if "merge" in rem:
        return "merge"
    if "crop" in rem:
        return "crop"
    if "re_pair" in rem:
        return "re_pair"
    if "trim" in rem:
        return "trim_moved" if "lines_on_other_face" in flags else "trim_edge"
    if "margin_block_kept" in flags:
        return "margin_block"
    if "edge_kept" in flags:
        return "edge_kept"
    return "as_is"


def pair_tier(pair: Mapping[str, Any], verdicts: Mapping[Tuple[str, str], Mapping[str, Any]],
              tiers: Sequence[str]) -> Tuple[Optional[str], List[str]]:
    """Novelty tier of a pair (worst of its sides, on the pair's photograph) and blocking reasons.

    A side without a side-level verdict takes its document's verdict (``side_key ""``): hold-out v2 runs on ledger
    rows judge a document as a whole, and that verdict covers every side of it.

    :param pair: Pair record.
    :param verdicts: Hold-out v2 verdicts by (row_id, side_key).
    :param tiers: Tiers allowed into the dataset.
    :return: ``(tier or None, reasons)``; reasons empty when the pair may be built.
    """
    order = {"novel_strict": 0, "novel_page_level": 1, "uncertain_history": 2, "previously_trained": 3}
    worst: Optional[str] = None
    reasons: List[str] = []
    for sk in pair["side_keys"]:
        v = verdicts.get((pair["row_id"], sk)) or verdicts.get((pair["row_id"], ""))
        if v is None:
            reasons.append("no_novelty_verdict")
            continue
        tier = (v.get("pair_tiers") or {}).get(pair["image_sha256"]) or v.get("tier")
        if worst is None or order.get(tier, 9) > order.get(worst, 9):
            worst = tier
        if v.get("eval_holdout"):
            reasons.append("eval_holdout:" + ",".join(sorted(set(v.get("eval_families") or ["?"]))))
    if worst is not None and worst not in tiers:
        reasons.append(f"tier:{worst}")
    return worst, reasons


def clean_pair(pair: Mapping[str, Any], cands: Mapping[str, Mapping[str, Any]], models: Mapping[str, Any],
               mirrored: Set[str]) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[int]]:
    """Clean a pair's kept lines side by side with the clean_v2 text rules. No line is removed on a read's evidence.

    A line the character model finds unlikely is flagged (``scrambled_candidate``) and stays in the text: it may
    be garbled source text or sound text in an unusual hand, and removing it shortens a target the photograph
    still shows in full.

    :param pair: Pair record (``line_refs`` = ``[side_key, line index]`` in output order).
    :param cands: Candidate records by row_id (for each side's full line list).
    :param models: clean_v2 3-gram models.
    :param mirrored: clean_v2 mirrored editions.
    :return: ``(cleaned lines, clean log, flags, numbers of the pair's lines flagged as scrambled)``; the numbers
        are 1-based positions in ``pair["lines"]``.
    :raises ValueError: When the pair's lines are not the candidate's lines at its ``line_refs``
        (:func:`source_segments`).
    """
    segments = source_segments(pair, cands)
    if segments is None:
        raise ValueError(f"pair {pair['pair_id']}: its lines are not the candidate's lines at its line_refs")
    out: List[str] = []
    log: List[Dict[str, Any]] = []
    flags: List[str] = []
    scrambled: List[int] = []
    source = pair["source"]
    offset = 0
    for sk, kept in segments:
        res = clean_v2.clean_side(kept, source=source, script=pair["script"], model=models,
                                  brackets_mirrored=clean_v2.work_key(pair["row_id"], source) in mirrored)
        for f in res.flags:
            if f.get("flag") == "scrambled_candidate" and f.get("out_index") is not None:
                scrambled.append(offset + int(f["line_index"]) + 1)
                flags.append("scrambled_candidate")
            if f.get("flag") in ("flattened", "side_flat"):
                flags.append("flattened")
        out.extend(res.lines)
        log.extend({**e, "side_key": sk} for e in res.clean_log)
        offset += len(kept)
    return out, log, sorted(set(flags)), sorted(set(scrambled))


def source_segments(pair: Mapping[str, Any], cands: Mapping[str, Mapping[str, Any]]
                    ) -> Optional[List[Tuple[str, List[str]]]]:
    """A pair's lines as the candidate holds them, in runs of consecutive lines of one side.

    Consecutive lines of one side are cleaned together and the pair's own order is kept: an assembled pair
    (:mod:`.assemble`) may return to a side after another side's lines (two pages of an opening).

    :param pair: Pair record (``line_refs`` = ``[side_key, line index]`` in output order).
    :param cands: Candidate records by row_id.
    :return: ``[(side key, lines)]``, or None when the pair does not fit the candidate: an unknown candidate or
        side, a line index the side does not have, or lines that differ from the pair's own (a pair record made
        from another version of the candidates file).
    """
    cand = cands.get(pair["row_id"])
    if cand is None:
        return None
    known = {(t.get("side_key") or "") for t in cand.get("targets") or []}
    segments: List[Tuple[str, List[str]]] = []
    taken: List[str] = []
    for sk, i in pair["line_refs"]:
        if sk not in known:
            return None
        lines = approval.side_lines(cand, sk)
        if not 0 <= i < len(lines):
            return None
        if segments and segments[-1][0] == sk:
            segments[-1][1].append(lines[i])
        else:
            segments.append((sk, [lines[i]]))
        taken.append(lines[i])
    return segments if taken == list(pair["lines"]) else None


@dataclass
class Answer:
    """The final answer of a pair: the text its row would hold.

    :ivar text: The answer.
    :ivar lines: Its non-empty lines.
    :ivar flags: Cleaning and normalisation flags.
    :ivar log: clean_v2 change log.
    :ivar fails: Reasons the text cannot be a row (sign normalisation failures, a stale pair record).
    :ivar scrambled_lines: Lines the character model flags (kept in the text).
    :ivar scrambled_at: Their numbers among the pair's lines (1-based).
    :ivar letters_kept: Whether the answer holds the letters of every line of the pair, line for line (lines without
        a letter aside). False when a text rule removed or changed letters after the pair's lines were chosen: the
        answer then is not the text a rule or a judge looked at.
    """

    text: str
    lines: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    log: List[Dict[str, Any]] = field(default_factory=list)
    fails: List[str] = field(default_factory=list)
    scrambled_lines: int = 0
    scrambled_at: List[int] = field(default_factory=list)
    letters_kept: bool = True


def final_answer(pair: Mapping[str, Any], cands: Mapping[str, Mapping[str, Any]], models: Mapping[str, Any],
                 mirrored: Set[str]) -> Answer:
    """The answer a pair's row would hold: clean_v2 text rules, then the sign normalisations of every earlier build.

    This is the only place a pair's text is made. Judges are shown its lines (``audit/make_items.py``), approvals
    are bound to it (:func:`approval.fingerprint`) and :func:`build` writes it unchanged.

    :param pair: Pair record.
    :param cands: Candidate records by row_id.
    :param models: clean_v2 3-gram models.
    :param mirrored: clean_v2 mirrored editions.
    :return: The answer.
    """
    if source_segments(pair, cands) is None:                            # a pair of another candidates file
        return Answer(text="", fails=["pair_lines_differ_from_candidate"], letters_kept=False)
    lines, log, cflags, scrambled = clean_pair(pair, cands, models, mirrored)
    # the sign normalisations every earlier build applied (presentation forms, harakat, fraction signs,
    # reversed supplied letters), so rows read like the rows the models trained on
    text, nflags, nfails = nb_quality.final_text("\n".join(lines).strip(), pair["source"])
    text, lone = drop_lone_angles(text.strip())
    flags = sorted(set(cflags) | set(nflags) | ({"lone_angle_dropped"} if lone else set()))
    out_lines = [ln for ln in text.split("\n") if ln.strip()]
    before = [s for s in (textnorm.skeleton(ln) for ln in pair["lines"]) if s]
    after = [s for s in (textnorm.skeleton(ln) for ln in out_lines) if s]
    return Answer(text=text, lines=out_lines, flags=flags, log=log, fails=list(nfails),
                  scrambled_lines=len(scrambled), scrambled_at=scrambled, letters_kept=before == after)


def pair_fingerprint(pair: Mapping[str, Any], answer: Answer, cands: Mapping[str, Mapping[str, Any]]) -> str:
    """Fingerprint of a pair as it would be built now (:func:`approval.fingerprint`).

    :param pair: Pair record.
    :param answer: Its final answer.
    :param cands: Candidate records by row_id.
    :return: Hex SHA-256.
    """
    return approval.fingerprint(pair, answer.text, approval.source_digest(cands[pair["row_id"]], pair["side_keys"]))


def stamp_approval(pair: MutableMapping[str, Any], cands: Mapping[str, Mapping[str, Any]], models: Mapping[str, Any],
                   mirrored: Set[str], by: str, **extra: Any) -> Optional[str]:
    """Bind an accepted pair's approval to its final answer, or say why it cannot be bound.

    :param pair: Accepted pair record (changed in place: ``approval`` is set).
    :param cands: Candidate records by row_id.
    :param models: clean_v2 3-gram models.
    :param mirrored: clean_v2 mirrored editions.
    :param by: Approval from the assembly rule, a direct edition and photograph review, or the judges
        (``rule``, ``source_review``, or ``judges``).
    :param extra: Further fields of the approval record (audit name, item id, judges).
    :return: None when the approval is bound; otherwise the reason: ``pair_lines_differ_from_candidate``,
        ``no_line_accounting`` (a rule approval of a pair whose record says nothing about the lines beside its
        text: an assembly run made before :func:`assemble.attach_open_lines`, or a gate that does not account for
        the lines it leaves out), ``unresolved_lines``,
        ``answer_changed_by_cleaning`` (a text rule removed or changed letters of the lines that were approved;
        judges must see the answer itself) or ``unlikely_unread_line`` (a rule approval of a text with a line that
        the character model finds unlikely and that no read supports either: nothing vouches for its letters, so a
        judge must look; the line is never removed. Measured 2026-11-22 on 862 rows: 365 of 376 flagged lines had
        read support, which is what vouches for them; 11 had none).
    """
    answer = final_answer(pair, cands, models, mirrored)
    if answer.fails and "pair_lines_differ_from_candidate" in answer.fails:
        return "pair_lines_differ_from_candidate"
    if by == "rule" and "open_lines" not in pair:
        return "no_line_accounting"
    if approval.unsettled_open_lines(pair):
        return "unresolved_lines"
    if not answer.letters_kept and not extra.get("answer_seen"):
        return "answer_changed_by_cleaning"
    if by == "rule" and set(answer.scrambled_at) & set(pair.get("unread_lines") or []):
        return "unlikely_unread_line"
    pair["approval"] = {"by": by, "scope": approval.SCOPE, "fingerprint": pair_fingerprint(pair, answer, cands),
                        "answer_sha256": hashlib.sha256(answer.text.encode("utf-8")).hexdigest(),
                        "answer_lines": len(answer.lines), **extra}
    return None


def derived_image(pair: Mapping[str, Any], out_dir: Path) -> Tuple[str, int, int]:
    """The photograph as the row shows it: rotation and crop applied to a derived copy, else the stored file.

    :param pair: Pair record.
    :param out_dir: Folder of derived copies.
    :return: ``(image path, width, height)``.
    """
    path = pair["image_path"]
    rotation = int(pair.get("rotation") or 0)
    crop = pair.get("crop")
    if not rotation and not crop:
        with Image.open(path) as im:
            return path, im.width, im.height
    tag = hashlib.sha1(json.dumps([pair["image_sha256"], rotation, crop]).encode()).hexdigest()[:16]
    out = out_dir / f"{pair['image_sha256'][:16]}_{tag}.jpg"
    if not out.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as im:
            im = im.convert("RGB")
            if crop:
                im = im.crop(tuple(int(v) for v in crop))
            if rotation:
                im = im.rotate(-rotation, expand=True)
            im.save(out, format="JPEG", quality=95)
    with Image.open(out) as im:
        return str(out), im.width, im.height


def split_of(document_group: str) -> str:
    """Split of a document group: a stable 3 % val sample by hash.

    :param document_group: Group id.
    :return: ``val`` or ``train_page``.
    """
    h = int(hashlib.sha256(f"ja_ar_v1|{document_group}".encode()).hexdigest()[:8], 16)
    return "val" if (h % 10000) < VAL_SHARE * 10000 else "train_page"


def load_prior(manifests: Sequence[Path]) -> Dict[str, Set[Any]]:
    """Photographs and sides that earlier builds made into rows.

    :param manifests: ``manifest.jsonl`` files of earlier builds.
    :return: ``{"images": {sha256}, "sides": {(row_id, side_key)}}``.
    """
    images: Set[str] = set()
    sides: Set[Tuple[str, str]] = set()
    for path in manifests:
        for m in load_jsonl(path):
            images.add(m["image_sha256"])
            sides.update((m["row_id"], k) for k in m.get("side_keys") or [])
    return {"images": images, "sides": sides}


def load_eval_photos(registered_dir: Path) -> Dict[str, Set[str]]:
    """Photographs of every registered evaluation set in a folder (digests and pHashes of their items).

    :param registered_dir: Folder of ``holdout_v2`` registered sets (``*.json``).
    :return: ``{"sha256": {...}, "phash": {...}}``.
    """
    shas: Set[str] = set()
    phashes: Set[str] = set()
    for path in sorted(Path(registered_dir).glob("*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")).get("items") or []:
            shas.update(item.get("image_sha256") or [])
            phashes.update(item.get("phash") or [])
    return {"sha256": shas, "phash": phashes}


def photo_phash(pair: Mapping[str, Any], cands: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    """pHash of a pair's photograph from its candidate's image list.

    :param pair: Pair record.
    :param cands: Candidates by row_id.
    :return: Hex pHash, or None.
    """
    for im in (cands.get(pair["row_id"]) or {}).get("images") or []:
        if im.get("sha256") == pair["image_sha256"]:
            return im.get("phash")
    return None


def is_eval_photo(sha: str, phash: Optional[str], eval_photos: Mapping[str, Set[str]], max_bits: int) -> bool:
    """Whether a photograph is (a near copy of) a registered evaluation photograph.

    :param sha: Photograph digest.
    :param phash: Its pHash, or None.
    :param eval_photos: :func:`load_eval_photos` result.
    :param max_bits: Largest pHash distance that counts as the same photograph.
    :return: True when held out.
    """
    if sha in eval_photos["sha256"]:
        return True
    if not phash:
        return False
    for other in eval_photos["phash"]:
        dist = assign.phash_distance(phash, other)
        if dist is not None and dist <= max_bits:
            return True
    return False


#: Folder of the clean_v2 3-gram models of this round.
MODELS_DIR = Path("/Volumes/home/studio_offload/datasets/genizah_ja_ar_v1_work/models")


def load_text_tools(cands: Mapping[str, Mapping[str, Any]], models_dir: Path = MODELS_DIR
                    ) -> Tuple[Mapping[str, Any], Set[str]]:
    """What :func:`final_answer` needs beside the pairs and candidates.

    :param cands: Candidate records by row id.
    :param models_dir: Folder of the clean_v2 3-gram models.
    :return: ``(models, mirrored editions)``.
    """
    models = clean_v2.load_models(models_dir)
    mirrored = clean_v2.mirrored_works((c["row_id"], c["source"], [ln for t in c.get("targets") or []
                                                                   for ln in t.get("lines") or []])
                                       for c in cands.values())
    return models, mirrored


def file_sha256(path: str) -> str:
    """Digest of a file.

    :param path: File path.
    :return: Hex SHA-256.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(pairs: Sequence[Mapping[str, Any]], cands: Mapping[str, Mapping[str, Any]],
          verdicts: Mapping[Tuple[str, str], Mapping[str, Any]],
          models: Mapping[str, Any], mirrored: Set[str], tokenizer: Any, tiers: Sequence[str],
          derived_dir: Path, params: assign.GateParams, write_images: bool = True,
          remedy_classes: Sequence[str] = DEFAULT_REMEDY_CLASSES, prior: Optional[Mapping[str, Set[Any]]] = None,
          eval_photos: Optional[Mapping[str, Set[str]]] = None
          ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select, clean and check pairs; return rows, manifest records and one decision per pair.

    :param pairs: Gate v2 pair records.
    :param cands: Candidate records by row_id.
    :param verdicts: Hold-out v2 verdicts by (row_id, side_key).
    :param models: clean_v2 3-gram models.
    :param mirrored: clean_v2 mirrored editions.
    :param tokenizer: Training tokenizer (None: tokens not measured, flagged).
    :param tiers: Tiers allowed.
    :param derived_dir: Folder of derived photographs.
    :param params: Gate thresholds (duplicate-text test).
    :param write_images: Write derived photographs (False in dry runs: sizes then come from the record).
    :param remedy_classes: Gate remedy classes built (:func:`remedy_class`); the others are held for repair.
    :param prior: Photographs and sides of earlier builds (:func:`load_prior`): a pair on a photograph or side that
        already became a row is excluded (an incremental build never repeats an earlier row's photograph).
    :param eval_photos: Photographs of the registered evaluation sets (:func:`load_eval_photos`): a pair on one of
        them (same digest, or pHash within ``params.phash_same_face``) is held out whatever its verdict says, since a
        verified page may use a photograph the draw did not register.
    :return: ``(rows, manifest, decisions)``.
    """
    from src.finetuning.qwen_hebrew.prompts import SCRIPT_NEUTRAL_PROMPT
    from src.finetuning.qwen_hebrew.build_pgp_arabic_editions import TASK

    by_photo: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    decisions: List[Dict[str, Any]] = []
    for p in pairs:
        if p["decision"] != "accept":
            decisions.append({"pair_id": p["pair_id"], "stage": "gate", "decision": p["decision"],
                              "reasons": p["reasons"], "source": p["source"], "script": p["script"],
                              "letters": p["evidence"].get("target_letters", 0)})
            continue
        by_photo[p["image_sha256"]].append(p)
    rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    for sha in sorted(by_photo):
        taken: List[str] = []
        for p in sorted(by_photo[sha], key=lambda x: (x.get("dup_rank") or 0, x["pair_id"])):
            dec = {"pair_id": p["pair_id"], "source": p["source"], "script": p["script"],
                   "letters": p["evidence"].get("target_letters", 0)}
            if prior and (sha in prior["images"] or any((p["row_id"], k) in prior["sides"] for k in p["side_keys"])):
                reason = "prior_build_photo" if sha in prior["images"] else "prior_build_side"
                decisions.append({**dec, "stage": "prior_build", "decision": "excluded", "reasons": [reason]})
                continue
            text_gate = "\n".join(p["lines"])
            if any(assign._texts_duplicate(text_gate, t, params) for t in taken):
                decisions.append({**dec, "stage": "duplicate", "decision": "shadowed", "reasons": ["dup_shadowed"]})
                continue
            rclass = remedy_class(p)
            if rclass not in remedy_classes:
                decisions.append({**dec, "stage": "repair_queue", "decision": "held", "reasons": [f"remedy:{rclass}"],
                                  "remedy_class": rclass})
                continue
            tier, reasons = pair_tier(p, verdicts, tiers)
            if eval_photos and is_eval_photo(sha, photo_phash(p, cands), eval_photos, params.phash_same_face):
                reasons = ["eval_holdout:eval_photo"] + reasons
            if reasons:
                stage = "holdout" if any(r.startswith("eval_holdout") for r in reasons) else "novelty"
                decisions.append({**dec, "stage": stage, "decision": "excluded", "reasons": reasons, "tier": tier})
                continue
            answer = final_answer(p, cands, models, mirrored)
            # the approval must cover exactly this photograph, crop, rotation, source text and answer
            state = None if answer.fails else approval.approval_state(p, pair_fingerprint(p, answer, cands))
            if state:
                decisions.append({**dec, "stage": "approval", "decision": "held", "reasons": [state], "tier": tier})
                continue
            text, log, cflags = answer.text, answer.log, answer.flags
            q_reasons: List[str] = list(answer.fails)
            n_letters = textnorm.letters(text)
            text_lines = [ln for ln in text.split("\n") if ln.strip()]
            if n_letters < MIN_LETTERS:
                q_reasons.append("too_few_letters")
            elif len(text_lines) < 2 and n_letters < MIN_SINGLE_LINE_LETTERS:
                q_reasons.append("short_single_line")
            if text_lines and n_letters / len(text_lines) > MAX_MEAN_LINE_LETTERS:
                q_reasons.append("flattened_needs_reline")
            if any(ch in text for ch in HYGIENE_CHARS):
                q_reasons.append("hygiene_chars")
            tokens = None
            if tokenizer is not None:
                tokens = (len(tokenizer(text, add_special_tokens=False)["input_ids"])
                          + len(tokenizer(SCRIPT_NEUTRAL_PROMPT, add_special_tokens=False)["input_ids"]))
                if tokens > TOKEN_BUDGET:
                    q_reasons.append("token_over_budget")
            if q_reasons:
                decisions.append({**dec, "stage": "quality", "decision": "excluded", "reasons": q_reasons,
                                  "tier": tier})
                continue
            cand = cands[p["row_id"]]
            if write_images:
                if file_sha256(p["image_path"]) != sha:               # the approval names this photograph's bytes
                    decisions.append({**dec, "stage": "approval", "decision": "held",
                                      "reasons": ["image_file_changed"], "tier": tier})
                    continue
                image, width, height = derived_image(p, derived_dir)
                if p.get("edition") and file_sha256(image) != p["edition"]["view_sha256"]:
                    raise ValueError(f"Exported image differs from the reviewed view of {p['pair_id']}")
            else:
                image, width, height = p["image_path"], 0, 0
            stem = (f"jaar_{slug(STEM_SOURCES.get(p['source'], p['source']))}_{slug(p['row_id'])}_"
                    f"{slug('+'.join(p['side_keys']))}_{sha[:10]}")
            if "split" in (p.get("remedies") or []):                    # the two pages of one photograph
                stem += "_" + str((p.get("evidence") or {}).get("page") or "page")[0]
            if p.get("edition"):
                stem += "_" + p["edition"]["view_sha256"][:12]
            split = "train_page" if p.get("edition") and cand.get("rnl") else split_of(p["document_group"])
            ja = textnorm.ja_signal(text) if p["script"] != "arabic" else False
            rows.append({"image": image, "question": SCRIPT_NEUTRAL_PROMPT, "answer": text, "task": TASK,
                         "section": SECTION, "stem": stem, "label_source": LABEL_SOURCES.get(p["source"], p["source"]),
                         "target_chars": len(text), "target_tokens": int(tokens or 0), "image_width": width,
                         "image_height": height, "_split": split})
            ev = p["evidence"]
            manifest.append({
                "stem": stem, "split": split, "pair_id": p["pair_id"], "row_id": p["row_id"],
                "side_keys": p["side_keys"], "source": p["source"], "document_group": p["document_group"],
                "canonical_ids": cand.get("canonical_ids") or [], "pgpids": cand.get("pgpids") or [],
                "shelfmark": cand.get("shelfmark"), "rnl": bool(cand.get("rnl")),
                "image_sha256": sha, "image_path": p["image_path"], "row_image": image,
                "rotation": p.get("rotation"), "crop": p.get("crop"), "remedies": p["remedies"],
                "gate_flags": p["flags"], "remedy_class": rclass, "clean_flags": cflags, "clean_changes": len(log),
                "scrambled_lines": answer.scrambled_lines, "approval": dict(p["approval"]),
                "row_image_sha256": file_sha256(image) if write_images else None,
                "open_lines_settled": sum(len(e.get("lines") or []) for e in p.get("open_lines") or []),
                "tier": tier, **({"edition": p["edition"]} if p.get("edition") else {}),
                "script": textnorm.script_family(text) if p["script"] != "mixed" else "mixed", "ja_signal": ja,
                "letters": n_letters, "target_tokens": tokens,
                "evidence": {k: ev.get(k) for k in ("reader", "k_chain", "k_null_max", "chain_q", "q_null_max",
                                                     "support_share", "uncovered_share", "interior_share")}})
            decisions.append({**dec, "stage": "row", "decision": "row", "reasons": [], "tier": tier, "stem": stem,
                              "row_letters": n_letters, "ja_signal": ja})
            taken.append(text_gate)
    return rows, manifest, decisions


def funnel_md(decisions: Sequence[Mapping[str, Any]], manifest: Sequence[Mapping[str, Any]]) -> str:
    """Generated funnel of a build.

    :param decisions: One record per pair.
    :param manifest: Row manifest.
    :return: Markdown.
    """
    lines = ["# genizah_ja_ar_v1 build funnel (generated)", ""]
    t = Counter()
    letters = Counter()
    for d in decisions:
        key = (d["source"], d["script"], d["stage"], d["decision"])
        t[key] += 1
        letters[key] += d.get("letters", 0)
    lines += ["## Pairs by source x script x last stage", "", "| source | script | stage | decision | pairs | letters |",
              "|---|---|---|---|---:|---:|"]
    for k in sorted(t):
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {t[k]} | {letters[k]:,} |")
    reasons = Counter(r for d in decisions for r in d.get("reasons") or [])
    lines += ["", "## Exclusion reasons", "", "| reason | pairs |", "|---|---:|"]
    lines += [f"| {r} | {n} |" for r, n in reasons.most_common()]
    rows = Counter((m["source"], m["script"], m["ja_signal"], m["tier"], m["split"]) for m in manifest)
    rl = Counter()
    for m in manifest:
        rl[(m["source"], m["script"], m["ja_signal"], m["tier"], m["split"])] += m["letters"]
    lines += ["", "## Rows by source x script x JA signal x tier x split", "",
              "| source | script | JA signal | tier | split | rows | letters |", "|---|---|---|---|---|---:|---:|"]
    for k in sorted(rows):
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {rows[k]} | {rl[k]:,} |")
    rem = Counter(",".join(m["remedies"]) or "as_is" for m in manifest)
    lines += ["", "## Rows by gate remedy", "", "| remedies | rows |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in rem.most_common()]
    return "\n".join(lines) + "\n"


def stats_of(manifest: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summary statistics.

    :param manifest: Row manifest.
    :return: Counts and letters by split, script, JA signal, tier and source.
    """
    out: Dict[str, Any] = {"rows": len(manifest), "letters": sum(m["letters"] for m in manifest)}
    for key in ("split", "script", "ja_signal", "tier", "source"):
        c: Counter = Counter()
        lt: Counter = Counter()
        for m in manifest:
            c[str(m[key])] += 1
            lt[str(m[key])] += m["letters"]
        out[f"rows_by_{key}"] = dict(c)
        out[f"letters_by_{key}"] = dict(lt)
    out["rnl_rows"] = sum(1 for m in manifest if m["rnl"])
    out["legacy_store_rows"] = sum(1 for m in manifest if m["source"] in LEGACY_STORE_SOURCES)
    out["distinct_images"] = len({m["image_sha256"] for m in manifest})
    out["document_groups"] = len({m["document_group"] for m in manifest})
    return out


def readme(stats: Mapping[str, Any], remedy_classes: Sequence[str] = DEFAULT_REMEDY_CLASSES,
           name: str = "genizah_ja_ar_v1") -> str:
    """README text (no storage paths, hosts or source names of the legacy store).

    :param stats: Build statistics.
    :param remedy_classes: Gate remedy classes this build admitted.
    :param name: Dataset name (the output folder's name).
    :return: Markdown.
    """
    banner = PRIVATE_BANNER if stats.get("rnl_rows") else (
        "**PRIVATE. Never publish, push or release this dataset or anything derived from it. It holds legacy-store "
        "material.**")
    return "\n".join([
        f"# {name}", "", banner, "",
        "Page transcription rows (Judaeo-Arabic and Arabic-script emphasis) from material no earlier fine-tune trained",
        "on. Target lines are placed on photographs by aligned reads (two evidence channels, each tested against its own",
        "null); evaluation sets are held out by identity and long-run text.",
        f"This version admits the gate's plain placements only ({', '.join(remedy_classes)}); pairs the gate repaired",
        "(trimmed, re-paired, merged or cropped) wait in a repair queue until a later version confirms them.", "",
        f"- rows: {stats['rows']:,}; letters: {stats['letters']:,}",
        f"- by split: {stats['rows_by_split']}", f"- by script: {stats['rows_by_script']}",
        f"- by JA signal: {stats['rows_by_ja_signal']}", f"- by novelty tier: {stats['rows_by_tier']}", "",
        "Columns: KTIV page schema (image, question = script-neutral prompt, answer, task, section, stem, label_source,",
        "target_chars, target_tokens, image_width, image_height).", ""])


def main(argv: Optional[List[str]] = None) -> None:
    """Build (or dry-run) the dataset.

    :param argv: Arguments (None: sys.argv).
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", type=Path, nargs="+", required=True)
    ap.add_argument("--line-evidence", type=Path, nargs="*", default=[],
                    help="accepted for older build scripts and not read: no line is removed on a read's evidence")
    ap.add_argument("--candidates", type=Path, nargs="+", required=True)
    ap.add_argument("--verdicts", type=Path, nargs="+", required=True)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--derived-dir", type=Path, default=None,
                    help="folder of the rotated / cropped copies (default: <out>_derived, made for this build "
                         "alone: a copy left by an earlier build or by older transform code is never reused)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tiers", default=",".join(DEFAULT_TIERS))
    ap.add_argument("--remedy-classes", default=",".join(DEFAULT_REMEDY_CLASSES),
                    help="gate remedy classes to build (others go to the repair queue)")
    ap.add_argument("--no-tokens", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="select and check only: manifest + funnel, no dataset")
    ap.add_argument("--prior-manifests", type=Path, nargs="*", default=[],
                    help="manifest.jsonl of earlier builds: their photographs and sides are not built again")
    ap.add_argument("--registered-dir", type=Path,
                    default=Path("/Volumes/home/studio_offload/datasets/genizah_ja_ar_v1_work/eval/registered"),
                    help="registered evaluation sets: pairs on their photographs are held out")
    args = ap.parse_args(argv)
    if not OUT_NAME.match(args.out.name):
        raise SystemExit(f"refusing to write {args.out}: the name must start with genizah_ja_ar_v<N> or "
                         "genizah_novel_v<N>")
    pairs = [p for path in args.pairs for p in load_jsonl(path)]
    cands = {c["row_id"]: c for path in args.candidates for c in load_jsonl(path)}
    verdicts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in args.verdicts:
        for v in load_jsonl(path):
            verdicts[(v["row_id"], v["side_key"])] = v
    models, mirrored = load_text_tools(cands, args.models_dir)
    tokenizer = None
    if not args.no_tokens:
        from src.finetuning.qwen_hebrew.novel_build import quality as nb_quality
        tokenizer = nb_quality.load_tokenizer()
    derived_dir = args.derived_dir or args.out.with_name(args.out.name + "_derived")
    if not args.dry_run and derived_dir.exists() and any(derived_dir.iterdir()):
        raise SystemExit(f"{derived_dir} holds files of another build; derived photographs are made per build")
    rows, manifest, decisions = build(pairs, cands, verdicts, models, mirrored, tokenizer,
                                      tuple(args.tiers.split(",")), derived_dir, assign.GateParams(),
                                      write_images=not args.dry_run,
                                      remedy_classes=tuple(args.remedy_classes.split(",")),
                                      prior=load_prior(args.prior_manifests) if args.prior_manifests else None,
                                      eval_photos=load_eval_photos(args.registered_dir))
    stats = stats_of(manifest)
    report_dir = args.out.with_name(args.out.name + "_build") if not args.dry_run else \
        args.out.with_name(args.out.name + "_dryrun")
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(report_dir / "decisions.jsonl", "w", encoding="utf-8") as fh:
        for d in decisions:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    (report_dir / "FUNNEL.md").write_text(funnel_md(decisions, manifest), encoding="utf-8")
    (report_dir / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("rows", "letters", "rows_by_split", "rows_by_script", "rows_by_ja_signal",
                                            "rows_by_tier")}, ensure_ascii=False))
    held = Counter(r for d in decisions if d["stage"] == "approval" for r in d["reasons"])
    print("accepted pairs held because their approval does not cover the answer:", dict(held))
    stale = sum(1 for d in decisions if "pair_lines_differ_from_candidate" in (d.get("reasons") or []))
    if stale:
        raise SystemExit(f"{stale} accepted pairs do not fit the candidates file (see decisions.jsonl): the pairs "
                         "were made from another version of it. Nothing is built from stale records.")
    if args.dry_run:
        return
    from src.finetuning.qwen_hebrew.build_pgp_editions import save_dataset
    splits: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        split = r.pop("_split")
        splits[split].append(r)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    remedy_classes = tuple(args.remedy_classes.split(","))
    save_dataset(dict(splits), args.out, {"README.md": readme(stats, remedy_classes, args.out.name), "PRIVATE.md": PRIVATE_BANNER + "\n",
                                          "stats.json": json.dumps({**stats, "built": built}, indent=1)})
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
