import os
import json
import numpy as np
from wordfreq import word_frequency
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import argparse

# Map sensor name -> JSON filename
SENSOR_FILES = {
    "thermal": "thermal_word_dict.json",
    "mmwave": "mmwave_word_dict.json",
    "acoustic": "acoustic_word_dict.json",
}

# ---------- Helpers ----------
def ensure_dir(path):
    if path is not None:
        os.makedirs(path, exist_ok=True)


# ---------- Helpers for loading / saving ----------

def _extract_groups(data):
    """
    Handle both:
    1) { "subgroup": [items...] }
    2) { "sensor_name": { "subgroup": [items...] } }
    Returns (groups_dict, root_key_or_None).
    """
    # Case 1: subgroups at top level
    if data and all(isinstance(v, list) for v in data.values()):
        return data, None

    # Case 2: single sensor key at top level
    if len(data) == 1:
        root_key = next(iter(data.keys()))
        inner = data[root_key]
        if isinstance(inner, dict) and all(isinstance(v, list) for v in inner.values()):
            return inner, root_key

    raise ValueError("Unsupported JSON structure for word dict.")


def _wrap_like_original(probs, root_key):
    """
    Reverse of _extract_groups: if there was a root_key, wrap again.
    probs: { subgroup: { item: prob } }
    """
    if root_key is None:
        return probs
    else:
        return {root_key: probs}


def load_word_groups(sensor, base_dir="."):
    """Load the word dictionary for a sensor and return (groups, root_key, path)."""
    if sensor not in SENSOR_FILES:
        raise ValueError(f"Unknown sensor '{sensor}'. Expected one of {list(SENSOR_FILES.keys())}.")

    path = os.path.join(base_dir, SENSOR_FILES[sensor])
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups, root_key = _extract_groups(data)
    return groups, root_key, path


def load_item_probs(sensor, lang="en", base_dir="."):
    """
    Load item probabilities for a sensor if the probs file exists;
    otherwise compute from wordfreq on the fly (but DO NOT write).
    Returns { subgroup: { item: prob } }.
    """
    prob_path = os.path.join(base_dir, f"{sensor}_word_probs_{lang}.json")
    if os.path.exists(prob_path):
        with open(prob_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # First: handle nested case {root_key: {subgroup: {item: prob}}}
        if len(data) == 1:
            inner = next(iter(data.values()))
            if isinstance(inner, dict) and all(isinstance(v, dict) for v in inner.values()):
                # inner is {subgroup: {item: prob}}
                return inner

        # Second: handle flat case {subgroup: {item: prob}}
        if data and all(isinstance(v, dict) for v in data.values()):
            return data

        raise ValueError("Unsupported JSON structure in probability file.")

    else:
        # Compute on the fly if probability file does not exist
        groups, _, _ = load_word_groups(sensor, base_dir)
        probs = {}
        for subgroup, items in groups.items():
            probs[subgroup] = {}
            for item in items:
                p = float(word_frequency(item, lang))
                probs[subgroup][item] = p
        return probs


# ---------- Mode 1: Language modeling (prob file) ----------

def build_probability_file_for_sensor(sensor, lang="en", base_dir=".", out_dir=None):
    """
    For a single sensor:
    - Load its word dict JSON.
    - Compute wordfreq probability for every item.
    - Store them in <sensor>_word_probs_<lang>.json with the same hierarchy.
    """
    if out_dir is None:
        out_dir = base_dir
    
    ensure_dir(out_dir)  # <--- add this


    groups, root_key, _ = load_word_groups(sensor, base_dir)

    probs = {}
    for subgroup, items in groups.items():
        probs[subgroup] = {}
        for item in items:
            p = float(word_frequency(item, lang))  # probability from wordfreq
            probs[subgroup][item] = p

    out_data = _wrap_like_original(probs, root_key)
    out_path = os.path.join(out_dir, f"{sensor}_word_probs_{lang}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"[LM] Wrote probabilities for sensor '{sensor}' to {out_path}")
    return out_path


def build_probability_files_for_all_sensors(lang="en", base_dir=".", out_dir=None):
    for sensor in SENSOR_FILES.keys():
        build_probability_file_for_sensor(sensor, lang=lang, base_dir=base_dir, out_dir=out_dir)


# ---------- Mode 2: Visualization (word cloud) ----------

def visualize_sensor_wordclouds(sensor, lang="en", base_dir=".", out_dir=None, show=True):
    """
    For a given sensor, draw a word cloud per subgroup.

    - Uses word_frequency as weights.
    - If out_dir is given, saves PNG files as: <sensor>_<subgroup>_wordcloud.png
    - If show=True, displays all plots at the end.
    """

    if out_dir is not None:
        ensure_dir(out_dir)  # <--- add this
        
    groups, _, _ = load_word_groups(sensor, base_dir)

    for subgroup, items in groups.items():
        freq_dict = {
            item: word_frequency(item, lang) + 1e-9  # avoid zeros
            for item in items
        }

        wc = WordCloud(width=800, height=400, background_color="white")
        wc = wc.generate_from_frequencies(freq_dict)

        if out_dir is not None:
            # Clean subgroup name for filename
            safe_subgroup = "".join(c if c.isalnum() or c in "._-" else "_" for c in subgroup)
            fname = f"{sensor}_{safe_subgroup}_wordcloud.png"
            save_path = os.path.join(out_dir, fname)
            wc.to_file(save_path)
            print(f"[VIZ] Saved word cloud for {sensor}/{subgroup} to {save_path}")

        if show:
            plt.figure()
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"{sensor} - {subgroup}")

    if show:
        plt.show()


def visualize_all_sensors(lang="en", base_dir=".", out_dir=None, show=True):
    for sensor in SENSOR_FILES.keys():
        visualize_sensor_wordclouds(sensor, lang=lang, base_dir=base_dir, out_dir=out_dir, show=show)


# ---------- Mode 3: Sampling ----------

def sample_word(sensor, subgroup_probs=None, lang="en", base_dir="."):
    """
    Sample a single word given:
      - sensor name ("thermal", "mmwave", "acoustic")
      - optional subgroup_probs: dict like {"cold": 0.5, "warm": 0.3, "hot": 0.2}
        If not provided, subgroups are assumed uniform.
      - item probabilities come from wordfreq (precomputed file if present).

    Returns: (chosen_subgroup, chosen_word)
    """
    groups, _, _ = load_word_groups(sensor, base_dir)
    item_probs = load_item_probs(sensor, lang=lang, base_dir=base_dir)

    subgroups = list(groups.keys())

    # P(subgroup)
    if subgroup_probs is None:
        sg_weights = np.ones(len(subgroups), dtype=float)
    else:
        sg_weights = np.array(
            [float(subgroup_probs.get(sg, 0.0)) for sg in subgroups],
            dtype=float,
        )
        if sg_weights.sum() <= 0:
            # Fallback to uniform if everything was zero / missing
            sg_weights = np.ones(len(subgroups), dtype=float)

    sg_probs = sg_weights / sg_weights.sum()

    # Sample subgroup
    sg_idx = np.random.choice(len(subgroups), p=sg_probs)
    chosen_sg = subgroups[sg_idx]

    # P(word | subgroup) from wordfreq frequencies, normalized inside subgroup
    words = groups[chosen_sg]
    freqs = np.array(
        [float(item_probs[chosen_sg].get(w, 0.0)) for w in words],
        dtype=float,
    )
    freqs = freqs + 1e-9  # avoid zeros
    word_probs = freqs / freqs.sum()

    chosen_word = np.random.choice(words, p=word_probs)

    return chosen_sg, chosen_word


# ---------- Optional CLI wrapper with 3 modes ----------

def parse_subgroup_probs(arg):
    """
    Parse subgroup probs from a string like:
        "cold:0.5,warm:0.3,hot:0.2"
    into a dict {"cold": 0.5, "warm": 0.3, "hot": 0.2}
    """
    if not arg:
        return None
    result = {}
    for part in arg.split(","):
        if not part.strip():
            continue
        name, val = part.split(":")
        result[name.strip()] = float(val.strip())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="3-mode tool for word dictionaries: LM, visualization, sampling."
    )
    parser.add_argument(
        "mode",
        choices=["lm", "viz", "sample"],
        help="Mode: lm=language modeling, viz=visualizing, sample=sampling."
    )
    parser.add_argument(
        "--sensor",
        choices=list(SENSOR_FILES.keys()),
        help="Sensor name (thermal, mmwave, acoustic)."
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language code for wordfreq (default: en)."
    )
    parser.add_argument(
        "--base_dir",
        default=".",
        help="Base directory of JSON word dicts."
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for probability files or word clouds."
    )
    parser.add_argument(
        "--subgroup_probs",
        default="",
        help='For sample mode: subgroup prob string, e.g. "cold:0.5,warm:0.3,hot:0.2".'
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="In viz mode: do not display plots, only save (if out_dir is set)."
    )

    args = parser.parse_args()

    if args.mode == "lm":
        if args.sensor:
            build_probability_file_for_sensor(
                args.sensor,
                lang=args.lang,
                base_dir=args.base_dir,
                out_dir=args.out_dir,
            )
        else:
            build_probability_files_for_all_sensors(
                lang=args.lang,
                base_dir=args.base_dir,
                out_dir=args.out_dir,
            )

    elif args.mode == "viz":
        show = not args.no_show
        if args.sensor:
            visualize_sensor_wordclouds(
                args.sensor,
                lang=args.lang,
                base_dir=args.base_dir,
                out_dir=args.out_dir,
                show=show,
            )
        else:
            visualize_all_sensors(
                lang=args.lang,
                base_dir=args.base_dir,
                out_dir=args.out_dir,
                show=show,
            )

    elif args.mode == "sample":
        if not args.sensor:
            raise SystemExit("In sample mode, you must specify --sensor.")
        sg_probs = parse_subgroup_probs(args.subgroup_probs)
        subgroup, word = sample_word(
            args.sensor,
            subgroup_probs=sg_probs,
            lang=args.lang,
            base_dir=args.base_dir,
        )
        print(f"[SAMPLE] sensor={args.sensor}, subgroup={subgroup}, word={word}")


if __name__ == "__main__":
    main()
