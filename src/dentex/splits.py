"""Seeded, class-aware train/holdout splitting for multi-label detection images."""
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split


def rarest_class_key(image_ids, anns_by_image):
    """Stratification key per image: its rarest diagnosis class (-1 if no annotations)."""
    freq = Counter(a["category_id_3"] for iid in image_ids for a in anns_by_image.get(iid, []))
    keys = []
    for iid in image_ids:
        classes = {a["category_id_3"] for a in anns_by_image.get(iid, [])}
        keys.append(min(classes, key=lambda c: freq[c]) if classes else -1)
    return np.array(keys)


def stratified_holdout(image_ids, anns_by_image, holdout, seed=0):
    """Split ids into (keep, holdout); ``holdout`` is a count or a fraction."""
    image_ids = sorted(image_ids)
    keys = rarest_class_key(image_ids, anns_by_image)
    counts = Counter(keys)
    keys = np.array([k if counts[k] >= 2 else -2 for k in keys])  # merge singleton strata
    keep, held = train_test_split(image_ids, test_size=holdout, random_state=seed, stratify=keys)
    return sorted(keep), sorted(held)
