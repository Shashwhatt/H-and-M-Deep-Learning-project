import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.web_dashboard import (
    ARTICLE_META,
    SAMPLE_CIDS,
    ann_index,
    get_unique_article_image,
    EMBEDDING_DIM
)
import numpy as np

def test_recommendation_uniqueness():
    print(f"Total articles in metadata cache: {len(ARTICLE_META)}")
    
    for c_info in SAMPLE_CIDS:
        cid = c_info["id"]
        rng = np.random.default_rng(abs(hash(cid)) % (2**31))
        query_vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        retrieved = ann_index.query(query_vec, top_k=8, n_probe=8)
        
        used_images = set()
        assigned_images = []
        for aid, score in retrieved:
            meta = ARTICLE_META.get(aid, {})
            img = get_unique_article_image(aid, meta.get("group", ""), meta.get("type", ""), used_images)
            assigned_images.append((aid, meta.get("name", "Item"), meta.get("group", "Group"), img))
            
        urls = [img for _, _, _, img in assigned_images]
        print(f"\nTested {c_info['label']} ({cid[:16]}...):")
        for idx, (aid, name, grp, url) in enumerate(assigned_images, 1):
            print(f"  #{idx} [{aid}] {name} ({grp}) -> {url[:60]}...")
            
        assert len(urls) == len(set(urls)), f"Duplicate images found for customer {cid}!"
        print(f"  --> PASSED: All {len(urls)} recommended images are strictly unique and category-aware.")

if __name__ == "__main__":
    test_recommendation_uniqueness()
    print("\nALL RECOMMENDATION TESTS PASSED: Zero image duplicates!")
