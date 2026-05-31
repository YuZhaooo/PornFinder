import argparse
import heapq
import os
import shutil
from pathlib import Path

from tqdm import tqdm

# === KONFIGURATION ===
START_PFAD = Path("~/Documents/Wixvorlagen").expanduser()
ZIEL_PFAD = Path("~/Documents/Wixvorlagen/pornfinder_ergebnisse").expanduser()
MAX_TREFFER = 50
BILD_ENDUNGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# =====================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bilder per CLIP-Suche in Unterordnern finden."
    )
    parser.add_argument(
        "schwellenwert",
        type=float,
        help="Mindest-Wahrscheinlichkeit für Treffer (0–1), z. B. 0.25",
    )
    parser.add_argument(
        "suchbegriff",
        nargs="+",
        help='Suchbegriff, z. B. naked buttocks oder "naked buttocks"',
    )
    return parser.parse_args()


def bilder_sammeln(start_pfad):
    bild_pfade = []
    for ext in BILD_ENDUNGEN:
        bild_pfade.extend(start_pfad.rglob(f"*{ext}"))
        bild_pfade.extend(start_pfad.rglob(f"*{ext.upper()}"))
    return bild_pfade


def main():
    args = parse_args()
    schwellenwert = args.schwellenwert
    such_begriff = " ".join(args.suchbegriff)

    from PIL import Image
    import torch
    from transformers import CLIPModel, CLIPProcessor

    print("Lade CLIP-Modell von Hugging Face...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    os.makedirs(ZIEL_PFAD, exist_ok=True)

    bild_pfade = bilder_sammeln(START_PFAD)
    print(f"{len(bild_pfade)} Bilder in den Unterordnern gefunden.")
    print(f"Suche nach: '{such_begriff}' (Schwellenwert: {schwellenwert}, beste {MAX_TREFFER} Treffer)")

    top_treffer = []
    for index, pfad in enumerate(tqdm(bild_pfade)):
        try:
            image = Image.open(pfad).convert("RGB")
            inputs = processor(
                text=[such_begriff, "an image of something else"],
                images=image,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=-1)

            score = probs[0][0].item()

            if score <= schwellenwert:
                continue

            eintrag = (score, index, pfad)
            if len(top_treffer) < MAX_TREFFER:
                heapq.heappush(top_treffer, eintrag)
            elif score > top_treffer[0][0]:
                heapq.heapreplace(top_treffer, eintrag)

        except Exception:
            continue

    for score, _, pfad in sorted(top_treffer, reverse=True):
        ziel_datei = ZIEL_PFAD / f"{score:.2f}_{pfad.name}"
        shutil.copy2(pfad, ziel_datei)

    print(f"\nSuche abgeschlossen. {len(top_treffer)} beste Treffer nach '{ZIEL_PFAD}' kopiert.")


if __name__ == "__main__":
    main()
