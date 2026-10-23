from pubmedclip_pipeline import DEFAULT_WEIGHTS_DIR, PubMedClipPipeline


def main() -> None:
    # Stages the pinned source, audits and converts it once (refused unless the result matches
    # the pinned digest), then strict-loads it. The Hub revision ships no model.safetensors.
    pipe = PubMedClipPipeline.from_pretrained(weights_dir=DEFAULT_WEIGHTS_DIR, allow_download=True)
    scores = pipe.zero_shot_classify(
        "figure.png",
        ["chest X-ray", "brain MRI", "abdominal CT scan"],
    )
    for item in scores:
        print(f"{item.label}: {item.score:.4f}")


if __name__ == "__main__":
    main()
