from pubmedclip_pipeline import load_pipeline


def main() -> None:
    pipe = load_pipeline()
    scores = pipe.zero_shot_classify(
        "figure.png",
        ["chest X-ray", "brain MRI", "abdominal CT scan"],
    )
    for item in scores:
        print(f"{item.label}: {item.score:.4f}")


if __name__ == "__main__":
    main()
