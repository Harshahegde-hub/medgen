import argparse
from rx_ocr import prescription_image_to_text

def main():
    parser = argparse.ArgumentParser(
        description="Medical Prescription Image → Text OCR"
    )
    parser.add_argument(
        "image",
        help="Path to prescription image"
    )

    args = parser.parse_args()

    try:
        result = prescription_image_to_text(args.image)

        print("\n--- Extracted Prescription Text ---\n")
        print(result["text"])

        if result["drugs"]:
            print("\n--- Detected Drugs ---")
            for drug in result["drugs"]:
                print(f"- {drug}")
        else:
            print("\nNo known drugs detected.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
