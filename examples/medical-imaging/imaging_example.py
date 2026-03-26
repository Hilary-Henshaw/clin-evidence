"""Medical imaging upload and validation example."""
from __future__ import annotations

import argparse
import io
import struct
import sys
import uuid
import zlib
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"


def create_test_png(path: Path) -> None:
    """Create a minimal valid PNG file for testing."""
    width, height = 64, 64

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = len(data)
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return (
            struct.pack(">I", length)
            + chunk_type
            + data
            + struct.pack(">I", crc)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(
        ">IIBBBBB", width, height, 8, 2, 0, 0, 0
    )
    ihdr = png_chunk(b"IHDR", ihdr_data)

    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00"
        for _ in range(width):
            raw_rows += b"\x80\x80\x80"
    compressed = zlib.compress(raw_rows)
    idat = png_chunk(b"IDAT", compressed)
    iend = png_chunk(b"IEND", b"")

    path.write_bytes(signature + ihdr + idat + iend)


def upload_image(
    client: httpx.Client,
    image_path: Path,
) -> dict[str, object]:
    """Upload a medical image and return the analysis."""
    print(f"\nUploading image: {image_path.name}")

    with image_path.open("rb") as fh:
        files = {"file": (image_path.name, fh, "image/png")}
        response = client.post(
            f"{BASE_URL}/v1/upload",
            files=files,
            timeout=120.0,
        )

    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def display_upload_result(
    result: dict[str, object],
) -> None:
    """Display the image analysis result."""
    image_type = result.get("image_type", "UNKNOWN")
    confidence = float(result.get("confidence", 0.0))
    requires_val = result.get("requires_validation", False)

    print(f"Image type detected: {image_type}")
    print(f"Confidence: {confidence:.0%}")
    print(f"Requires validation: {'Yes' if requires_val else 'No'}")
    print("\nAnalysis:")
    print("-" * 40)
    print(result.get("analysis", ""))


def submit_validation(
    client: httpx.Client,
    session_id: str,
    approved: bool,
    feedback: str | None = None,
) -> dict[str, object]:
    """Submit validation decision for a pending analysis."""
    payload: dict[str, object] = {
        "session_id": session_id,
        "approved": approved,
    }
    if feedback:
        payload["feedback"] = feedback

    response = client.post(
        f"{BASE_URL}/v1/validate",
        json=payload,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def main() -> None:
    """Run the medical imaging example."""
    parser = argparse.ArgumentParser(
        description="ClinEvidence medical imaging example"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=Path,
        help="Path to medical image (PNG/JPG)",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Use a generated test PNG image",
    )
    args = parser.parse_args()

    print("ClinEvidence Medical Imaging Example")
    print("=" * 40)

    if args.demo:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test_scan.png"
            create_test_png(img_path)
            run_demo(img_path)
    else:
        img_path: Path = args.image
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            sys.exit(1)
        run_demo(img_path)


def run_demo(image_path: Path) -> None:
    """Execute the full upload + validation flow."""
    with httpx.Client() as client:
        try:
            result = upload_image(client, image_path)
            display_upload_result(result)

            session_id = str(result.get("session_id", ""))
            requires_val = result.get(
                "requires_validation", False
            )

            if requires_val and session_id:
                print("\n" + "-" * 40)
                print("Submitting validation: APPROVED")
                val_result = submit_validation(
                    client,
                    session_id=session_id,
                    approved=True,
                    feedback=(
                        "Reviewed by example script — "
                        "for demonstration only."
                    ),
                )
                print(
                    f"Status: {val_result.get('status')}"
                )
                print(
                    f"Message: {val_result.get('message')}"
                )

        except httpx.HTTPStatusError as exc:
            print(
                f"HTTP error {exc.response.status_code}: "
                f"{exc.response.text}"
            )
            sys.exit(1)
        except httpx.ConnectError:
            print(
                "Connection failed. Ensure ClinEvidence "
                f"is running at {BASE_URL}"
            )
            sys.exit(1)

    print("\nExample complete.")


if __name__ == "__main__":
    main()
