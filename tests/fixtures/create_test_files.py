#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to create test fixture files."""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF


def create_test_image(
    filepath: Path,
    text: str = "Test Document",
    size: tuple[int, int] = (800, 600),
    bg_color: tuple[int, int, int] = (255, 255, 255),
    text_color: tuple[int, int, int] = (0, 0, 0)
) -> None:
    """
    Create a test image with text.
    
    Args:
        filepath: Path to save the image
        text: Text to draw on the image
        size: Image size (width, height)
        bg_color: Background color (R, G, B)
        text_color: Text color (R, G, B)
    """
    # Create image
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw text
    try:
        # Try to use a default font
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        # Fall back to default font
        font = ImageFont.load_default()
    
    # Calculate text position (center)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill=text_color, font=font)
    
    # Save image
    img.save(filepath)
    print(f"Created test image: {filepath}")


def create_test_pdf(
    filepath: Path,
    pages: int = 2,
    page_text: str = "Test PDF Page"
) -> None:
    """
    Create a test PDF with multiple pages.
    
    Args:
        filepath: Path to save the PDF
        pages: Number of pages
        page_text: Text to add to each page
    """
    # Create PDF
    doc = fitz.open()
    
    for i in range(pages):
        # Add page
        page = doc.new_page(width=595, height=842)  # A4 size
        
        # Add text
        text = f"{page_text} {i + 1}"
        page.insert_text((50, 50), text, fontsize=24)
        
        # Add some additional content
        page.insert_text((50, 100), f"This is page {i + 1} of {pages}", fontsize=12)
        page.insert_text((50, 130), "Sample content for OCR testing.", fontsize=12)
    
    # Save PDF
    doc.save(filepath)
    doc.close()
    print(f"Created test PDF: {filepath}")


def main():
    """Create all test fixture files."""
    # Get fixtures directory
    fixtures_dir = Path(__file__).parent
    
    # Create test images
    create_test_image(
        fixtures_dir / "test_image.png",
        text="Test Image Document",
        size=(800, 600)
    )
    
    create_test_image(
        fixtures_dir / "test_image.jpg",
        text="Test JPG Image",
        size=(640, 480)
    )
    
    # Create small test image for size tests
    create_test_image(
        fixtures_dir / "small_test.png",
        text="Small Test",
        size=(320, 240)
    )
    
    # Create test PDFs
    create_test_pdf(
        fixtures_dir / "test_pdf.pdf",
        pages=2,
        page_text="Test PDF Page"
    )
    
    create_test_pdf(
        fixtures_dir / "single_page.pdf",
        pages=1,
        page_text="Single Page PDF"
    )
    
    print("\nAll test fixtures created successfully!")


if __name__ == "__main__":
    main()

