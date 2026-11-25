"""Shared builders for test fixtures (documents)."""

import io


def make_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF with a real text layer."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(output)
    output += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        output += b"%010d 00000 n \n" % offset
    output += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return output


def make_docx() -> bytes:
    import docx

    document = docx.Document()
    document.add_heading("Hostel Rules", level=1)
    document.add_paragraph("Lights out at 11 pm on weekdays.")
    document.add_heading("Visitors", level=2)
    document.add_paragraph("Visitors are allowed until 7 pm.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Room", "Fee"
    table.cell(1, 0).text, table.cell(1, 1).text = "Single", "Rs 40,000"
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
