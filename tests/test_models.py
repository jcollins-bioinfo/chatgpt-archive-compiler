from chatgpt_archive_compiler.models import Archive, ContentBlock, ContentBlockType, SourceManifest


def test_minimal_archive_model() -> None:
    archive = Archive(source_manifest=SourceManifest())
    assert archive.archive_version == "1.0"
    assert archive.conversations == []


def test_content_block_model() -> None:
    block = ContentBlock(type=ContentBlockType.MARKDOWN, text="hello")
    assert block.text == "hello"
