from gridcast.report import replace_blocks


def test_generated_readme_blocks_are_replaced_and_the_rest_kept():
    text = "intro\n<!-- BEGIN:t -->\nold\nrows\n<!-- END:t -->\noutro\n<!-- BEGIN:u -->\nx\n<!-- END:u -->"
    out = replace_blocks(text, {"t": "| new |", "missing": "ignored"})
    assert (
        out
        == "intro\n<!-- BEGIN:t -->\n| new |\n<!-- END:t -->\noutro\n<!-- BEGIN:u -->\nx\n<!-- END:u -->"
    )
    assert replace_blocks(out, {"t": "| new |"}) == out  # idempotent


def test_empty_block_is_filled():
    text = "<!-- BEGIN:t -->\n<!-- END:t -->"
    assert replace_blocks(text, {"t": "a\nb"}) == "<!-- BEGIN:t -->\na\nb\n<!-- END:t -->"
