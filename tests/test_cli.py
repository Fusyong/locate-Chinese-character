from hanloc.cli import main


def test_help_lists_locate(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    text = capsys.readouterr().out
    assert "locate" in text
