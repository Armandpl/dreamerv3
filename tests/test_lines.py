from pathlib import Path


def test_nb_lines():
    # count the number of lines in the project
    count = 0
    root_path = Path("minidream")
    for path in root_path.rglob("*"):
        if path.suffix != ".py":
            continue
        print(path)
        nb = len(open(path).readlines())
        count += nb

    assert count < 1000, f"yo wtf there are {count} lines in the project"
