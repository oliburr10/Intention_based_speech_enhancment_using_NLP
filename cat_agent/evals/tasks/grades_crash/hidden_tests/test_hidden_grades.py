import pathlib

from grades import averages, main, read_grades

CLASS_CSV = pathlib.Path(__file__).parent.parent / "class.csv"


def test_blank_grades_are_left_out():
    grades = read_grades(CLASS_CSV)
    assert grades["Ana"] == [72.0, 88.0]
    assert grades["Ben"] == [65.0]


def test_student_with_nothing_marked_is_still_listed():
    grades = read_grades(CLASS_CSV)
    assert grades["Dana"] == []
    assert averages(grades)["Dana"] is None


def test_main_prints_everyone(capsys):
    main(CLASS_CSV)
    out = capsys.readouterr().out.splitlines()
    assert out == ["Ana: 80.0", "Ben: 65.0", "Cleo: 85.0", "Dana: not marked yet"]


def test_existing_behaviour_unchanged(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("student,assignment,grade\nAna,essay,70\nAna,quiz,90\n")
    assert averages(read_grades(path)) == {"Ana": 80.0}
