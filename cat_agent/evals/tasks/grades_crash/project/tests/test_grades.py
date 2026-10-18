from grades import averages, read_grades


def test_reads_and_averages(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("student,assignment,grade\nAna,essay,70\nAna,quiz,90\nBen,essay,60\n")
    assert averages(read_grades(path)) == {"Ana": 80.0, "Ben": 60.0}
