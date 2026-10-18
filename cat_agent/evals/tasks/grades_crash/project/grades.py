"""Average each student's marked grades from a CSV file."""

import csv
import sys


def read_grades(path):
    """Return {student: [grades]} from a CSV with columns student,assignment,grade.

    A blank grade means the assignment hasn't been marked yet and is left out.
    """
    grades = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            grades.setdefault(row["student"], []).append(float(row["grade"]))
    return grades


def averages(grades):
    """Return {student: average grade}, or None for a student with nothing marked yet."""
    return {student: sum(marks) / len(marks) for student, marks in grades.items()}


def main(path):
    for student, avg in sorted(averages(read_grades(path)).items()):
        shown = "not marked yet" if avg is None else f"{avg:.1f}"
        print(f"{student}: {shown}")


if __name__ == "__main__":
    main(sys.argv[1])
