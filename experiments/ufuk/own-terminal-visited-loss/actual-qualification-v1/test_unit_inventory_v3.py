"""Synthetic XML corruption controls; does not claim any executed CUDA test."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("unit8", Path(__file__).with_name("unit_owner_v3.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def xml():
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", tests="59", failures="0", errors="0", skipped="0")
    expected = []
    for i in range(59):
        name = f"test_case_{i}"
        module = f"test_module_{i}"
        ET.SubElement(suite, "testcase", classname=module if i < 58 else "", name=name)
        expected.append(f"{module}.py::{name}")
    return root, expected


def test_exact_frozen_identity_with_one_empty_external_classname():
    root, expected = xml()
    assert m.identities(root, expected) == expected


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate",
        "wrong_module",
        "wrong_name",
        "failure",
        "error",
        "skipped",
        "wrong_count",
        "ambiguous_external",
    ),
)
def test_corrupt_or_ambiguous_identity_rejected(mutation):
    root, expected = xml()
    suite = root.find("testsuite")
    first = suite[0]
    if mutation == "duplicate":
        suite[1].attrib = dict(first.attrib)
    elif mutation == "wrong_module":
        first.set("classname", "unexpected")
    elif mutation == "wrong_name":
        first.set("name", "unexpected")
    elif mutation in ("failure", "error", "skipped"):
        ET.SubElement(first, mutation)
    elif mutation == "wrong_count":
        suite.set("tests", "60")
    elif mutation == "ambiguous_external":
        expected[0] = "another.py::test_case_58"
    with pytest.raises((AssertionError, ValueError)):
        m.identities(root, expected)
